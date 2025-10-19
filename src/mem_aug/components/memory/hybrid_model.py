"""
Hybrid Transformer Model combining LM2's internal memory with LongMem's external retrieval.
Generic implementation that works with any transformer architecture (Llama, Qwen, GPT, etc.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from .internal_memory import InternalMemoryModule
from .external_memory import ExternalMemoryBank

# LoRA support removed - models should be prepared externally


class HybridTransformerConfig(PretrainedConfig):
    """Extended config with hybrid memory parameters for any transformer model."""
    
    model_type = "hybrid_transformer"
    
    def __init__(self, **kwargs):
        # Extract memory-specific parameters before calling super
        self.use_internal_memory = kwargs.pop("use_internal_memory", True)
        
        # Dynamic memory slot calculation based on hidden_dim
        hidden_dim = kwargs.get("hidden_size", 2048)
        default_memory_slots = hidden_dim // 128
        self.memory_slots = kwargs.pop("memory_slots", default_memory_slots)
        
        # Dynamic memory heads calculation
        default_num_mem_heads = max(1, self.memory_slots // 4)
        self.num_mem_heads = kwargs.pop("num_mem_heads", default_num_mem_heads)
        
        self.use_external_memory = kwargs.pop("use_external_memory", True)
        self.external_memory_size = kwargs.pop("external_memory_size", 1048576)
        self.retrieval_k = kwargs.pop("retrieval_k", 8)
        self.chunk_size = kwargs.pop("chunk_size", 4)
        self.use_gpu_to_search = kwargs.pop("use_gpu_to_search", True)
        
        # Learnable retrieval gate
        self.use_gated_retrieval = kwargs.pop("use_gated_retrieval", False)
        
        # Embedding layer path for fast text encoding (user-configurable)
        self.embedding_layer_path = kwargs.pop("embedding_layer_path", None)
        # Remove fixed batch size constraint for better parallelization
        self.max_batch_size = kwargs.pop("max_batch_size", 32)
        self.batch_size = kwargs.pop("batch_size", None)  # Allow dynamic
        self.log_freq = kwargs.pop("log_freq", 100)
        
        # Retrieval layer configuration
        self.num_retrieval_layers = kwargs.pop("num_retrieval_layers", None)
        
        # LoRA parameters removed - models should be prepared externally
        
        # Call parent init with remaining kwargs
        super().__init__(**kwargs)
    
    def get_retrieval_layers(self) -> list:
        """Calculate which layers should perform external memory retrieval."""
        total_layers = self.num_hidden_layers
        
        if self.num_retrieval_layers is None:
            # Default: L // 6 layers retrieve
            retrieval_layers = max(1, total_layers // 6)
        else:
            retrieval_layers = min(self.num_retrieval_layers, total_layers)
        
        return [int(k * total_layers / retrieval_layers) for k in range(retrieval_layers)]


class HybridMemoryAttention(nn.Module):
    """Attention layer with both internal and external memory."""
    
    def __init__(
        self,
        config: HybridTransformerConfig,
        self_attn: nn.Module,
        layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.self_attn = self_attn
        self.layer_idx = layer_idx
        
        # Internal memory module (LM2-style)
        if config.use_internal_memory:
            head_size = config.memory_slots // config.num_mem_heads
            self.internal_memory = InternalMemoryModule(
                mem_slots=config.memory_slots,
                head_size=head_size,
                hidden_dim=config.hidden_size,
                num_heads=config.num_mem_heads,
            )
        else:
            self.internal_memory = None
        
        # External memory will be managed at model level
        self.use_external_memory = config.use_external_memory
        
        # Joint attention for fusing retrieved memories
        if self.use_external_memory:
            self.num_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // self.num_heads
            
            # Projection for external memory fusion
            self.external_mem_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False
            )
        
        # Gating network for dynamic memory fusion
        # Projects hidden states to gate logits for each memory source
        # Maximum 3 sources: self-attention, internal memory, external memory
        self.gate_network = nn.Linear(
            config.hidden_size,
            3,  # Max number of memory sources
            bias=True
        )
        
        # Determine active sources at initialization
        self.active_sources = 1  # Always have self-attention
        if config.use_internal_memory:
            self.active_sources += 1
        if config.use_external_memory:
            self.active_sources += 1
        
        # Learnable retrieval gate (if enabled)
        use_gated_retrieval = getattr(config, 'use_gated_retrieval', False)
        if use_gated_retrieval and config.use_external_memory:
            self.g_retrieve = nn.Parameter(torch.tensor(0.5))
            self.use_gated_retrieval = True
            # Cache for previously retrieved context (reused when gate < threshold)
            self.cached_ext_context = None
        else:
            self.g_retrieve = None
            self.use_gated_retrieval = False
            self.cached_ext_context = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        internal_memory: Optional[torch.Tensor],
        external_kv: Optional[Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with hybrid memory using softmax-based gating.
        
        Implements: H_t = g[0] * A_t + g[1] * M_int + g[2] * M_ext
        where g = softmax(W_g @ h_t + b_g)
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            internal_memory: Internal memory state [batch, mem_slots, mem_slots]
            external_kv: Retrieved external memory keys/values
            attention_mask: Attention mask
            position_ids: Position IDs
            position_embeddings: Rotary position embeddings
        
        Returns:
            output: Gated combination of memory outputs
            updated_internal_memory: Updated internal memory state
        """
        # Standard self-attention (A_t)
        # Handle different model architectures
        try:
            # Try Llama-style (requires position_embeddings and attention_mask)
            if position_embeddings is not None:
                attn_result = self.self_attn(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask
                )
            else:
                # Try GPT2-style (just hidden_states and optional attention_mask)
                attn_result = self.self_attn(hidden_states, attention_mask=attention_mask)
        except TypeError:
            # Fallback: just hidden_states
            attn_result = self.self_attn(hidden_states)
        # Handle both return formats
        if isinstance(attn_result, tuple):
            attn_output = attn_result[0]
        else:
            attn_output = attn_result
        
        # Collect active memory sources
        memory_outputs = [attn_output]  # Always include self-attention
        
        # Internal memory (M_int)
        updated_internal_memory = None
        if self.internal_memory is not None and internal_memory is not None:
            internal_mem_output, updated_internal_memory = self.internal_memory(
                attn_output,
                internal_memory,
                attention_mask
            )
            memory_outputs.append(internal_mem_output)
        
        # External memory (M_ext) with gating and context caching
        ext_out = None
        if self.use_external_memory and external_kv is not None:
            if self.use_gated_retrieval and self.g_retrieve is not None:
                # Compute gate value
                gate = torch.sigmoid(self.g_retrieve)
                
                # Check against threshold
                if self.training:
                    # During training: always retrieve but weight the output
                    ext_context = self._compute_external_memory(
                        attn_output,
                        external_kv,
                        attention_mask
                    )
                    ext_out = gate * ext_context
                    # Cache the retrieved context for potential reuse
                    self.cached_ext_context = ext_context.detach()
                else:
                    # During inference: adaptive retrieval with caching
                    threshold = getattr(self.config, 'retrieval_threshold', 0.5)
                    if gate.item() > threshold:
                        # Gate above threshold: retrieve fresh context
                        ext_out = self._compute_external_memory(
                            attn_output,
                            external_kv,
                            attention_mask
                        )
                        # Update cache with fresh retrieval
                        self.cached_ext_context = ext_out.detach()
                    else:
                        # Gate below threshold: reuse cached context if available
                        if self.cached_ext_context is not None:
                            # Reuse previous retrieval (model learns when context is still valid)
                            ext_out = self.cached_ext_context
                        # else: ext_out remains None (no cache available yet)
            else:
                # Standard retrieval without gating
                ext_out = self._compute_external_memory(
                    attn_output,
                    external_kv,
                    attention_mask
                )
            
            if ext_out is not None:
                memory_outputs.append(ext_out)
        
        # Compute gating weights using softmax
        # g = softmax(W_g @ h_t + b_g)  shape: [batch, seq_len, num_sources]
        gate_logits = self.gate_network(hidden_states)  # [batch, seq_len, 3]
        
        # Only use gates for active sources (determined at init)
        gate_logits = gate_logits[:, :, :self.active_sources]  # [batch, seq_len, active_sources]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [batch, seq_len, active_sources]
        
        # Combine memory outputs using gating weights
        # H_t = sum(g[i] * M[i]) for i in active_sources
        combined_output = torch.zeros_like(attn_output)
        for i, mem_output in enumerate(memory_outputs):
            # gate_weights[:, :, i:i+1] has shape [batch, seq_len, 1]
            # mem_output has shape [batch, seq_len, hidden_dim]
            combined_output = combined_output + gate_weights[:, :, i:i+1] * mem_output
        
        return combined_output, updated_internal_memory
    
    def log_gate_value(self, step: int) -> Optional[float]:
        """Log retrieval gate value for debugging."""
        if self.use_gated_retrieval and self.g_retrieve is not None:
            gate_value = torch.sigmoid(self.g_retrieve).item()
            if step % 1000 == 0:
                print(f"Layer {self.layer_idx} - Step {step}: Retrieval gate = {gate_value:.4f}")
            return gate_value
        return None
    
    def _compute_external_memory(
        self,
        hidden_states: torch.Tensor,
        external_kv: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute external memory output using joint attention.
        Returns the memory output without residual connection (for gating).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape for multi-head attention
        query = hidden_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # External keys and values: [batch*num_heads, seq_len, k, head_dim]
        ext_keys = external_kv['k']
        ext_values = external_kv['v']
        
        # Reshape external memories
        ext_keys = ext_keys.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )  # [batch, num_heads, seq_len, k, head_dim]
        ext_values = ext_values.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )
        
        # Compute attention scores with external memories
        scores = torch.einsum(
            'bhqd,bhqkd->bhqk',
            query,
            ext_keys
        ) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of external values
        external_context = torch.einsum(
            'bhqk,bhqkd->bhqd',
            attn_weights,
            ext_values
        )
        
        # Reshape and project
        external_context = external_context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        external_context = self.external_mem_proj(external_context)
        
        return external_context
    
    def _fuse_external_memory(
        self,
        hidden_states: torch.Tensor,
        external_kv: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse retrieved external memories using joint attention.
        Legacy method - kept for backward compatibility.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape for multi-head attention
        query = hidden_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # External keys and values: [batch*num_heads, seq_len, k, head_dim]
        ext_keys = external_kv['k']
        ext_values = external_kv['v']
        
        # Reshape external memories
        ext_keys = ext_keys.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )  # [batch, num_heads, seq_len, k, head_dim]
        ext_values = ext_values.view(
            batch_size, self.num_heads, seq_len, -1, self.head_dim
        )
        
        # Compute attention scores with external memories
        scores = torch.einsum(
            'bhqd,bhqkd->bhqk',
            query,
            ext_keys
        ) / (self.head_dim ** 0.5)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of external values
        external_context = torch.einsum(
            'bhqk,bhqkd->bhqd',
            attn_weights,
            ext_values
        )
        
        # Reshape and project
        external_context = external_context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        external_context = self.external_mem_proj(external_context)
        
        # Residual connection
        return hidden_states + external_context


class HybridTransformerModel(PreTrainedModel):
    """
    Hybrid Transformer model with dual memory systems.
    Works with any transformer architecture (Llama, Qwen, GPT, etc.)
    """
    
    config_class = HybridTransformerConfig
    supports_gradient_checkpointing = True
    
    def __init__(self, config: HybridTransformerConfig, tokenizer: AutoTokenizer, base_model=None):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer
        
        if base_model is not None:
            # Use provided model (could be LoRA-enhanced or any other model)
            self.full_model = base_model  # Keep reference to full model for forward pass
            
            # Extract transformer and lm_head components based on architecture
            # Handle PEFT wrapped models
            if hasattr(base_model, 'base_model') and hasattr(base_model.base_model, 'model'):
                # PEFT wrapped model: PeftModel -> LoraModel -> GPT2LMHeadModel -> GPT2Model
                peft_base = base_model.base_model.model  # This is GPT2LMHeadModel
                if hasattr(peft_base, 'transformer'):
                    self.model = peft_base.transformer  # GPT2Model
                    self.lm_head = peft_base.lm_head
                elif hasattr(peft_base, 'model'):
                    self.model = peft_base.model
                    self.lm_head = peft_base.lm_head
                else:
                    self.model = peft_base
                    self.lm_head = None
            elif hasattr(base_model, 'model'):
                # For models like GPT2LMHeadModel -> GPT2Model
                self.model = base_model.model
                self.lm_head = base_model.lm_head
            elif hasattr(base_model, 'transformer'):
                # For some older models
                self.model = base_model.transformer
                self.lm_head = getattr(base_model, 'lm_head', None)
            else:
                # Direct transformer model
                self.model = base_model
                self.lm_head = None
                self.full_model = base_model
        else:
            # Create base config without memory parameters
            base_config = AutoConfig.for_model(
                model_type=config.model_type if hasattr(config, 'model_type') and config.model_type != 'hybrid_transformer' else 'llama',
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
                rope_theta=getattr(config, 'rope_theta', 10000.0),
                attention_dropout=getattr(config, 'attention_dropout', 0.0),
            )
            
            # Load base model from config
            base_model = AutoModelForCausalLM.from_config(base_config)
            self.full_model = base_model
            self.model = base_model.model  # Get the base transformer
            self.lm_head = base_model.lm_head
        
        # Initialize internal memory with dynamic batch support
        if config.use_internal_memory:
            # Create template memory that can be expanded dynamically
            self.memory_template = torch.eye(config.memory_slots, requires_grad=False)
            self.register_buffer("memory_template_buffer", self.memory_template)
            self.internal_memory = None  # Will be created dynamically
        else:
            self.internal_memory = None
        
        # Initialize external memory bank
        if config.use_external_memory:
            self.external_memory = ExternalMemoryBank(
                config, 
                model=self, 
                tokenizer=tokenizer,
                embedding_layer_path=config.embedding_layer_path
            )
        else:
            self.external_memory = None
        
        # Wrap decoder layers with hybrid memory (if they exist)
        if hasattr(self.model, 'layers') or hasattr(self.model, 'h'):
            self._wrap_decoder_layers()
        
        # Initialize gradient checkpointing state
        self.gradient_checkpointing = False
    
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        # Count trainable parameters across the entire model
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        # Check for PEFT in the original full model
        has_peft = False
        peft_info = {}
        
        if hasattr(self, 'full_model'):
            # Check if full_model is a PEFT model
            if hasattr(self.full_model, 'peft_config') and self.full_model.peft_config:
                has_peft = True
                peft_info["peft_type"] = "PEFT_MODEL"
                peft_info["active_adapters"] = list(self.full_model.peft_config.keys())
            elif hasattr(self.full_model, 'base_model') and hasattr(self.full_model.base_model, 'peft_config'):
                has_peft = True
                peft_info["peft_type"] = "PEFT_MODEL"
                peft_info["active_adapters"] = list(self.full_model.base_model.peft_config.keys())
        
        info = {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
            "has_peft": has_peft,
        }
        
        info.update(peft_info)
        return info
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the model."""
        self.gradient_checkpointing = True
        
        # Enable on base model as well
        if hasattr(self, 'full_model') and hasattr(self.full_model, 'gradient_checkpointing_enable'):
            try:
                self.full_model.gradient_checkpointing_enable()
            except Exception:
                pass
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the model."""
        self.gradient_checkpointing = False
        
        # Disable on base model as well
        if hasattr(self, 'full_model') and hasattr(self.full_model, 'gradient_checkpointing_disable'):
            try:
                self.full_model.gradient_checkpointing_disable()
            except Exception:
                pass
    
    def _wrap_decoder_layers(self):
        """Wrap existing decoder layers with hybrid memory attention."""
        # Get layers based on architecture
        if hasattr(self.model, 'layers'):
            original_layers = self.model.layers
            layer_attr = 'layers'
        elif hasattr(self.model, 'h'):  # GPT-style
            original_layers = self.model.h
            layer_attr = 'h'
        else:
            raise ValueError(f"Cannot find layers in model: {type(self.model)}")
        
        class HybridDecoderLayer(nn.Module):
            def __init__(self, original_layer, config, layer_idx):
                super().__init__()
                self.original_layer = original_layer
                
                # Handle different layer architectures
                if hasattr(original_layer, 'self_attn'):
                    # Llama/standard architecture
                    attn_layer = original_layer.self_attn
                    self.input_layernorm = original_layer.input_layernorm
                    self.post_attention_layernorm = original_layer.post_attention_layernorm
                elif hasattr(original_layer, 'attn'):
                    # GPT2 architecture
                    attn_layer = original_layer.attn
                    self.input_layernorm = original_layer.ln_1
                    self.post_attention_layernorm = original_layer.ln_2
                else:
                    raise ValueError(f"Cannot find attention layer in {type(original_layer)}")
                
                self.hybrid_attn = HybridMemoryAttention(
                    config,
                    attn_layer,
                    layer_idx
                )
                self.mlp = original_layer.mlp
            
            def forward(
                self,
                hidden_states,
                attention_mask=None,
                position_ids=None,
                internal_memory=None,
                external_kv=None,
                position_embeddings=None,
                **kwargs  # Accept additional arguments from generate
            ):
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
                
                # Hybrid memory attention
                attn_output, updated_internal_memory = self.hybrid_attn(
                    hidden_states=hidden_states,
                    internal_memory=internal_memory,
                    external_kv=external_kv,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                
                hidden_states = residual + attn_output
                
                # Feed-forward network
                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                
                return hidden_states, updated_internal_memory
        
        # Replace layers
        setattr(self.model, layer_attr, nn.ModuleList([
            HybridDecoderLayer(layer, self.config, idx)
            for idx, layer in enumerate(original_layers)
        ]))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_external_memory: bool = True,
        output_hidden_states: bool = False,
        **kwargs
    ):
        """
        Forward pass with hybrid memory.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            targets: Target token IDs for loss calculation
            attention_mask: Attention mask
            position_ids: Position IDs
            use_external_memory: Whether to use external memory retrieval
            output_hidden_states: Whether to return hidden states
        
        Returns:
            Output object with logits, loss, memory_info, and optionally hidden_states
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        
        # Create internal memory dynamically based on actual batch size
        if hasattr(self, 'memory_template'):
            # Create memory for current batch size
            internal_memory = self.memory_template.unsqueeze(0).expand(
                batch_size, -1, -1
            ).contiguous().to(device)
        elif self.internal_memory is not None:
            # Fallback to old method
            if self.internal_memory.device != device:
                self.internal_memory = self.internal_memory.to(device)
            internal_memory = self.internal_memory.detach()
        else:
            internal_memory = None
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=device
            ).unsqueeze(0).repeat(batch_size, 1)
        
        # Get embeddings - handle different model architectures
        # The model here is the transformer part (e.g., GPT2Model), not the LMHead wrapper
        if hasattr(self.model, 'embed_tokens'):
            inputs_embeds = self.model.embed_tokens(input_ids)
        elif hasattr(self.model, 'wte'):  # GPT-style models
            inputs_embeds = self.model.wte(input_ids)
        elif hasattr(self.model, 'embeddings'):  # BERT-style models
            inputs_embeds = self.model.embeddings.word_embeddings(input_ids)
        else:
            # Try to find embedding layer recursively
            embedding_layer = None
            for name, module in self.model.named_modules():
                if ('embed' in name.lower() or 'wte' in name.lower()) and hasattr(module, 'weight'):
                    try:
                        # Test if this is an embedding layer by checking forward
                        test_output = module(input_ids[:1, :1])  # Test with small input
                        if len(test_output.shape) == 3:  # [batch, seq, hidden]
                            embedding_layer = module
                            break
                    except:
                        continue
            
            if embedding_layer is not None:
                inputs_embeds = embedding_layer(input_ids)
            else:
                # Ultimate fallback: use the lm_head weight transposed as embedding
                if self.lm_head is not None and hasattr(self.lm_head, 'weight'):
                    # Create embedding layer from lm_head
                    vocab_size, hidden_size = self.lm_head.weight.shape
                    embedding_weight = self.lm_head.weight.t()  # Transpose to [hidden, vocab]
                    inputs_embeds = torch.nn.functional.embedding(input_ids, embedding_weight.t())
                else:
                    raise ValueError(f"Cannot find embedding layer in model architecture: {type(self.model)}")
        
        hidden_states = inputs_embeds
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create 4D causal mask
        causal_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_len), inputs_embeds, 0
        )
        
        # Get position embeddings - handle different model architectures
        if hasattr(self.model, 'rotary_emb'):
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        else:
            # For models without rotary embeddings, skip position embeddings
            position_embeddings = None
        
        # Get retrieval layers
        retrieval_layers = set(self.config.get_retrieval_layers())
        
        # Collect hidden states if requested
        all_hidden_states = []
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Process through decoder layers - handle different architectures
        if hasattr(self.model, 'layers'):
            layers = self.model.layers
        elif hasattr(self.model, 'h'):  # GPT-style models
            layers = self.model.h
        else:
            raise ValueError(f"Cannot find transformer layers in model: {type(self.model)}")
        
        for layer_idx, decoder_layer in enumerate(layers):
            # Retrieve from external memory only at specified layers
            external_kv = None
            if (use_external_memory and 
                self.external_memory is not None and
                self.external_memory.is_ready() and
                layer_idx in retrieval_layers):
                
                # Get queries for retrieval (use hidden states)
                queries = hidden_states.transpose(0, 1)  # [seq_len, batch, hidden]
                external_kv = self.external_memory.retrieve(queries)
            
            # Forward through layer with optional gradient checkpointing
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, internal_memory = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    internal_memory,
                    external_kv,
                    position_embeddings,
                    use_reentrant=False
                )
            else:
                hidden_states, internal_memory = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    internal_memory=internal_memory,
                    external_kv=external_kv,
                    position_embeddings=position_embeddings,
                )
            
            # Collect hidden states if requested
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        
        # Final layer norm and output projection - handle different architectures
        if hasattr(self.model, 'norm'):
            hidden_states = self.model.norm(hidden_states)
        elif hasattr(self.model, 'ln_f'):  # GPT2 uses ln_f
            hidden_states = self.model.ln_f(hidden_states)
        # Some models don't have final norm
        
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states).float()
        else:
            # If no lm_head, assume hidden_states are logits
            logits = hidden_states.float()
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Use standard ignore_index for cross entropy
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = targets.view(-1)
            
            # Check if we have valid targets (not all -100)
            valid_targets = flat_targets != -100
            if valid_targets.sum() > 0:
                loss = F.cross_entropy(
                    flat_logits,
                    flat_targets,
                    ignore_index=-100,
                )
            else:
                # If no valid targets, create a dummy loss
                loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
        
        # Update internal memory
        if self.internal_memory is not None:
            self.internal_memory = internal_memory
            self.internal_memory_bank = internal_memory.clone()
        
        # Prepare memory info
        memory_info = {
            'internal_memory': internal_memory,
            'external_memory_size': (
                self.external_memory.get_size()
                if self.external_memory is not None else 0
            ),
        }
        
        # Create output object similar to HuggingFace models
        class ModelOutput:
            def __init__(self, logits, loss=None, memory_info=None, hidden_states=None):
                self.logits = logits
                self.loss = loss
                self.memory_info = memory_info
                self.hidden_states = hidden_states
                self.last_hidden_state = hidden_states[-1] if hidden_states else None
        
        # Return based on requested outputs
        if output_hidden_states:
            return ModelOutput(logits, loss, memory_info, tuple(all_hidden_states))
        else:
            return logits, loss, memory_info
    
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Create causal attention mask compatible with different transformers versions."""
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = self._expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None 
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, 
        past_key_values_length: int = 0
    ):
        """Make causal mask for autoregressive generation."""
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), 
                mask
            ], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
        """Expand attention mask from 2D to 4D."""
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )
    
    def generate(self, *args, **kwargs):
        """
        Generate text using the underlying model's generate method.
        Delegates to the full_model (which may be PEFT-enhanced).
        """
        if hasattr(self, 'full_model') and hasattr(self.full_model, 'generate'):
            return self.full_model.generate(*args, **kwargs)
        else:
            raise NotImplementedError(
                "Generate method not available. The underlying model doesn't support generation."
            )
    
    def configure_optimizers(self, train_config):
        """Configure optimizer with proper parameter groups."""
        import re
        
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
        )
        
        # Add RMSNorm if it exists
        try:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            blacklist_weight_modules = blacklist_weight_modules + (LlamaRMSNorm,)
        except:
            pass
        
        # Pattern for internal memory parameters
        pattern = re.compile(
            r"^model\.layers\.\d+\.hybrid_attn\.internal_memory\."
            r"input_gate_projector\.w$"
        )
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                
                # Skip lm_head parameters from weight decay categorization
                # They will be added to decay group by default
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pattern.match(fpn):
                    no_decay.add(fpn)
        
        # Validate parameter separation
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        # Add any remaining parameters (like lm_head) to decay group
        remaining_params = param_dict.keys() - union_params
        if remaining_params:
            decay.update(remaining_params)
            union_params = decay | no_decay
        
        assert len(inter_params) == 0, (
            f"Parameters in both decay/no_decay: {inter_params}"
        )
        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters not in either set: {param_dict.keys() - union_params}"
        )
        
        # Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas
        )
        return optimizer
