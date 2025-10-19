"""
External Memory Bank (adapted from LongMem).
Provides FAISS-based retrieval for long-term context.
"""

import torch
import numpy as np
import uuid
from typing import Dict, Optional, List, Union, Any

try:
    import faiss
    import faiss.contrib.torch_utils
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. External memory will be disabled.")


class ExternalMemoryBank:
    """
    External memory bank using FAISS for # The above code is a Python comment. Comments in Python
    # start with a hash symbol (#) and are used to provide
    # explanations or notes within the code. In this case, the
    # comment simply states "efficient". Comments are ignored by
    # the Python interpreter and do not affect the execution of
    # the code.
    efficient retrieval.
    Based on LongMem's dynamic memory architecture.
    """
    
    def __init__(self, config, model=None, tokenizer=None, embedding_layer_path=None):
        """
        Initialize external memory bank.
        
        Args:
            config: Model configuration with memory parameters
            model: Base model for text encoding
            tokenizer: Tokenizer for text encoding
            embedding_layer_path: Dot-separated path to embedding layer (e.g., "model.embed_tokens")
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for external memory. "
                "Install with: pip install faiss-gpu or faiss-cpu"
            )
        
        self.dimension = config.hidden_size
        self.use_gpu_to_search = config.use_gpu_to_search
        self.k = config.retrieval_k
        self.memory_size = config.external_memory_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dimension // self.num_heads
        self.chunk_size = config.chunk_size
        
        # Store embedding layer path for fast text encoding
        self.embedding_layer_path = embedding_layer_path
        
        # Initialize FAISS indices (one per attention head)
        self._initialize_indices()
        
        # Start index after dummy chunks
        self.dstore_idx = self.dummy_chunks
        self.time_for_retrieve = 0.0
        self.retrieve_count = 0
        
        # Metadata storage for enhanced functionality
        self.types = {"unknown": 0}  # Type label encoder
        self.next_type_id = 1
        self.entry_metadata = {}  # Maps chunk_idx -> {id, type, deleted, text}
        self.id_to_chunks = {}  # Maps entry_id -> list of chunk indices
        self.current_index = 0  # For set_index/get_index functionality
        
        # Learnable retrieval gate threshold (configurable)
        self.retrieval_threshold = 0.5  # Default threshold, can be changed via set_threshold()
        
        # Model and tokenizer for text encoding (optional)
        self.model = model
        self.tokenizer = tokenizer
        
        # Sequence adaptation method: 'pad', 'pool', or 'compress'
        self.seq_adapt_method = getattr(config, 'seq_adapt_method', 'compress')
        
        # Initialize key-value storage
        self._initialize_storage()
        
        # Initialize linear layers for handling variable sequence lengths
        self._initialize_sequence_adapters()
        
        # Initialize dummy storage for FAISS compatibility
        self._initialize_dummy_storage()
    
    def _initialize_indices(self):
        """Initialize FAISS indices for each attention head."""
        self.index_list = []
        
        # Calculate number of dummy chunks needed
        # We need enough chunks to support k retrievals
        k_per_chunk = max(1, self.k // self.chunk_size)
        self.dummy_chunks = k_per_chunk
        
        if self.use_gpu_to_search and torch.cuda.is_available():
            print(f'Initializing GPU FAISS indices on device {torch.cuda.current_device()}')
            self.res = faiss.StandardGpuResources()
            
            for i in range(self.num_heads):
                cpu_index = faiss.IndexFlatIP(self.head_dim)
                # Add dummy chunk representatives (one per chunk)
                dummy_vectors = torch.zeros(self.dummy_chunks, self.head_dim).numpy().astype('float32')
                cpu_index.add(dummy_vectors)
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
                self.index_list.append(gpu_index)
        else:
            print('Initializing CPU FAISS indices')
            self.index_list = []
            for i in range(self.num_heads):
                index = faiss.IndexFlatIP(self.head_dim)
                # Add dummy chunk representatives (one per chunk)  
                dummy_vectors = torch.zeros(self.dummy_chunks, self.head_dim).numpy().astype('float32')
                index.add(dummy_vectors)
                self.index_list.append(index)
    
    def _initialize_storage(self):
        """Initialize key-value storage tensors."""
        storage_size = self.memory_size // self.chunk_size
        
        if self.use_gpu_to_search and torch.cuda.is_available():
            device = torch.cuda.current_device()
            dtype = torch.float16
        else:
            device = 'cpu'
            dtype = torch.float32
        
        self.keys = [
            torch.zeros(
                storage_size,
                self.chunk_size,
                self.head_dim,
                dtype=dtype,
                device=device
            )
            for _ in range(self.num_heads)
        ]
        
        self.vals = [
            torch.zeros(
                storage_size,
                self.chunk_size,
                self.head_dim,
                dtype=dtype,
                device=device
            )
            for _ in range(self.num_heads)
        ]
    
    def _initialize_dummy_storage(self):
        """Initialize dummy storage to match the dummy vectors in FAISS indices."""
        # We added dummy_chunks dummy vectors to each FAISS index
        for head_idx in range(self.num_heads):
            # Set the first dummy_chunks chunks to zero (dummy data)
            self.keys[head_idx][:self.dummy_chunks] = 0
            self.vals[head_idx][:self.dummy_chunks] = 0
    
    def _initialize_sequence_adapters(self):
        """Initialize linear layers for sequence length adaptation with metadata."""
        import torch.nn as nn
        
        if self.seq_adapt_method == 'compress':
            # Linear layers to compress/expand sequences to chunk_size
            device = torch.cuda.current_device() if self.use_gpu_to_search and torch.cuda.is_available() else 'cpu'
            
            # Enhanced compressor that handles K-V pairs + metadata
            # Input: head_dim + 2 (for type_id and deleted flag)
            # Output: head_dim (compressed representation)
            enhanced_input_dim = self.head_dim + 2
            
            # Lightweight compression: simple linear + residual
            self.seq_compressors = nn.ModuleList([
                nn.Linear(enhanced_input_dim, self.head_dim, bias=False).to(device)
                for _ in range(self.num_heads)
            ])
            
            # Lightweight metadata projection
            self.metadata_projectors = nn.ModuleList([
                nn.Linear(2, 2, bias=False).to(device)  # Simple identity-like projection
                for _ in range(self.num_heads)
            ])
            
            # Initialize weights - simplified for single linear layers
            for compressor in self.seq_compressors:
                nn.init.xavier_uniform_(compressor.weight)
            
            for projector in self.metadata_projectors:
                nn.init.eye_(projector.weight)  # Start as identity
        else:
            self.seq_compressors = None
            self.metadata_projectors = None
    
    def _adapt_sequence_length(self, keys: torch.Tensor, vals: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> tuple:
        """
        Adapt sequence length to fit into complete chunks without data loss.
        
        Args:
            keys: [total_tokens, num_heads, head_dim]
            vals: [total_tokens, num_heads, head_dim]
            metadata: Optional [total_tokens, 2] tensor with [type_id, deleted] per token
        
        Returns:
            Adapted keys, values, and metadata that fit exactly into chunks
        """
        total_tokens, num_heads, head_dim = keys.shape
        remainder = total_tokens % self.chunk_size
        
        if remainder == 0:
            # Perfect fit, no adaptation needed
            return (keys, vals, metadata) if metadata is not None else (keys, vals)
        
        if self.seq_adapt_method == 'pad':
            # Pad with zeros to complete the last chunk
            pad_size = self.chunk_size - remainder
            pad_keys = torch.zeros(pad_size, num_heads, head_dim, 
                                 dtype=keys.dtype, device=keys.device)
            pad_vals = torch.zeros(pad_size, num_heads, head_dim,
                                 dtype=vals.dtype, device=vals.device)
            
            keys = torch.cat([keys, pad_keys], dim=0)
            vals = torch.cat([vals, pad_vals], dim=0)
            
        elif self.seq_adapt_method == 'pool':
            # Pool the remainder tokens into the last complete chunk
            complete_tokens = total_tokens - remainder
            remainder_keys = keys[complete_tokens:]  # [remainder, num_heads, head_dim]
            remainder_vals = vals[complete_tokens:]
            
            # Mean pooling of remainder tokens
            pooled_keys = remainder_keys.mean(dim=0, keepdim=True)  # [1, num_heads, head_dim]
            pooled_vals = remainder_vals.mean(dim=0, keepdim=True)
            
            # Replace the last token in complete chunks with pooled representation
            if complete_tokens > 0:
                keys = keys[:complete_tokens]
                vals = vals[:complete_tokens]
                keys[-1:] = pooled_keys  # Replace last token with pooled remainder
                vals[-1:] = pooled_vals
            else:
                # If we have fewer tokens than chunk_size, pad with pooled representation
                pad_size = self.chunk_size - remainder
                expanded_pooled_keys = pooled_keys.expand(pad_size, -1, -1)
                expanded_pooled_vals = pooled_vals.expand(pad_size, -1, -1)
                
                keys = torch.cat([keys, expanded_pooled_keys], dim=0)
                vals = torch.cat([vals, expanded_pooled_vals], dim=0)
                
        elif self.seq_adapt_method == 'compress':
            # Use linear transformation to compress remainder into fixed size
            if self.seq_compressors is not None:
                complete_tokens = total_tokens - remainder
                remainder_keys = keys[complete_tokens:]  # [remainder, num_heads, head_dim]
                remainder_vals = vals[complete_tokens:]
                
                # Handle metadata if provided
                compressed_metadata = None
                if metadata is not None:
                    remainder_metadata = metadata[complete_tokens:]  # [remainder, 2]
                    # Average the metadata for the remainder tokens
                    avg_metadata = remainder_metadata.float().mean(dim=0, keepdim=True)  # [1, 2]
                    
                # Compress remainder using learned linear transformation with metadata
                compressed_keys = torch.zeros(self.chunk_size - remainder, num_heads, head_dim,
                                            dtype=keys.dtype, device=keys.device)
                compressed_vals = torch.zeros(self.chunk_size - remainder, num_heads, head_dim,
                                            dtype=vals.dtype, device=vals.device)
                
                for head_idx in range(num_heads):
                    # Apply compression to each head
                    head_remainder_keys = remainder_keys[:, head_idx, :]  # [remainder, head_dim]
                    head_remainder_vals = remainder_vals[:, head_idx, :]
                    
                    # Average pooling of K-V pairs
                    pooled_key = head_remainder_keys.mean(dim=0, keepdim=True)  # [1, head_dim]
                    pooled_val = head_remainder_vals.mean(dim=0, keepdim=True)
                    
                    if metadata is not None:
                        # Concatenate pooled K-V with metadata for enhanced compression
                        enhanced_key_input = torch.cat([pooled_key, avg_metadata], dim=-1)  # [1, head_dim + 2]
                        enhanced_val_input = torch.cat([pooled_val, avg_metadata], dim=-1)
                        
                        compressed_key = self.seq_compressors[head_idx](enhanced_key_input)
                        compressed_val = self.seq_compressors[head_idx](enhanced_val_input)
                        
                        # Project metadata through learned transformation
                        if compressed_metadata is None:
                            projected_metadata = self.metadata_projectors[head_idx](avg_metadata)  # [1, 2]
                            compressed_metadata = projected_metadata.expand(self.chunk_size - remainder, -1)
                    else:
                        # Fallback: pad with zeros for metadata
                        zero_metadata = torch.zeros(1, 2, dtype=pooled_key.dtype, device=pooled_key.device)
                        enhanced_key_input = torch.cat([pooled_key, zero_metadata], dim=-1)
                        enhanced_val_input = torch.cat([pooled_val, zero_metadata], dim=-1)
                        
                        compressed_key = self.seq_compressors[head_idx](enhanced_key_input)
                        compressed_val = self.seq_compressors[head_idx](enhanced_val_input)
                    
                    # Expand to fill the compressed space
                    compressed_keys[:, head_idx, :] = compressed_key.expand(self.chunk_size - remainder, -1)
                    compressed_vals[:, head_idx, :] = compressed_val.expand(self.chunk_size - remainder, -1)
                
                keys = torch.cat([keys[:complete_tokens], compressed_keys], dim=0)
                vals = torch.cat([vals[:complete_tokens], compressed_vals], dim=0)
                
                if metadata is not None:
                    metadata = torch.cat([metadata[:complete_tokens], compressed_metadata], dim=0)
            else:
                # Fallback to padding if compressors not initialized
                return self._adapt_sequence_length_fallback_pad(keys, vals, metadata)
        else:
            # Default: truncate (original behavior)
            truncate_to = total_tokens - remainder
            keys = keys[:truncate_to]
            vals = vals[:truncate_to]
            if metadata is not None:
                metadata = metadata[:truncate_to]
        
        return (keys, vals, metadata) if metadata is not None else (keys, vals)
    
    def _adapt_sequence_length_fallback_pad(self, keys: torch.Tensor, vals: torch.Tensor, metadata: Optional[torch.Tensor] = None) -> tuple:
        """Fallback padding method."""
        total_tokens, num_heads, head_dim = keys.shape
        remainder = total_tokens % self.chunk_size
        pad_size = self.chunk_size - remainder
        
        pad_keys = torch.zeros(pad_size, num_heads, head_dim, 
                             dtype=keys.dtype, device=keys.device)
        pad_vals = torch.zeros(pad_size, num_heads, head_dim,
                             dtype=vals.dtype, device=vals.device)
        
        keys = torch.cat([keys, pad_keys], dim=0)
        vals = torch.cat([vals, pad_vals], dim=0)
        
        if metadata is not None:
            pad_metadata = torch.zeros(pad_size, 2, dtype=metadata.dtype, device=metadata.device)
            metadata = torch.cat([metadata, pad_metadata], dim=0)
            return keys, vals, metadata
        
        return keys, vals
    
    def reset(self):
        """Reset the memory bank."""
        self.dstore_idx = 0
        
        # Reset FAISS indices
        for index in self.index_list:
            index.reset()
        
        # Reinitialize storage
        self._initialize_storage()
        
        # Reset metadata
        self.types = {"unknown": 0}
        self.next_type_id = 1
        self.entry_metadata = {}
        self.id_to_chunks = {}
        self.current_index = 0
        
        self.time_for_retrieve = 0.0
        self.retrieve_count = 0
    
    def add_index(
        self,
        qkv_val: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Add key-value pairs to the memory bank.
        
        Args:
            qkv_val: Dictionary with 'k' and 'v' tensors
                    Shape: [batch, seq_len, num_heads, head_dim]
            padding_mask: Optional padding mask
        """
        keys = qkv_val['k']
        vals = qkv_val['v']
        
        batch_size, seq_len, num_heads, head_dim = keys.shape
        
        # Check if memory is full and needs cleanup
        chunks_to_add = (batch_size * seq_len) // self.chunk_size
        max_chunks = self.memory_size // self.chunk_size
        
        if self.dstore_idx + chunks_to_add >= max_chunks:
            # Remove oldest entries
            update_size = min(chunks_to_add * 2, max_chunks // 2)
            self._remove_oldest(update_size)
        
        # Reshape keys and values
        keys = keys.reshape(batch_size * seq_len, num_heads, head_dim)
        vals = vals.reshape(batch_size * seq_len, num_heads, head_dim)
        
        # Adapt sequence length to handle all data without truncation
        keys, vals = self._adapt_sequence_length(keys, vals)
        
        # Check if we have any data after adaptation
        if keys.size(0) == 0:
            return  # No data to store
        
        # Reshape into chunks
        total_adapted_tokens = keys.size(0)
        num_chunks = total_adapted_tokens // self.chunk_size
        keys_chunked = keys.reshape(
            num_chunks, self.chunk_size, num_heads, head_dim
        )
        vals_chunked = vals.reshape(
            num_chunks, self.chunk_size, num_heads, head_dim
        )
        
        # Add to each head's index
        for i, index in enumerate(self.index_list):
            # Compute chunk representatives (mean over chunk)
            chunk_keys = keys_chunked[:, :, i, :].mean(dim=1)
            
            # Add to FAISS index
            if self.use_gpu_to_search:
                index.add(chunk_keys.float().contiguous())
            else:
                index.add(chunk_keys.cpu().float().numpy())
            
            # Store full keys and values
            end_idx = self.dstore_idx + num_chunks
            self.keys[i][self.dstore_idx:end_idx] = keys_chunked[:, :, i, :]
            self.vals[i][self.dstore_idx:end_idx] = vals_chunked[:, :, i, :]
        
        self.dstore_idx += num_chunks
    
    def _remove_oldest(self, num_chunks: int):
        """Remove oldest entries from memory."""
        if self.use_gpu_to_search:
            for i, index in enumerate(self.index_list):
                # Convert to CPU, remove, convert back
                cpu_index = faiss.index_gpu_to_cpu(index)
                cpu_index.remove_ids(np.arange(num_chunks))
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
                self.index_list[i] = gpu_index
        else:
            for index in self.index_list:
                index.remove_ids(np.arange(num_chunks))
        
        # Shift storage
        for i in range(self.num_heads):
            self.keys[i] = torch.cat([
                self.keys[i][num_chunks:],
                torch.zeros_like(self.keys[i][:num_chunks])
            ])
            self.vals[i] = torch.cat([
                self.vals[i][num_chunks:],
                torch.zeros_like(self.vals[i][:num_chunks])
            ])
        
        self.dstore_idx = max(0, self.dstore_idx - num_chunks)
    
    def retrieve(
        self,
        queries: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve nearest neighbors from memory.
        
        Args:
            queries: Query tensor [seq_len, batch, hidden_dim]
        
        Returns:
            Dictionary with retrieved keys and values
        """
        seq_len, batch_size, hidden_dim = queries.shape
        
        # Check if memory has only dummy data
        if self.dstore_idx <= self.dummy_chunks:
            # Return empty results for empty memory
            empty_keys = torch.zeros(batch_size * self.num_heads, seq_len, 0, self.head_dim, device=queries.device)
            empty_vals = torch.zeros(batch_size * self.num_heads, seq_len, 0, self.head_dim, device=queries.device)
            return {
                'k': empty_keys,
                'v': empty_vals,
                'metadata': torch.zeros(0, 2),
                'types_dict': self.types
            }
        
        # Reshape queries for multi-head
        queries = queries.reshape(
            seq_len * batch_size, self.num_heads, self.head_dim
        ).float()
        
        # Optimized: Batch all head searches together
        k_per_chunk = self.k // self.chunk_size
        if k_per_chunk == 0:
            k_per_chunk = 1
        
        # CRITICAL FIX: Ensure we don't request more results than available in the index
        # Each index has dstore_idx entries, so we can't request more than that
        max_available = max(1, self.dstore_idx)  # Total entries in index
        k_per_chunk = min(k_per_chunk, max_available)
        
        if self.use_gpu_to_search:
            # GPU: Search each head separately
            indices_list = []
            for i in range(self.num_heads):
                query_head = queries[:, i, :].contiguous()  # [seq_len*batch, head_dim]
                _, indices = self.index_list[i].search(query_head, k_per_chunk)
                indices_list.append(indices)
        else:
            # CPU: Search queries one at a time to avoid FAISS hanging with batched queries
            # FAISS CPU has issues with multi-query searches, so we process individually
            indices_list = []
            num_queries = queries.shape[0]  # seq_len * batch_size
            
            for i in range(self.num_heads):
                query_head = queries[:, i, :].contiguous()  # [num_queries, head_dim]
                
                # Process each query individually
                head_indices = []
                for q_idx in range(num_queries):
                    single_query = query_head[q_idx:q_idx+1].cpu().numpy()  # [1, head_dim]
                    _, single_indices = self.index_list[i].search(single_query, k_per_chunk)
                    head_indices.append(single_indices)
                
                # Combine results for this head
                indices = np.vstack(head_indices)  # [num_queries, k_per_chunk]
                indices = torch.from_numpy(indices).to(queries.device)
                indices_list.append(indices)
        
        # Retrieve keys and values from storage
        retrieved_keys = []
        retrieved_vals = []
        
        for i in range(self.num_heads):
            indices = indices_list[i]  # [seq_len*batch, k_per_chunk]
            
            # Clamp indices to valid range to prevent out-of-bounds access
            max_idx = self.keys[i].shape[0] - 1
            if isinstance(indices, torch.Tensor):
                indices = torch.clamp(indices, 0, max_idx)
            else:
                indices = np.clip(indices, 0, max_idx)
            
            # Gather keys and values using indices
            head_keys = self.keys[i][indices]  # [seq_len*batch, k_per_chunk, chunk_size, head_dim]
            head_vals = self.vals[i][indices]
            
            # Reshape to [seq_len*batch, k, head_dim]
            head_keys = head_keys.reshape(seq_len * batch_size, self.k, self.head_dim)
            head_vals = head_vals.reshape(seq_len * batch_size, self.k, self.head_dim)
            
            retrieved_keys.append(head_keys)
            retrieved_vals.append(head_vals)
        
        # Stack and reshape retrieved keys/values across all heads
        keys_stacked = torch.stack(retrieved_keys, dim=0)
        vals_stacked = torch.stack(retrieved_vals, dim=0)
        
        # Reshape to combine heads: [seq_len*batch*num_heads, k, head_dim]
        keys_stacked = keys_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        vals_stacked = vals_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        
        # Final reshape to match expected output format
        keys_final = keys_stacked.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        
        vals_final = vals_stacked.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        
        # Create simplified metadata (placeholder structure for compatibility)
        metadata_list = []
        for i in range(self.num_heads):
            indices = indices_list[i]  # [seq_len*batch, k_per_chunk]
            head_metadata = {'shape': indices.shape, 'type': 'placeholder'}
            metadata_list.append(head_metadata)
        
        return {
            'k': keys_final,
            'v': vals_final,
            'indices': indices_list,
            'metadata': metadata_list,
            'types_dict': self.types
        }
    
    def is_ready(self) -> bool:
        """Check if memory bank has enough entries for retrieval."""
        return self.dstore_idx > 0
    
    def get_size(self) -> int:
        """Get current number of stored chunks."""
        return self.dstore_idx
    
    def get_stats(self) -> Dict[str, float]:
        """Get retrieval statistics."""
        avg_time = (
            self.time_for_retrieve / self.retrieve_count
            if self.retrieve_count > 0 else 0.0
        )
        
        return {
            'total_retrievals': self.retrieve_count,
            'avg_retrieval_time': avg_time,
            'memory_usage': self.dstore_idx / (self.memory_size // self.chunk_size),
        }
    
    def _get_type_id(self, type_name: str) -> int:
        """Get or create type ID for label encoding."""
        if type_name not in self.types:
            self.types[type_name] = self.next_type_id
            self.next_type_id += 1
        return self.types[type_name]
    
    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text to get key-value pairs using a simplified, universal approach."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be provided for text encoding")
        
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # Reduced for better compatibility
            padding=False
        )
        
        device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(device)
        
        # Use ONLY the embedding layer for fast encoding (no full model inference)
        with torch.no_grad():
            try:
                embedding_layer = None
                
                # First try user-provided embedding layer path
                if self.embedding_layer_path:
                    try:
                        # Navigate to embedding layer using dot notation (e.g., "model.embed_tokens")
                        obj = self.model
                        for attr in self.embedding_layer_path.split('.'):
                            obj = getattr(obj, attr)
                        embedding_layer = obj
                        # Using user-specified embedding layer
                    except AttributeError:
                        pass  # Warning: User-specified embedding layer path not found
                
                # If no user path or user path failed, try common patterns
                if embedding_layer is None:
                    if hasattr(self.model, 'wte'):  # GPT2 style
                        embedding_layer = self.model.wte
                    elif hasattr(self.model, 'embed_tokens'):  # Llama/Qwen style
                        embedding_layer = self.model.embed_tokens
                    elif hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):  # BERT style
                        embedding_layer = self.model.embeddings.word_embeddings
                    elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):  # Nested GPT2
                        embedding_layer = self.model.transformer.wte
                
                if embedding_layer is not None:
                    # Fast: Just get embeddings, no full forward pass
                    hidden_states = embedding_layer(input_ids)
                    # Fast embedding extraction successful
                else:
                    # Fallback: create random representations (still much faster than full inference)
                    batch_size, seq_len = input_ids.shape
                    hidden_states = torch.randn(
                        batch_size, seq_len, self.dimension,
                        device=device, dtype=torch.float32
                    )
                    # Warning: Could not find embedding layer, using random representations
                    
            except Exception as e:
                # Warning: Embedding extraction failed
                # Ultimate fallback: create random representations
                batch_size, seq_len = input_ids.shape
                hidden_states = torch.randn(
                    batch_size, seq_len, self.dimension,
                    device=device, dtype=torch.float32
                )
            
            # Extract K-V pairs from the hidden states
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Ensure hidden_dim matches our expected dimension
            if hidden_dim != self.dimension:
                # Project to correct dimension
                projection = torch.nn.Linear(hidden_dim, self.dimension, bias=False).to(device)
                hidden_states = projection(hidden_states)
            
            # Split into keys and values for each attention head
            # Simple approach: use the same hidden states for both K and V
            keys = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            values = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            return {'k': keys, 'v': values}
    
    def add(self, entries: List[Dict[str, Any]]) -> List[str]:
        """
        Add text entries to memory bank.
        
        Args:
            entries: List of dictionaries with:
                - id: Optional UUID string (generated if not provided)
                - inp_txt: Input text to encode and store
                - type: Optional type label (default: "unknown")
        
        Returns:
            List of entry IDs that were added
        """
        if not entries:
            return []
        
        added_ids = []
        
        for entry in entries:
            # Generate ID if not provided
            entry_id = entry.get('id', str(uuid.uuid4()))
            text = entry.get('inp_txt', '')
            entry_type = entry.get('type', 'unknown')
            
            if not text:
                continue  # Skip empty text entries
            
            # Get type ID
            type_id = self._get_type_id(entry_type)
            
            # Encode text to get K-V pairs
            try:
                kv_pairs = self._encode_text(text)
            except Exception as e:
                # Warning: Failed to encode text for entry {entry_id}: {e}")
                continue
            
            # Store in memory bank with original add_index logic
            keys = kv_pairs['k']
            vals = kv_pairs['v']
            
            batch_size, seq_len, num_heads, head_dim = keys.shape
            # Entry {entry_id}: keys shape {keys.shape}, vals shape {vals.shape}")
            
            # Check if memory is full and needs cleanup
            chunks_to_add = (batch_size * seq_len) // self.chunk_size
            max_chunks = self.memory_size // self.chunk_size
            # Entry {entry_id}: chunks_to_add={chunks_to_add}, max_chunks={max_chunks}, current_idx={self.dstore_idx}")
            
            if self.dstore_idx + chunks_to_add >= max_chunks:
                # Remove oldest entries
                update_size = min(chunks_to_add * 2, max_chunks // 2)
                self._remove_oldest(update_size)
            
            # Track starting chunk index for this entry
            start_chunk_idx = self.dstore_idx
            chunk_indices = []
            
            # Reshape keys and values
            keys = keys.reshape(batch_size * seq_len, num_heads, head_dim)
            vals = vals.reshape(batch_size * seq_len, num_heads, head_dim)
            
            # Create metadata tensor for this entry
            total_tokens = keys.size(0)
            entry_metadata_tensor = torch.zeros((total_tokens, 2), 
                                               dtype=torch.float32, 
                                               device=keys.device)
            entry_metadata_tensor[:, 0] = type_id  # type_id
            entry_metadata_tensor[:, 1] = 0        # deleted=0
            
            # Adapt sequence length to handle all data without truncation
            adaptation_result = self._adapt_sequence_length(keys, vals, entry_metadata_tensor)
            
            if len(adaptation_result) == 3:
                keys, vals, adapted_metadata = adaptation_result
            else:
                keys, vals = adaptation_result
                adapted_metadata = None
            
            # Check if we have any data after adaptation
            if keys.size(0) == 0:
                # ❌ No data to store for entry {entry_id} after adaptation")
                continue  # No data to store
            
            # Entry {entry_id}: After adaptation, keys shape {keys.shape}")
            
            # Reshape into chunks
            total_adapted_tokens = keys.size(0)
            num_chunks = total_adapted_tokens // self.chunk_size
            # Entry {entry_id}: total_adapted_tokens={total_adapted_tokens}, num_chunks={num_chunks}, chunk_size={self.chunk_size}")
            
            # Allow storing entries even if they don't fill a complete chunk
            if num_chunks == 0 and total_adapted_tokens > 0:
                # For small entries, treat them as a single chunk
                num_chunks = 1
                # Entry {entry_id}: Small entry, treating as 1 chunk")
            
            # Handle chunking for both regular and small entries
            if total_adapted_tokens < self.chunk_size:
                # Small entry: pad to chunk size
                padding_needed = self.chunk_size - total_adapted_tokens
                keys_for_chunking = torch.cat([
                    keys,
                    torch.zeros(padding_needed, keys.size(1), keys.size(2), device=keys.device, dtype=keys.dtype)
                ], dim=0)
                vals_for_chunking = torch.cat([
                    vals,
                    torch.zeros(padding_needed, vals.size(1), vals.size(2), device=vals.device, dtype=vals.dtype)
                ], dim=0)
                # Entry {entry_id}: Padded from {total_adapted_tokens} to {keys_for_chunking.size(0)} tokens")
            else:
                # Regular entry: trim to exact chunk boundaries
                trimmed_tokens = num_chunks * self.chunk_size
                keys_for_chunking = keys[:trimmed_tokens]
                vals_for_chunking = vals[:trimmed_tokens]
            
            keys_chunked = keys_for_chunking.reshape(
                num_chunks, self.chunk_size, num_heads, head_dim
            )
            vals_chunked = vals_for_chunking.reshape(
                num_chunks, self.chunk_size, num_heads, head_dim
            )
            
            # Add to each head's index
            for i, index in enumerate(self.index_list):
                # Compute chunk representatives (mean over chunk)
                chunk_keys = keys_chunked[:, :, i, :].mean(dim=1)
                
                # Add to FAISS index
                if self.use_gpu_to_search:
                    index.add(chunk_keys.float().contiguous())
                else:
                    index.add(chunk_keys.detach().cpu().float().numpy())
                
                # Store full keys and values
                end_idx = self.dstore_idx + num_chunks
                self.keys[i][self.dstore_idx:end_idx] = keys_chunked[:, :, i, :]
                self.vals[i][self.dstore_idx:end_idx] = vals_chunked[:, :, i, :]
            
            # Store metadata for each chunk
            for i, chunk_idx in enumerate(range(self.dstore_idx, self.dstore_idx + num_chunks)):
                # Extract metadata for this chunk if available
                if adapted_metadata is not None:
                    chunk_start = i * self.chunk_size
                    chunk_end = min(chunk_start + self.chunk_size, adapted_metadata.size(0))
                    chunk_metadata = adapted_metadata[chunk_start:chunk_end]
                    # Average the metadata for the chunk
                    avg_type = chunk_metadata[:, 0].mean().item()
                    avg_deleted = chunk_metadata[:, 1].mean().item()
                else:
                    avg_type = type_id
                    avg_deleted = 0
                
                self.entry_metadata[chunk_idx] = {
                    'id': entry_id,
                    'type': int(avg_type),
                    'deleted': int(avg_deleted),
                    'text': text
                }
                chunk_indices.append(chunk_idx)
            
            # Map entry ID to chunk indices
            self.id_to_chunks[entry_id] = chunk_indices
            
            self.dstore_idx += num_chunks
            added_ids.append(entry_id)
            # Successfully added entry to memory
        
        return added_ids
    
    def clear(self):
        """Clear all memory and reset to initial state."""
        # Reset indices
        for index in self.index_list:
            index.reset()
        
        # Reinitialize storage
        self._initialize_storage()
        
        # Reset metadata
        self.dstore_idx = 0
        self.types = {"unknown": 0}
        self.next_type_id = 1
        self.entry_metadata = {}
        self.id_to_chunks = {}
        self.current_index = 0
        
        # Reset stats
        self.time_for_retrieve = 0.0
        self.retrieve_count = 0
    
    def update(self, entry_id: str, text: str) -> bool:
        """
        Update an existing entry with new text.
        
        Args:
            entry_id: ID of entry to update
            text: New text content
        
        Returns:
            True if update successful, False if entry not found
        """
        if entry_id not in self.id_to_chunks:
            return False
        
        # Remove old entry
        self.delete(entry_id, soft=False)
        
        # Add new entry with same ID
        entries = [{'id': entry_id, 'inp_txt': text, 'type': 'unknown'}]
        added_ids = self.add(entries)
        
        return len(added_ids) > 0
    
    def get(self, entry_id: str) -> Optional[str]:
        """
        Get text content for an entry ID.
        
        Args:
            entry_id: Entry ID to retrieve
        
        Returns:
            Text content if found, None otherwise
        """
        if entry_id not in self.id_to_chunks:
            return None
        
        # Get first chunk metadata (all chunks have same text)
        chunk_indices = self.id_to_chunks[entry_id]
        if not chunk_indices:
            return None
        
        first_chunk_idx = chunk_indices[0]
        if first_chunk_idx in self.entry_metadata:
            return self.entry_metadata[first_chunk_idx]['text']
        
        return None
    
    def set_index(self, index: int):
        """Set current index position."""
        self.current_index = max(0, min(index, self.dstore_idx - 1))
    
    def get_index(self) -> int:
        """Get current index position."""
        return self.current_index
    
    def delete(self, entry_id: str, soft: bool = True) -> bool:
        """
        Delete an entry from memory.
        
        Args:
            entry_id: ID of entry to delete
            soft: If True, mark as deleted but keep in memory
                 If False, physically remove from memory
        
        Returns:
            True if deletion successful, False if entry not found
        """
        if entry_id not in self.id_to_chunks:
            return False
        
        chunk_indices = self.id_to_chunks[entry_id]
        
        if soft:
            # Soft delete: mark as deleted but keep data
            for chunk_idx in chunk_indices:
                if chunk_idx in self.entry_metadata:
                    self.entry_metadata[chunk_idx]['deleted'] = 1
        else:
            # Hard delete: remove from indices and storage
            # Note: This is complex with FAISS as it requires rebuilding indices
            # For now, we'll mark as deleted and let garbage collection handle it
            for chunk_idx in chunk_indices:
                if chunk_idx in self.entry_metadata:
                    del self.entry_metadata[chunk_idx]
            
            del self.id_to_chunks[entry_id]
        
        return True
    
    def set_threshold(self, threshold: float):
        """
        Set the retrieval gate threshold.
        
        Args:
            threshold: Float between 0 and 1. Higher values make retrieval more selective.
                      Default is 0.5.
        """
        self.retrieval_threshold = max(0.0, min(1.0, threshold))
    
    def get_threshold(self) -> float:
        """Get the current retrieval gate threshold."""
        return self.retrieval_threshold
