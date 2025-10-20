"""
Enhanced External Memory Bank with Optimized Vector Management for Large Contexts
Improvements:
- Better vector size management
- Automatic compression for large contexts
- Memory efficiency optimizations
- Gradient-friendly storage
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Any

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class EnhancedExternalMemoryBank(nn.Module):
    """
    Enhanced external memory with better vector management.
    Key improvements:
    1. Gradient-friendly storage (nn.Parameter)
    2. Automatic vector size optimization
    3. Efficient large context handling
    4. Memory usage monitoring
    """
    
    def __init__(self, config, model=None, tokenizer=None):
        super().__init__()
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install: pip install faiss-cpu")
        
        self.config = config
        self.dimension = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.dimension // self.num_heads
        self.k = config.retrieval_k
        self.chunk_size = config.chunk_size
        self.memory_size = config.external_memory_size
        self.use_gpu = config.use_gpu_to_search and torch.cuda.is_available()
        
        self.logger = logging.getLogger(__name__)
        
        # Calculate optimal storage size
        self.storage_size = self.memory_size // self.chunk_size
        
        # FIX #1: Use nn.Parameter for gradient flow
        self._initialize_gradient_friendly_storage()
        
        # Initialize FAISS indices
        self._initialize_faiss_indices()
        
        # Memory management
        self.dstore_idx = 0
        self.max_vector_norm = 10.0  # Clip vectors to prevent explosion
        self.compression_threshold = 512  # Compress sequences longer than this
        
        # Statistics
        self.stats = {
            'total_additions': 0,
            'total_retrievals': 0,
            'avg_vector_norm': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def _initialize_gradient_friendly_storage(self):
        """Initialize storage with gradient support."""
        device = 'cuda' if self.use_gpu else 'cpu'
        dtype = torch.float32  # Use float32 for stability
        
        # Use nn.ParameterList for gradient flow
        self.keys = nn.ParameterList([
            nn.Parameter(
                torch.zeros(
                    self.storage_size,
                    self.chunk_size,
                    self.head_dim,
                    dtype=dtype,
                    device=device
                ),
                requires_grad=True
            )
            for _ in range(self.num_heads)
        ])
        
        self.vals = nn.ParameterList([
            nn.Parameter(
                torch.zeros(
                    self.storage_size,
                    self.chunk_size,
                    self.head_dim,
                    dtype=dtype,
                    device=device
                ),
                requires_grad=True
            )
            for _ in range(self.num_heads)
        ])
        
        self.logger.info(f"Initialized gradient-friendly storage: {self.storage_size} chunks")
        self._log_memory_usage()
    
    def _initialize_faiss_indices(self):
        """Initialize FAISS indices for each head."""
        self.index_list = []
        
        for i in range(self.num_heads):
            if self.use_gpu:
                # GPU index
                self.res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(self.head_dim)
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
                self.index_list.append(gpu_index)
            else:
                # CPU index
                index = faiss.IndexFlatIP(self.head_dim)
                self.index_list.append(index)
        
        self.logger.info(f"Initialized {len(self.index_list)} FAISS indices")
    
    def _log_memory_usage(self):
        """Log current memory usage."""
        total_params = sum(p.numel() for p in self.parameters())
        memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        self.stats['memory_usage_mb'] = memory_mb
        self.logger.info(f"Memory usage: {memory_mb:.2f} MB ({total_params:,} parameters)")
    
    def _optimize_vector_size(
        self,
        vectors: torch.Tensor,
        target_size: int
    ) -> torch.Tensor:
        """
        Optimize vector size for large contexts.
        
        Args:
            vectors: [batch, seq_len, num_heads, head_dim]
            target_size: Target sequence length
        
        Returns:
            Optimized vectors: [batch, target_size, num_heads, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = vectors.shape
        
        if seq_len <= target_size:
            # Pad if needed
            if seq_len < target_size:
                padding = torch.zeros(
                    batch_size,
                    target_size - seq_len,
                    num_heads,
                    head_dim,
                    dtype=vectors.dtype,
                    device=vectors.device
                )
                vectors = torch.cat([vectors, padding], dim=1)
            return vectors
        
        # Compress using adaptive pooling
        # Reshape for pooling: [batch * num_heads, head_dim, seq_len]
        vectors_reshaped = vectors.permute(0, 2, 3, 1).reshape(
            batch_size * num_heads, head_dim, seq_len
        )
        
        # Adaptive average pooling
        pooled = nn.functional.adaptive_avg_pool1d(
            vectors_reshaped, target_size
        )
        
        # Reshape back: [batch, target_size, num_heads, head_dim]
        optimized = pooled.reshape(
            batch_size, num_heads, head_dim, target_size
        ).permute(0, 3, 1, 2)
        
        return optimized
    
    def _clip_vector_norms(self, vectors: torch.Tensor) -> torch.Tensor:
        """Clip vector norms to prevent explosion."""
        norms = torch.norm(vectors, dim=-1, keepdim=True)
        scale = torch.clamp(norms / self.max_vector_norm, min=1.0)
        return vectors / scale
    
    def add_index(
        self,
        qkv_val: Dict[str, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Add key-value pairs to memory with optimizations.
        
        Args:
            qkv_val: Dict with 'k' and 'v' tensors [batch, seq_len, num_heads, head_dim]
            padding_mask: Optional mask for padding tokens
        """
        keys = qkv_val['k']
        vals = qkv_val['v']
        
        batch_size, seq_len, num_heads, head_dim = keys.shape
        
        # Optimize vector size for large contexts
        if seq_len > self.compression_threshold:
            self.logger.info(f"Compressing sequence from {seq_len} to {self.compression_threshold}")
            keys = self._optimize_vector_size(keys, self.compression_threshold)
            vals = self._optimize_vector_size(vals, self.compression_threshold)
            seq_len = self.compression_threshold
        
        # Clip norms to prevent explosion
        keys = self._clip_vector_norms(keys)
        vals = self._clip_vector_norms(vals)
        
        # Update statistics
        avg_norm = torch.norm(keys, dim=-1).mean().item()
        self.stats['avg_vector_norm'] = (
            0.9 * self.stats['avg_vector_norm'] + 0.1 * avg_norm
        )
        
        # Reshape for storage
        keys = keys.reshape(batch_size * seq_len, num_heads, head_dim)
        vals = vals.reshape(batch_size * seq_len, num_heads, head_dim)
        
        # Adapt to chunk size
        total_tokens = keys.size(0)
        num_chunks = total_tokens // self.chunk_size
        
        if num_chunks == 0:
            # Too small, pad to one chunk
            padding_size = self.chunk_size - total_tokens
            keys = torch.cat([
                keys,
                torch.zeros(
                    padding_size, num_heads, head_dim,
                    dtype=keys.dtype, device=keys.device
                )
            ], dim=0)
            vals = torch.cat([
                vals,
                torch.zeros(
                    padding_size, num_heads, head_dim,
                    dtype=vals.dtype, device=vals.device
                )
            ], dim=0)
            num_chunks = 1
        else:
            # Trim to exact chunks
            total_tokens = num_chunks * self.chunk_size
            keys = keys[:total_tokens]
            vals = vals[:total_tokens]
        
        # Reshape into chunks
        keys_chunked = keys.reshape(num_chunks, self.chunk_size, num_heads, head_dim)
        vals_chunked = vals.reshape(num_chunks, self.chunk_size, num_heads, head_dim)
        
        # Check if memory is full
        if self.dstore_idx + num_chunks >= self.storage_size:
            # Remove oldest 25% of memory
            remove_size = self.storage_size // 4
            self._remove_oldest(remove_size)
        
        # Add to each head's index and storage
        for i in range(self.num_heads):
            # Compute chunk representatives (mean pooling)
            chunk_keys = keys_chunked[:, :, i, :].mean(dim=1)  # [num_chunks, head_dim]
            
            # Add to FAISS index
            if self.use_gpu:
                self.index_list[i].add(chunk_keys.float().contiguous())
            else:
                self.index_list[i].add(chunk_keys.detach().cpu().float().numpy())
            
            # Store in parameter tensors (with gradient support)
            end_idx = self.dstore_idx + num_chunks
            with torch.no_grad():  # Don't track these operations
                self.keys[i].data[self.dstore_idx:end_idx] = keys_chunked[:, :, i, :]
                self.vals[i].data[self.dstore_idx:end_idx] = vals_chunked[:, :, i, :]
        
        self.dstore_idx += num_chunks
        self.stats['total_additions'] += 1
        
        if self.stats['total_additions'] % 100 == 0:
            self.logger.info(
                f"Memory stats: {self.dstore_idx} chunks, "
                f"avg_norm={self.stats['avg_vector_norm']:.4f}"
            )
    
    def retrieve(
        self,
        queries: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve nearest neighbors with optimizations.
        
        Args:
            queries: [seq_len, batch, hidden_dim]
        
        Returns:
            Dict with retrieved keys and values
        """
        seq_len, batch_size, hidden_dim = queries.shape
        
        # FIX #2: Handle empty memory properly
        if self.dstore_idx == 0:
            # Return properly sized dummy data
            dummy_keys = torch.zeros(
                batch_size * self.num_heads, seq_len, self.k, self.head_dim,
                device=queries.device, dtype=queries.dtype
            )
            dummy_vals = torch.zeros(
                batch_size * self.num_heads, seq_len, self.k, self.head_dim,
                device=queries.device, dtype=queries.dtype
            )
            return {'k': dummy_keys, 'v': dummy_vals}
        
        # Reshape queries for multi-head
        queries = queries.reshape(
            seq_len * batch_size, self.num_heads, self.head_dim
        ).float()
        
        # Calculate retrieval size
        k_per_chunk = max(1, self.k // self.chunk_size)
        
        # FIX #8: Don't request more than available
        max_available = max(1, self.dstore_idx)
        k_per_chunk = min(k_per_chunk, max_available)
        
        # Retrieve from each head
        retrieved_keys = []
        retrieved_vals = []
        
        for i in range(self.num_heads):
            query_head = queries[:, i, :].contiguous()
            
            # Search FAISS index
            if self.use_gpu:
                _, indices = self.index_list[i].search(query_head, k_per_chunk)
            else:
                # CPU: Process queries individually to avoid hanging
                indices_list = []
                for q_idx in range(query_head.size(0)):
                    single_query = query_head[q_idx:q_idx+1].detach().cpu().numpy()
                    _, single_indices = self.index_list[i].search(single_query, k_per_chunk)
                    indices_list.append(single_indices)
                indices = np.vstack(indices_list)
                indices = torch.from_numpy(indices).to(queries.device)
            
            # Clamp indices to valid range
            max_idx = self.keys[i].size(0) - 1
            indices = torch.clamp(indices, 0, max_idx)
            
            # Gather keys and values
            head_keys = self.keys[i][indices]  # [seq*batch, k_per_chunk, chunk_size, head_dim]
            head_vals = self.vals[i][indices]
            
            # Reshape to [seq*batch, k, head_dim]
            head_keys = head_keys.reshape(seq_len * batch_size, self.k, self.head_dim)
            head_vals = head_vals.reshape(seq_len * batch_size, self.k, self.head_dim)
            
            retrieved_keys.append(head_keys)
            retrieved_vals.append(head_vals)
        
        # Stack across heads
        keys_stacked = torch.stack(retrieved_keys, dim=0)  # [num_heads, seq*batch, k, head_dim]
        vals_stacked = torch.stack(retrieved_vals, dim=0)
        
        # Reshape to final format
        keys_final = keys_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        vals_final = vals_stacked.transpose(0, 1).reshape(
            seq_len * batch_size * self.num_heads, self.k, self.head_dim
        )
        
        # Final reshape
        keys_final = keys_final.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        vals_final = vals_final.view(
            seq_len, batch_size * self.num_heads, self.k, self.head_dim
        ).transpose(0, 1)
        
        self.stats['total_retrievals'] += 1
        
        return {
            'k': keys_final,
            'v': vals_final
        }
    
    def _remove_oldest(self, num_chunks: int):
        """Remove oldest chunks from memory."""
        if num_chunks >= self.dstore_idx:
            # Clear all
            self.dstore_idx = 0
            for i in range(self.num_heads):
                self.index_list[i].reset()
            return
        
        # Shift data
        for i in range(self.num_heads):
            with torch.no_grad():
                self.keys[i].data[:-num_chunks] = self.keys[i].data[num_chunks:]
                self.vals[i].data[:-num_chunks] = self.vals[i].data[num_chunks:]
        
        self.dstore_idx -= num_chunks
        
        # Rebuild FAISS indices
        self._rebuild_faiss_indices()
        
        self.logger.info(f"Removed {num_chunks} oldest chunks")
    
    def _rebuild_faiss_indices(self):
        """Rebuild FAISS indices from current storage."""
        for i in range(self.num_heads):
            # Reset index
            if self.use_gpu:
                cpu_index = faiss.IndexFlatIP(self.head_dim)
                self.index_list[i] = faiss.index_cpu_to_gpu(
                    self.res,
                    torch.cuda.current_device(),
                    cpu_index
                )
            else:
                self.index_list[i] = faiss.IndexFlatIP(self.head_dim)
            
            # Re-add all chunks
            if self.dstore_idx > 0:
                chunk_keys = self.keys[i][:self.dstore_idx].mean(dim=1)
                if self.use_gpu:
                    self.index_list[i].add(chunk_keys.float().contiguous())
                else:
                    self.index_list[i].add(chunk_keys.detach().cpu().float().numpy())
    
    def get_size(self) -> int:
        """Get current memory size in chunks."""
        return self.dstore_idx
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            'current_size': self.dstore_idx,
            'capacity': self.storage_size,
            'utilization': self.dstore_idx / self.storage_size if self.storage_size > 0 else 0.0
        }
    
    def clear(self):
        """Clear all memory."""
        self.dstore_idx = 0
        for i in range(self.num_heads):
            self.index_list[i].reset()
            with torch.no_grad():
                self.keys[i].data.zero_()
                self.vals[i].data.zero_()
        
        self.stats['total_additions'] = 0
        self.stats['total_retrievals'] = 0
        self.logger.info("Memory cleared")
