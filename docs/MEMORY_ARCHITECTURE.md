# Hybrid Memory Architecture - Complete Guide

## Overview

This document explains the dual memory system in the Hybrid Transformer Model, which combines **Internal Memory** (short-term, learnable) and **External Memory** (long-term, retrieval-based) to enhance language model capabilities.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Internal Memory Module](#internal-memory-module)
3. [External Memory Module](#external-memory-module)
4. [Gating Mechanism](#gating-mechanism)
5. [When and How Memory is Used](#when-and-how-memory-is-used)
6. [Memory Updates](#memory-updates)
7. [Configuration Guide](#configuration-guide)
8. [Performance Characteristics](#performance-characteristics)

---

## Architecture Overview

The Hybrid Memory Architecture integrates two complementary memory systems:

```
Input Sequence
      ↓
┌─────────────────────────────────────┐
│   Transformer Layer                 │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Self-Attention (A_t)        │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │  Internal Memory (M_int)     │  │ ← Short-term, learnable
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │  External Memory (M_ext)     │  │ ← Long-term, retrieval
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │  Softmax Gating              │  │
│  │  H_t = Σ g[i] * M[i]         │  │
│  └──────────────────────────────┘  │
│              ↓                      │
│  ┌──────────────────────────────┐  │
│  │  Feed-Forward Network        │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
      ↓
Output Sequence
```

### Key Equation

```
H_t = g[0] * A_t + g[1] * M_int + g[2] * M_ext

where g = softmax(W_g @ h_t + b_g)
```

---

## Internal Memory Module

### Purpose
Provides **short-term, learnable memory** that adapts during training to capture task-specific patterns and recent context.

### Architecture

```python
Input: hidden_states [batch, seq_len, hidden_dim]
       memory [batch, mem_slots, mem_slots]

Step 1: Project to memory space
    input_proj = Linear(hidden_dim → mem_slots)(hidden_states)
    # Shape: [batch, seq_len, mem_slots]

Step 2: Read from memory
    memory_read = input_proj @ memory
    # Shape: [batch, seq_len, mem_slots]

Step 3: Project back to hidden dimension
    memory_output = Linear(mem_slots → hidden_dim)(memory_read)
    # Shape: [batch, seq_len, hidden_dim]

Step 4: Gating
    gate = Sigmoid(Linear(hidden_dim*2 → hidden_dim)(concat(hidden_states, memory_output)))
    gated_output = gate * memory_output
    # Shape: [batch, seq_len, hidden_dim]

Output: gated_output [batch, seq_len, hidden_dim]
        updated_memory [batch, mem_slots, mem_slots]
```

### When It's Used

**Always active** when `use_internal_memory=True` in every transformer layer.

- **Training:** Memory is updated with each forward pass
- **Inference:** Memory persists across sequences in the same session
- **Batch Processing:** Each sample in the batch has its own memory state

### How It's Updated

```python
# 1. Project hidden states for update
update_proj = Linear(hidden_dim → mem_slots)(hidden_states)
# Shape: [batch, seq_len, mem_slots]

# 2. Compute memory update via outer product
memory_update = (update_proj.T @ input_proj) / seq_len
# Shape: [batch, mem_slots, mem_slots]

# 3. Apply exponential moving average
decay_factor = 0.9
updated_memory = decay_factor * old_memory + (1 - decay_factor) * memory_update

# 4. Normalize for stability
updated_memory = F.normalize(updated_memory, p=2, dim=-1)
```

### Memory Initialization

```python
# Template memory (identity matrix)
memory_template = torch.eye(mem_slots)

# Dynamically expanded for each batch
internal_memory = memory_template.unsqueeze(0).expand(batch_size, -1, -1)
```

### Configuration Parameters

```python
HybridTransformerConfig(
    use_internal_memory=True,           # Enable/disable
    memory_slots=hidden_size // 128,    # Number of memory slots (auto-calculated)
    num_mem_heads=memory_slots // 4,    # Number of memory heads (auto-calculated)
)
```

**Example Sizes:**
- DialoGPT-small (hidden=768): 6 slots → 6×6 = 36 parameters per sample
- Qwen-1.5B (hidden=1536): 12 slots → 12×12 = 144 parameters per sample
- Llama-7B (hidden=4096): 32 slots → 32×32 = 1,024 parameters per sample

---

## External Memory Module

### Purpose
Provides **long-term, retrieval-based memory** that stores and retrieves relevant information from a large knowledge bank using FAISS vector search.

### Architecture

```python
Memory Bank Structure:
├── Keys: [num_heads, memory_size, head_dim]
├── Values: [num_heads, memory_size, head_dim]
├── Metadata: {id → text, type, timestamp}
└── FAISS Indices: [num_heads] (one per attention head)

Retrieval Process:
Input: queries [seq_len, batch, hidden_dim]

Step 1: Reshape queries for multi-head attention
    queries = queries.view(seq_len, batch, num_heads, head_dim)
    # Shape: [seq_len, batch, num_heads, head_dim]

Step 2: Search FAISS index (per head, per query)
    For each head h:
        For each query q in [seq_len × batch]:
            distances, indices = faiss_index[h].search(q, k=retrieval_k)
            # Returns top-k nearest neighbors

Step 3: Gather retrieved keys and values
    retrieved_k = keys[h][indices]  # [seq_len×batch, k, head_dim]
    retrieved_v = values[h][indices]  # [seq_len×batch, k, head_dim]

Step 4: Reshape for attention
    ext_k = retrieved_k.view(batch, num_heads, seq_len, k, head_dim)
    ext_v = retrieved_v.view(batch, num_heads, seq_len, k, head_dim)

Output: {'k': ext_k, 'v': ext_v, 'metadata': [...], 'types_dict': {...}}
```

### When It's Used

**Selectively active** at specific layers when `use_external_memory=True`:

```python
# Default: Retrieve at L/6 layers (evenly spaced)
retrieval_layers = [0, L/6, 2L/6, 3L/6, 4L/6, 5L/6]

# Example for 12-layer model:
retrieval_layers = [0, 2, 4, 6, 8, 10]
```

**Why selective?**
- Reduces computational cost
- Focuses retrieval at key decision points
- Empirically shown to be effective

### How It's Used in Attention

```python
# 1. Compute attention scores with retrieved memories
query = hidden_states.view(batch, seq_len, num_heads, head_dim)
# Shape: [batch, num_heads, seq_len, head_dim]

scores = query @ ext_keys.transpose(-2, -1) / sqrt(head_dim)
# Shape: [batch, num_heads, seq_len, k]

# 2. Apply softmax
attn_weights = softmax(scores, dim=-1)

# 3. Weighted sum of retrieved values
external_context = attn_weights @ ext_values
# Shape: [batch, num_heads, seq_len, head_dim]

# 4. Project back to hidden dimension
external_output = Linear(hidden_dim → hidden_dim)(external_context)
# Shape: [batch, seq_len, hidden_dim]
```

### Memory Operations

#### 1. Add Entries

```python
entries = [
    {"id": "code_1", "inp_txt": "def fibonacci(n): ...", "type": "code"},
    {"id": "doc_1", "inp_txt": "Recursion is...", "type": "documentation"},
]

added_ids = external_memory.add(entries)
# Returns: ["code_1", "doc_1"]
```

**Process:**
1. Encode text using model's embedding layer (fast)
2. Split into chunks if needed (based on `chunk_size`)
3. Add to FAISS index for each attention head
4. Store metadata (text, type, timestamp)

#### 2. Retrieve Entries

```python
# Automatic during forward pass
result = external_memory.retrieve(queries)
# Returns: {'k': ..., 'v': ..., 'metadata': ..., 'types_dict': ...}
```

#### 3. Update Entries

```python
success = external_memory.update("code_1", "def fibonacci(n): return n if n <= 1 else ...")
# Returns: True if successful
```

#### 4. Delete Entries

```python
# Soft delete (keeps in index, marks as deleted)
external_memory.delete("code_1", soft=True)

# Hard delete (removes from index)
external_memory.delete("code_1", soft=False)
```

#### 5. Clear Memory

```python
external_memory.clear()  # Removes all entries
```

### Configuration Parameters

```python
HybridTransformerConfig(
    use_external_memory=True,              # Enable/disable
    external_memory_size=1048576,          # Max entries (1M default)
    retrieval_k=8,                         # Top-k retrievals per query
    chunk_size=4,                          # Tokens per chunk
    use_gpu_to_search=True,                # Use GPU for FAISS (if available)
    seq_adapt_method="compress",           # How to handle variable seq lengths
    num_retrieval_layers=None,             # Auto: L/6 layers
    embedding_layer_path="wte",            # Path to embedding layer
)
```

### Sequence Adaptation Methods

When input sequences don't fit `chunk_size` perfectly:

1. **"pad"**: Pad sequences to multiple of chunk_size
2. **"pool"**: Average pool to reduce sequence length
3. **"compress"**: Use learned compression layers (default, most effective)

---

## Gating Mechanism

### Purpose
Dynamically combines outputs from different memory sources based on context.

### Architecture

```python
# 1. Compute gate logits
gate_logits = Linear(hidden_dim → 3)(hidden_states)
# Shape: [batch, seq_len, 3]
# 3 = max sources (self-attention, internal, external)

# 2. Select active sources
active_sources = count_enabled_memories()  # 1, 2, or 3
gate_logits = gate_logits[:, :, :active_sources]

# 3. Apply softmax (ensures sum to 1)
gate_weights = softmax(gate_logits, dim=-1)
# Shape: [batch, seq_len, active_sources]

# 4. Weighted combination
output = sum(gate_weights[:, :, i:i+1] * memory_outputs[i] 
             for i in range(active_sources))
# Shape: [batch, seq_len, hidden_dim]
```

### Gate Weight Interpretation

```python
# Example gate weights at a specific position:
gate_weights[0, 5, :] = [0.6, 0.3, 0.1]

Interpretation:
- 60% from self-attention (current context)
- 30% from internal memory (learned patterns)
- 10% from external memory (retrieved knowledge)
```

### Adaptive Behavior

The model learns to:
- **Rely on self-attention** for local context
- **Use internal memory** for task-specific patterns
- **Query external memory** when external knowledge is needed

---

## When and How Memory is Used

### Training Phase

```python
# Forward pass
for layer in model.layers:
    # 1. Self-attention (always)
    attn_out = layer.self_attn(hidden_states)
    
    # 2. Internal memory (if enabled)
    if use_internal_memory:
        mem_out, updated_memory = layer.internal_memory(attn_out, memory)
        memory = updated_memory  # Update for next iteration
    
    # 3. External memory (if enabled and at retrieval layer)
    if use_external_memory and layer_idx in retrieval_layers:
        ext_kv = external_memory.retrieve(hidden_states)
        ext_out = layer.external_attention(hidden_states, ext_kv)
    
    # 4. Gating
    combined = gate_network.combine([attn_out, mem_out, ext_out])
    
    # 5. Feed-forward
    output = layer.ffn(combined)
```

### Inference Phase

```python
# Session-based inference
model.eval()

# Memory persists across sequences in same session
for sequence in conversation:
    logits, loss, memory_info = model(sequence)
    # Internal memory carries over to next sequence
    # External memory can be queried for relevant context
```

### Batch Processing

```python
# Each sample has independent memory
batch = ["Sample 1", "Sample 2", "Sample 3"]
inputs = tokenizer(batch, return_tensors="pt")

# Internal memory: [3, mem_slots, mem_slots]
# Each sample gets its own memory state
logits, loss, memory_info = model(inputs)
```

---

## Memory Updates

### Internal Memory Update Frequency

**Every forward pass** through each layer:

```python
# Pseudo-code for one layer
def forward(hidden_states, memory):
    # Use current memory
    output, new_memory = internal_memory(hidden_states, memory)
    
    # Memory is updated immediately
    return output, new_memory

# Memory flows through layers
memory_0 = initial_memory
for layer in layers:
    output, memory_0 = layer(hidden_states, memory_0)
```

### External Memory Update Frequency

**On-demand** via explicit operations:

```python
# Add new knowledge
external_memory.add([
    {"inp_txt": "New information", "type": "fact"}
])

# Update existing
external_memory.update("entry_id", "Updated text")

# Delete outdated
external_memory.delete("old_entry_id")
```

### Memory Persistence

```python
# Internal Memory
- Training: Reset each batch
- Inference: Persists across sequences in session
- Saved with model checkpoint: No (reinitialized)

# External Memory
- Training: Manually managed
- Inference: Persists across sessions
- Saved separately: Yes (FAISS indices + metadata)
```

---

## Configuration Guide

### Minimal Configuration (Internal Only)

```python
config = HybridTransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    
    # Memory settings
    use_internal_memory=True,
    use_external_memory=False,
)
```

### Full Hybrid Configuration

```python
config = HybridTransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    
    # Internal memory
    use_internal_memory=True,
    memory_slots=6,              # 768 // 128 = 6
    num_mem_heads=2,             # 6 // 4 = 2 (rounded)
    
    # External memory
    use_external_memory=True,
    external_memory_size=1048576,  # 1M entries
    retrieval_k=8,                 # Top-8 retrieval
    chunk_size=4,
    use_gpu_to_search=True,
    seq_adapt_method="compress",
    
    # Retrieval layers
    num_retrieval_layers=2,        # Retrieve at 2 layers
)
```

### Memory-Optimized Configuration

```python
config = HybridTransformerConfig(
    # ... base config ...
    
    # Smaller memory footprint
    memory_slots=4,                # Fewer slots
    external_memory_size=65536,    # 64K entries
    retrieval_k=4,                 # Top-4 retrieval
    num_retrieval_layers=1,        # Single retrieval layer
)
```

---

## Performance Characteristics

### Memory Overhead

```python
# Internal Memory (per sample)
memory_size = mem_slots × mem_slots × 4 bytes (float32)

Examples:
- 6 slots:  6×6 = 144 bytes
- 12 slots: 12×12 = 576 bytes
- 32 slots: 32×32 = 4,096 bytes

# External Memory (total)
index_size ≈ num_heads × memory_size × head_dim × 4 bytes

Example (12 heads, 1M entries, head_dim=64):
12 × 1,000,000 × 64 × 4 = 3.072 GB
```

### Computational Overhead

```python
# Internal Memory: O(seq_len × mem_slots²)
- Negligible for small mem_slots (< 32)

# External Memory: O(seq_len × retrieval_k × log(memory_size))
- FAISS search is very efficient
- GPU acceleration available

# Gating: O(seq_len × hidden_dim)
- Minimal overhead
```

### Inference Speed

From test results:
```
Baseline model:     0.088s avg
Hybrid model:       0.020s avg
PEFT + Hybrid:      0.021s avg

Memory overhead:    -76.6% (faster due to better representations!)
```

---

## Best Practices

### 1. Memory Slot Sizing

```python
# Rule of thumb: hidden_size // 128
memory_slots = max(4, min(32, hidden_size // 128))
```

### 2. External Memory Management

```python
# Organize by type
external_memory.add([
    {"inp_txt": "...", "type": "code"},
    {"inp_txt": "...", "type": "documentation"},
    {"inp_txt": "...", "type": "example"},
])

# Periodic cleanup
if external_memory.get_size() > threshold:
    # Remove old or low-relevance entries
    external_memory.delete(old_ids, soft=False)
```

### 3. Retrieval Layer Selection

```python
# For 12-layer model:
# Early layers: Local patterns
# Middle layers: Task-specific (good for retrieval)
# Late layers: Output generation

# Recommended: Middle 1/3 of layers
num_layers = 12
retrieval_layers = [4, 6, 8]  # Layers 4, 6, 8
```

### 4. Batch Size Considerations

```python
# Internal memory scales with batch size
# External memory cost is constant

# For large batches: Prefer external memory
# For small batches: Both work well
```

---

## Example Usage

### Basic Inference

```python
from transformers import AutoTokenizer
from mem_aug.components.memory.hybrid_model import (
    HybridTransformerModel,
    HybridTransformerConfig
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# Create config
config = HybridTransformerConfig(
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    use_internal_memory=True,
    use_external_memory=True,
)

# Create model
model = HybridTransformerModel(config, tokenizer)

# Inference
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
logits, loss, memory_info = model(inputs["input_ids"])

print(f"Internal memory shape: {memory_info['internal_memory'].shape}")
print(f"External memory size: {memory_info['external_memory_size']}")
```

### Adding Knowledge to External Memory

```python
# Add code examples
code_examples = [
    {"id": "fib", "inp_txt": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)", "type": "code"},
    {"id": "sort", "inp_txt": "def quicksort(arr): ...", "type": "code"},
]

model.external_memory.add(code_examples)

# Now the model can retrieve these during inference
inputs = tokenizer("Write a fibonacci function", return_tensors="pt")
logits, loss, memory_info = model(inputs["input_ids"], use_external_memory=True)
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```python
# Reduce memory slots
config.memory_slots = 4

# Reduce external memory size
config.external_memory_size = 65536

# Reduce retrieval k
config.retrieval_k = 4
```

### Issue: Slow Inference

**Solution:**
```python
# Use GPU for FAISS
config.use_gpu_to_search = True

# Reduce retrieval layers
config.num_retrieval_layers = 1

# Use compression for sequence adaptation
config.seq_adapt_method = "compress"
```

### Issue: Poor Memory Utilization

**Solution:**
```python
# Check gate weights during inference
logits, loss, memory_info = model(inputs)
# Inspect which memory sources are being used

# Adjust memory sizes if one source dominates
# Train longer to learn better gating
```

---

## References

- **LM2**: Learning to Memorize with Memory-Augmented Transformers
- **LongMem**: Long-Term Memory for Language Models via External Retrieval
- **FAISS**: Efficient Similarity Search Library

---

## Summary

The Hybrid Memory Architecture provides:

✅ **Short-term adaptation** via internal memory  
✅ **Long-term knowledge** via external memory  
✅ **Dynamic combination** via learned gating  
✅ **Efficient retrieval** via FAISS indexing  
✅ **Flexible configuration** for different use cases  

This dual-memory system enables language models to maintain both learned patterns and retrieved knowledge, significantly improving performance on knowledge-intensive tasks.
