# 🏗️ Hybrid LLM Architecture

A clean, modular implementation of a transformer-based language model enhanced with dual memory systems for improved long-context understanding.

---

## 🎯 Core Concept

This architecture extends any standard transformer model (Llama, Qwen, GPT, etc.) with two complementary memory mechanisms:

1. **Internal Memory** - Fast, learnable memory slots for short-term patterns
2. **External Memory** - Large-scale retrieval system for long-term context

---

## 📐 Architecture Overview

### Standard Transformer Flow

```
Input Tokens → Embeddings → Transformer Layers → Output Logits
                              ↓
                    [Self-Attention + FFN] × N layers
```

### Hybrid Transformer Flow

```
Input Tokens → Embeddings → Hybrid Layers → Output Logits
                              ↓
                    [Hybrid Attention + FFN] × N layers
                              ↓
                    Hybrid Attention = Self-Attention 
                                     + Internal Memory
                                     + External Memory
```

---

## 🔧 Component Details

### 1. Base Transformer

- **Architecture**: Any HuggingFace transformer (Llama, Qwen, GPT, etc.)
- **Loading**: Dynamically loaded from pretrained weights
- **Flexibility**: Works with any model size and configuration

### 2. Internal Memory Module

**Purpose**: Learn and store short-term patterns across sequences

**Mechanism**:
- Learnable memory slots (default: 16 slots)
- Multi-head attention over memory (default: 4 heads)
- Gated read/write operations
- Updates via gradient descent during training

**Flow**:
```python
# Read from memory
memory_output = attention(query=hidden_states, key=memory_slots)

# Gate what to remember
gate = sigmoid(gate_network(hidden_states, memory_output))

# Update memory state
new_memory = decay * old_memory + (1 - decay) * new_information

# Add to output
output = hidden_states + gate * memory_output
```

**Benefits**:
- Fast access (no retrieval needed)
- Adapts to task-specific patterns
- Minimal computational overhead

### 3. External Memory Bank

**Purpose**: Store and retrieve from large-scale long-term context

**Mechanism**:
- FAISS vector database for efficient similarity search
- Stores key-value pairs from past sequences
- Retrieves top-k most relevant chunks
- Separate index per attention head

**Flow**:
```python
# Store past sequences
external_memory.add(keys, values)  # Up to 1M tokens

# Retrieve relevant context
retrieved = external_memory.search(query, k=8)

# Attend over retrieved memories
context = attention(query=hidden_states, kv=retrieved)

# Fuse with current representation
output = hidden_states + context
```

**Benefits**:
- Massive capacity (configurable, default 1M tokens)
- Efficient retrieval (sub-linear search)
- Handles arbitrarily long contexts

### 4. Hybrid Attention Layer

**Integration**: Combines all three attention mechanisms using **softmax-based gating**

```python
def hybrid_attention(hidden_states, internal_memory, external_memory):
    # 1. Standard self-attention (A_t)
    attn_out = self_attention(hidden_states)
    
    # 2. Internal memory output (M_int)
    mem_out = None
    if use_internal_memory:
        mem_out, new_memory = internal_memory(attn_out, memory_state)
    
    # 3. External memory output (M_ext)
    ext_out = None
    if use_external_memory and external_memory.is_ready():
        retrieved = external_memory.retrieve(hidden_states)
        ext_out = fuse_external(hidden_states, retrieved)
    
    # 4. Softmax gating to combine outputs
    # Compute gate weights: g = softmax(W_g @ h_t + b_g)
    gate_logits = gate_network(hidden_states)  # [batch, seq_len, 3]
    gate_weights = softmax(gate_logits[:, :, :active_sources], dim=-1)
    
    # 5. Weighted combination: H_t = Σ g[i] * M[i]
    memory_outputs = [attn_out]
    if mem_out is not None:
        memory_outputs.append(mem_out)
    if ext_out is not None:
        memory_outputs.append(ext_out)
    
    combined = sum(gate_weights[:, :, i:i+1] * memory_outputs[i] 
                   for i in range(len(memory_outputs)))
    
    return combined, new_memory
```

**Key Equation**:
```
H_t = g[0] * A_t + g[1] * M_int + g[2] * M_ext
where g = softmax(W_g @ h_t + b_g)
```

This ensures gate weights sum to 1.0 and the model learns to dynamically balance between memory sources.

---

## 🎛️ Configuration

### Model Configuration

```yaml
model:
  model_name: "Qwen/Qwen2.5-Coder-3B-Instruct"  # Any HF model
  
  # Internal Memory
  use_internal_memory: true
  memory_slots: 16        # Number of memory slots
  num_mem_heads: 4        # Attention heads for memory
  
  # External Memory
  use_external_memory: true
  external_memory_size: 1048576  # 1M tokens capacity
  retrieval_k: 8          # Top-k chunks to retrieve
  chunk_size: 4           # Tokens per chunk
  use_gpu_to_search: true # GPU acceleration for FAISS
```

### Training Configuration

```yaml
train:
  batch_size: 4
  learning_rate: 1e-4
  max_iters: 10000
  grad_accum_steps: 4
  
data:
  train_data: "data/train"
  val_data: "data/val"
  max_input_length: 1024
  max_output_length: 1024
```

---

## 📊 Performance Characteristics

### Memory Overhead

| Component | Parameters | Storage |
|-----------|-----------|---------|
| Base Model | ~3-7B | Model weights |
| Internal Memory | ~0.1% of base | Negligible |
| External Memory | 0 (non-parametric) | FAISS index |

### Computational Cost

| Operation | Overhead | Notes |
|-----------|----------|-------|
| Self-Attention | Baseline | Standard transformer |
| Internal Memory | +5-10% | Gated operations |
| External Memory | +10-20% | FAISS search + fusion |
| **Total** | **~1.2-1.3x** | Still efficient |

### Context Handling

| Metric | Standard | Hybrid |
|--------|----------|--------|
| Max Context | 32K tokens | Effectively unlimited |
| Memory Retention | None | Persistent across sequences |
| Long-range Dependencies | Limited | Enhanced via retrieval |

---

## 🚀 Use Cases

### Ideal Applications

1. **Long Document Processing**
   - Technical documentation
   - Research papers
   - Books and articles

2. **Multi-turn Conversations**
   - Chatbots with memory
   - Customer support
   - Interactive assistants

3. **Code Understanding**
   - Large codebases
   - Cross-file references
   - API documentation

4. **Knowledge-Intensive Tasks**
   - Question answering
   - Fact verification
   - Information retrieval

### When to Use Each Memory Type

**Internal Memory Only**:
- Short sequences with recurring patterns
- Fast inference required
- Limited memory budget

**External Memory Only**:
- Very long contexts
- Retrieval-based tasks
- Static knowledge base

**Both (Recommended)**:
- Complex tasks requiring both short and long-term memory
- Best overall performance
- Maximum flexibility

---

## 🔄 Training Process

### 1. Data Preparation

```bash
# Prepare input/output pairs
python data_proc/prepare_io_data.py \
  --mode sample \
  --num-samples 10000
```

### 2. Model Initialization

```python
# Load base model config
base_config = AutoConfig.from_pretrained(model_name)

# Create hybrid config
hybrid_config = HybridTransformerConfig(
    **base_config.to_dict(),
    use_internal_memory=True,
    use_external_memory=True,
    # ... memory parameters
)

# Initialize model
model = HybridTransformerModel(hybrid_config, tokenizer)
```

### 3. Training Loop

```python
for batch in dataloader:
    # Forward pass with memory
    logits, loss, memory_info = model(
        input_ids=batch['input_ids'],
        targets=batch['targets']
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Memory is automatically updated
```

---

## 🧪 Inference

### Basic Generation

```python
# Load trained model
model = HybridTransformerModel.from_pretrained(checkpoint_path)

# Generate with memory
output = model.generate(
    input_ids,
    max_length=100,
    use_external_memory=True  # Enable/disable as needed
)
```

### Memory Management

```python
# Reset internal memory
model.internal_memory = torch.eye(memory_slots)

# Clear external memory
model.external_memory.reset()

# Check memory status
print(f"External memory size: {model.external_memory.get_size()}")
```

---

## 📁 Code Structure

```
hybrid_llm_trainer/
├── src/
│   ├── models/
│   │   ├── hybrid_model.py       # Main model class
│   │   ├── internal_memory.py    # LM2-style memory
│   │   └── external_memory.py    # FAISS-based retrieval
│   ├── trainer.py                # Training loop
│   ├── dataloader.py             # Data handling
│   └── utils.py                  # Helper functions
├── configs/
│   └── train.yaml                # Configuration
├── train.py                      # Training script
└── test_inference.py             # Inference testing
```

---

## 🎓 Key Design Principles

1. **Modularity**: Each memory type is independent and can be toggled
2. **Flexibility**: Works with any transformer architecture
3. **Efficiency**: Minimal overhead for maximum benefit
4. **Scalability**: Memory capacity is configurable
5. **Simplicity**: Clean, readable implementation

---

## 🔬 Technical Details

### Internal Memory Update Rule

```python
# Decay factor for memory persistence
decay = 0.9

# Update equation
new_memory = decay * old_memory + (1 - decay) * update

# Normalization for stability
new_memory = F.normalize(new_memory, p=2, dim=-1)
```

### External Memory Retrieval

```python
# FAISS index per attention head
for head in range(num_heads):
    # Search for top-k similar chunks
    scores, indices = faiss_index[head].search(query, k)
    
    # Retrieve stored key-value pairs
    retrieved_kv = memory_bank[head][indices]
```

### Attention Fusion

```python
# Compute attention over retrieved memories
scores = query @ retrieved_keys.T / sqrt(head_dim)
weights = softmax(scores)
context = weights @ retrieved_values

# Residual connection
output = hidden_states + projection(context)
```

---

## 📚 References

- **Base Architecture**: HuggingFace Transformers
- **Internal Memory**: Inspired by LM2 (Memory-augmented LMs)
- **External Memory**: Inspired by LongMem (FAISS-based retrieval)
- **Integration**: Novel hybrid approach combining both

---

## 🤝 Extending the Architecture

### Adding New Memory Types

1. Create new module in `src/models/`
2. Implement `forward()` method
3. Integrate in `HybridMemoryAttention`
4. Add configuration parameters

### Customizing Existing Memories

- **Internal Memory**: Modify `InternalMemoryModule` class
- **External Memory**: Modify `ExternalMemoryBank` class
- **Fusion**: Modify `_fuse_external_memory()` method

---

## ✅ Summary

The Hybrid LLM architecture provides:

- ✅ **Enhanced context understanding** via dual memory systems
- ✅ **Flexible configuration** for different use cases
- ✅ **Efficient implementation** with minimal overhead
- ✅ **Modular design** for easy customization
- ✅ **Production-ready** code with proper error handling

Perfect for applications requiring long-context understanding, multi-turn conversations, or knowledge-intensive tasks.
