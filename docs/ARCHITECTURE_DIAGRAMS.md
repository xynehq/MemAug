# Architecture Diagrams

## Comparison of Transformer Architectures

### 1. Normal Transformer (Baseline)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Embedding Layer                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────┐
        │   Transformer Layer 1   │
        │  ┌──────────────────┐  │
        │  │  Self-Attention  │  │
        │  └────────┬─────────┘  │
        │           │             │
        │           ▼             │
        │  ┌──────────────────┐  │
        │  │   Feed Forward   │  │
        │  └────────┬─────────┘  │
        └───────────┼─────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │   Transformer Layer 2   │
        │         ...             │
        └───────────┼─────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │   Transformer Layer N   │
        └───────────┼─────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                    Output Logits                         │
└─────────────────────────────────────────────────────────┘

Key: Only uses self-attention within the current context window
```

---

### 2. Transformer with Internal Memory (LM²)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Embedding Layer                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │     Transformer Layer 1             │
        │  ┌──────────────────┐              │
        │  │  Self-Attention  │              │
        │  └────────┬─────────┘              │
        │           │                         │
        │           ▼                         │
        │  ┌──────────────────┐              │
        │  │ Internal Memory  │◄─────────┐   │
        │  │   (M_int)        │          │   │
        │  │ ┌──────────────┐ │          │   │
        │  │ │ Memory Slots │ │  Persistent  │
        │  │ │ [S×S matrix] │ │   State   │   │
        │  │ └──────────────┘ │          │   │
        │  └────────┬─────────┘          │   │
        │           │                     │   │
        │           ▼                     │   │
        │  ┌──────────────────┐          │   │
        │  │  Gating Network  │          │   │
        │  │  (Softmax Fusion)│          │   │
        │  └────────┬─────────┘          │   │
        │           │                     │   │
        │           ▼                     │   │
        │  ┌──────────────────┐          │   │
        │  │   Feed Forward   │          │   │
        │  └────────┬─────────┘          │   │
        └───────────┼─────────────────────┼───┘
                    │                     │
                    │    Memory State     │
                    │    Propagates       │
                    ▼                     │
        ┌────────────────────────────────┼───┐
        │     Transformer Layer 2         │   │
        │         ...                     │   │
        └───────────┼─────────────────────┼───┘
                    │                     │
                    ▼                     │
        ┌────────────────────────────────┼───┐
        │     Transformer Layer N         │   │
        └───────────┼─────────────────────┘   │
                    │                         │
                    ▼                         │
┌─────────────────────────────────────────────────────────┐
│                    Output Logits                         │
└─────────────────────────────────────────────────────────┘

Key Features:
- Internal memory slots maintain state across layers
- Gating network learns to combine self-attention + memory
- Memory state persists and updates through the network
```

---

### 3. Transformer with External Memory (LongMem)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Embedding Layer                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────┐
        │     Transformer Layer 1             │
        │  ┌──────────────────┐              │
        │  │  Self-Attention  │              │
        │  └────────┬─────────┘              │
        │           │                         │
        │           ▼                         │
        │  ┌──────────────────┐              │
        │  │ External Memory  │              │
        │  │   Retrieval      │              │
        │  │                  │              │
        │  │  Query ──────────┼──────────┐   │
        │  └────────┬─────────┘          │   │
        │           │                     │   │
        │           │                     ▼   │
        │           │         ┌─────────────────────────┐
        │           │         │  External Memory Bank   │
        │           │         │  (FAISS Index)          │
        │           │         │                         │
        │           │         │  ┌─────────────────┐   │
        │           │         │  │ Stored K-V Pairs│   │
        │           │         │  │ from History    │   │
        │           │         │  └─────────────────┘   │
        │           │         │                         │
        │           │         │  Top-k Retrieval        │
        │           │◄────────┤  (Similarity Search)    │
        │           │         └─────────────────────────┘
        │           │
        │           ▼
        │  ┌──────────────────┐
        │  │  Gating Network  │
        │  │  (Softmax Fusion)│
        │  └────────┬─────────┘
        │           │
        │           ▼
        │  ┌──────────────────┐
        │  │   Feed Forward   │
        │  └────────┬─────────┘
        └───────────┼─────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────┐
        │     Transformer Layer 2             │
        │         ...                         │
        └───────────┼─────────────────────────┘
                    │
                    ▼
        ┌────────────────────────────────────┐
        │     Transformer Layer N             │
        └───────────┼─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                    Output Logits                         │
└─────────────────────────────────────────────────────────┘

Key Features:
- External memory bank stores historical K-V pairs
- FAISS index enables fast similarity search
- Only selected layers perform retrieval (e.g., every 6th layer)
- Retrieved context augments current attention
```

---

### 4. Hybrid Transformer (Internal + External Memory)

```
┌─────────────────────────────────────────────────────────┐
│                    Input Tokens                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Embedding Layer                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────┐
        │     Transformer Layer 1 (Retrieval Layer)      │
        │  ┌──────────────────┐                          │
        │  │  Self-Attention  │                          │
        │  │      (A_t)       │                          │
        │  └────────┬─────────┘                          │
        │           │                                     │
        │           ▼                                     │
        │  ┌──────────────────┐                          │
        │  │ Internal Memory  │◄─────────┐               │
        │  │   (M_int)        │          │               │
        │  │ ┌──────────────┐ │  Persistent              │
        │  │ │ Memory Slots │ │   State                  │
        │  │ └──────────────┘ │          │               │
        │  └────────┬─────────┘          │               │
        │           │                     │               │
        │           ▼                     │               │
        │  ┌──────────────────┐          │               │
        │  │ External Memory  │          │               │
        │  │   Retrieval      │          │               │
        │  │   (M_ext)        │          │               │
        │  │                  │          │               │
        │  │  Query ──────────┼──────────┼───────┐       │
        │  └────────┬─────────┘          │       │       │
        │           │                     │       │       │
        │           │                     │       ▼       │
        │           │                     │  ┌──────────────────────┐
        │           │                     │  │ External Memory Bank │
        │           │                     │  │ (FAISS Index)        │
        │           │                     │  │                      │
        │           │                     │  │ ┌────────────────┐  │
        │           │                     │  │ │ Historical K-V │  │
        │           │                     │  │ │ Pairs Storage  │  │
        │           │                     │  │ └────────────────┘  │
        │           │                     │  │                      │
        │           │                     │  │ ┌────────────────┐  │
        │           │                     │  │ │ Learnable Gate │  │
        │           │                     │  │ │  g_retrieve    │  │
        │           │                     │  │ │ (σ(0.5)≈0.62)  │  │
        │           │                     │  │ └────────────────┘  │
        │           │◄────────────────────┤  │                      │
        │           │                     │  │ Top-k Retrieval      │
        │           │                     │  │ (if gate > thresh)   │
        │           │                     │  └──────────────────────┘
        │           │                     │
        │           ▼                     │
        │  ┌──────────────────┐          │
        │  │  Gating Network  │          │
        │  │                  │          │
        │  │ H_t = g[0]·A_t + │          │
        │  │       g[1]·M_int+│          │
        │  │       g[2]·M_ext │          │
        │  │                  │          │
        │  │ g = softmax(W_g) │          │
        │  └────────┬─────────┘          │
        │           │                     │
        │           ▼                     │
        │  ┌──────────────────┐          │
        │  │   Feed Forward   │          │
        │  └────────┬─────────┘          │
        └───────────┼─────────────────────┼───┘
                    │                     │
                    │    Memory State     │
                    │    Propagates       │
                    ▼                     │
        ┌────────────────────────────────┼───┐
        │     Transformer Layer 2         │   │
        │     (Non-Retrieval Layer)       │   │
        │         ...                     │   │
        └───────────┼─────────────────────┼───┘
                    │                     │
                    ▼                     │
        ┌────────────────────────────────┼───┐
        │     Transformer Layer N         │   │
        │     (Retrieval Layer)           │   │
        └───────────┼─────────────────────┘   │
                    │                         │
                    ▼                         │
┌─────────────────────────────────────────────────────────┐
│                    Output Logits                         │
└─────────────────────────────────────────────────────────┘

Key Features:
✓ Combines both internal and external memory
✓ Internal memory: Fast, persistent state across layers
✓ External memory: Large-scale historical context retrieval
✓ Learnable gate (g_retrieve) controls external retrieval
✓ Softmax gating fuses all three sources (A_t, M_int, M_ext)
✓ Adaptive retrieval: Skip external memory when not needed
```

---

## Memory Fusion Equation

### Hybrid Model Output:
```
H_t = g[0] · A_t + g[1] · M_int + g[2] · M_ext

where:
  g = softmax(W_g @ h_t + b_g)  ← Learned gating weights
  A_t = Self-Attention output
  M_int = Internal memory output
  M_ext = External memory output (with learnable gate)
```

### Learnable Retrieval Gate:
```
Training Mode:
  M_ext = σ(g_retrieve) · ExternalMemory(query)
  ↳ Always retrieves, but weights the contribution

Inference Mode:
  if σ(g_retrieve) > threshold:
      M_ext = ExternalMemory(query)
  else:
      M_ext = 0  ← Skip retrieval for speed
```

---

## Performance Characteristics

| Architecture | Context Length | Speed | Memory Usage | Adaptability |
|-------------|----------------|-------|--------------|--------------|
| Normal Transformer | Limited (2K-32K) | Fast | Low | None |
| Internal Memory | Limited | Medium | Medium | Layer-wise |
| External Memory | Unlimited | Slow | High | Query-based |
| **Hybrid (Both)** | **Unlimited** | **Adaptive** | **Medium-High** | **Full** |

---

## Use Cases

### Normal Transformer
- Short documents
- Real-time inference
- Resource-constrained environments

### Internal Memory Only
- Medium-length documents
- Need persistent state
- Moderate computational budget

### External Memory Only
- Very long documents
- Historical context critical
- Can tolerate retrieval latency

### Hybrid (Recommended)
- **Code generation with large codebases**
- **Long-form document understanding**
- **Conversational AI with history**
- **Any task requiring both short-term and long-term memory**
