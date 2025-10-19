# Functional Architecture - Layer-by-Layer Transformations

## Detailed Function Flow for Each Architecture

---

## 1. Normal Transformer (Baseline)

### Layer-by-Layer Functions

```python
# Input Processing
x₀ = input_tokens                                    # [batch, seq_len]
x₁ = Embedding(x₀)                                   # [batch, seq_len, hidden_dim]
x₂ = PositionalEncoding(x₁)                          # [batch, seq_len, hidden_dim]

# Layer 1
h₁ = LayerNorm₁(x₂)                                  # Pre-normalization
q₁, k₁, v₁ = SelfAttention.QKV(h₁)                  # Query, Key, Value projections
attn₁ = Softmax(q₁ @ k₁ᵀ / √d)                      # Attention weights
a₁ = attn₁ @ v₁                                      # Attention output
x₃ = x₂ + a₁                                         # Residual connection

h₂ = LayerNorm₂(x₃)                                  # Pre-normalization
f₁ = FeedForward(h₂)                                 # FFN: W₂·ReLU(W₁·h₂)
x₄ = x₃ + f₁                                         # Residual connection

# Layer 2
h₃ = LayerNorm₃(x₄)
q₂, k₂, v₂ = SelfAttention.QKV(h₃)
attn₂ = Softmax(q₂ @ k₂ᵀ / √d)
a₂ = attn₂ @ v₂
x₅ = x₄ + a₂

h₄ = LayerNorm₄(x₅)
f₂ = FeedForward(h₄)
x₆ = x₅ + f₂

# ... (repeat for N layers)

# Layer N
h₂ₙ₋₁ = LayerNorm₂ₙ₋₁(x₂ₙ₋₂)
qₙ, kₙ, vₙ = SelfAttention.QKV(h₂ₙ₋₁)
attnₙ = Softmax(qₙ @ kₙᵀ / √d)
aₙ = attnₙ @ vₙ
x₂ₙ₋₁ = x₂ₙ₋₂ + aₙ

h₂ₙ = LayerNorm₂ₙ(x₂ₙ₋₁)
fₙ = FeedForward(h₂ₙ)
x₂ₙ = x₂ₙ₋₁ + fₙ

# Output
h_final = FinalLayerNorm(x₂ₙ)                        # Final normalization
logits = LMHead(h_final)                             # [batch, seq_len, vocab_size]
```

### Key Functions:
- `Embedding(x)`: Token → Vector
- `SelfAttention.QKV(x)`: Linear projections for Q, K, V
- `Softmax(scores)`: Attention weights
- `FeedForward(x)`: W₂·ReLU(W₁·x + b₁) + b₂
- `LayerNorm(x)`: (x - μ) / σ · γ + β

---

## 2. Transformer with Internal Memory (LM²)

### Layer-by-Layer Functions with Memory

```python
# Input Processing
x₀ = input_tokens                                    # [batch, seq_len]
x₁ = Embedding(x₀)                                   # [batch, seq_len, hidden_dim]
x₂ = PositionalEncoding(x₁)                          # [batch, seq_len, hidden_dim]

# Initialize Internal Memory
M₀ = Eye(mem_slots)                                  # [mem_slots, mem_slots]
M₀ = M₀.expand(batch, -1, -1)                       # [batch, mem_slots, mem_slots]

# Layer 1
h₁ = LayerNorm₁(x₂)                                  # Pre-normalization

# Self-Attention
q₁, k₁, v₁ = SelfAttention.QKV(h₁)                  # [batch, seq_len, hidden_dim]
attn₁ = Softmax(q₁ @ k₁ᵀ / √d)                      # [batch, seq_len, seq_len]
a₁ = attn₁ @ v₁                                      # [batch, seq_len, hidden_dim]

# Internal Memory Processing
m_query₁ = MemoryQuery(a₁)                          # [batch, seq_len, mem_slots]
m_key₁ = MemoryKey(M₀)                              # [batch, mem_slots, head_size]
m_value₁ = MemoryValue(M₀)                          # [batch, mem_slots, head_size]

m_attn₁ = Softmax(m_query₁ @ m_key₁ᵀ)              # [batch, seq_len, mem_slots]
m_out₁ = m_attn₁ @ m_value₁                         # [batch, seq_len, head_size]
m_out₁ = MemoryProjection(m_out₁)                   # [batch, seq_len, hidden_dim]

# Update Memory State
M₁ = UpdateMemory(M₀, a₁, m_out₁)                   # [batch, mem_slots, mem_slots]
# M₁ = M₀ + InputGate(a₁) @ ForgetGate(M₀)

# Gating Network (Softmax Fusion)
gate_logits₁ = GateNetwork(h₁)                      # [batch, seq_len, 2]
g₁ = Softmax(gate_logits₁)                          # [batch, seq_len, 2]
# g₁[0] = weight for self-attention
# g₁[1] = weight for internal memory

# Fused Output
fused₁ = g₁[:,:,0:1] * a₁ + g₁[:,:,1:2] * m_out₁   # [batch, seq_len, hidden_dim]
x₃ = x₂ + fused₁                                     # Residual connection

# Feed Forward
h₂ = LayerNorm₂(x₃)
f₁ = FeedForward(h₂)
x₄ = x₃ + f₁

# Layer 2 (with memory propagation)
h₃ = LayerNorm₃(x₄)

# Self-Attention
q₂, k₂, v₂ = SelfAttention.QKV(h₃)
attn₂ = Softmax(q₂ @ k₂ᵀ / √d)
a₂ = attn₂ @ v₂

# Internal Memory Processing (using M₁ from previous layer)
m_query₂ = MemoryQuery(a₂)
m_key₂ = MemoryKey(M₁)                              # Uses updated memory
m_value₂ = MemoryValue(M₁)
m_attn₂ = Softmax(m_query₂ @ m_key₂ᵀ)
m_out₂ = m_attn₂ @ m_value₂
m_out₂ = MemoryProjection(m_out₂)

# Update Memory State
M₂ = UpdateMemory(M₁, a₂, m_out₂)                   # Memory propagates

# Gating
gate_logits₂ = GateNetwork(h₃)
g₂ = Softmax(gate_logits₂)
fused₂ = g₂[:,:,0:1] * a₂ + g₂[:,:,1:2] * m_out₂
x₅ = x₄ + fused₂

h₄ = LayerNorm₄(x₅)
f₂ = FeedForward(h₄)
x₆ = x₅ + f₂

# ... (repeat for N layers, memory Mᵢ propagates through all layers)

# Layer N
h₂ₙ₋₁ = LayerNorm₂ₙ₋₁(x₂ₙ₋₂)
qₙ, kₙ, vₙ = SelfAttention.QKV(h₂ₙ₋₁)
attnₙ = Softmax(qₙ @ kₙᵀ / √d)
aₙ = attnₙ @ vₙ

m_queryₙ = MemoryQuery(aₙ)
m_keyₙ = MemoryKey(Mₙ₋₁)
m_valueₙ = MemoryValue(Mₙ₋₁)
m_attnₙ = Softmax(m_queryₙ @ m_keyₙᵀ)
m_outₙ = m_attnₙ @ m_valueₙ
m_outₙ = MemoryProjection(m_outₙ)

Mₙ = UpdateMemory(Mₙ₋₁, aₙ, m_outₙ)

gate_logitsₙ = GateNetwork(h₂ₙ₋₁)
gₙ = Softmax(gate_logitsₙ)
fusedₙ = gₙ[:,:,0:1] * aₙ + gₙ[:,:,1:2] * m_outₙ
x₂ₙ₋₁ = x₂ₙ₋₂ + fusedₙ

h₂ₙ = LayerNorm₂ₙ(x₂ₙ₋₁)
fₙ = FeedForward(h₂ₙ)
x₂ₙ = x₂ₙ₋₁ + fₙ

# Output
h_final = FinalLayerNorm(x₂ₙ)
logits = LMHead(h_final)
```

### Key Memory Functions:
- `MemoryQuery(x)`: Project to memory query space
- `MemoryKey(M)`: Extract keys from memory matrix
- `MemoryValue(M)`: Extract values from memory matrix
- `UpdateMemory(M_prev, attn, mem_out)`: Update memory state
- `GateNetwork(x)`: W_gate @ x + b_gate → [batch, seq, 2]

---

## 3. Transformer with External Memory (LongMem)

### Layer-by-Layer Functions with Retrieval

```python
# Input Processing
x₀ = input_tokens                                    # [batch, seq_len]
x₁ = Embedding(x₀)                                   # [batch, seq_len, hidden_dim]
x₂ = PositionalEncoding(x₁)                          # [batch, seq_len, hidden_dim]

# External Memory Bank (Pre-populated)
# ExternalMemory = {
#     'keys': [num_chunks, chunk_size, num_heads, head_dim],
#     'values': [num_chunks, chunk_size, num_heads, head_dim],
#     'faiss_index': FAISS index for similarity search
# }

# Determine Retrieval Layers
retrieval_layers = GetRetrievalLayers(num_layers)    # e.g., [0, 6, 12, 18, 24]

# Layer 1 (Retrieval Layer)
h₁ = LayerNorm₁(x₂)

# Self-Attention
q₁, k₁, v₁ = SelfAttention.QKV(h₁)                  # [batch, seq_len, hidden_dim]
attn₁ = Softmax(q₁ @ k₁ᵀ / √d)
a₁ = attn₁ @ v₁

# External Memory Retrieval
if layer_idx in retrieval_layers:
    # Reshape query for retrieval
    query₁ = h₁.view(batch, seq_len, num_heads, head_dim)
    
    # FAISS Similarity Search
    chunk_indices₁ = FAISSSearch(query₁, k=top_k)   # [batch, seq_len, top_k]
    
    # Retrieve K-V pairs
    ext_keys₁ = ExternalMemory.keys[chunk_indices₁] # [batch, seq_len, top_k, head_dim]
    ext_vals₁ = ExternalMemory.values[chunk_indices₁]
    
    # Compute attention with retrieved memories
    ext_scores₁ = query₁ @ ext_keys₁.transpose(-2,-1) / √d
    ext_attn₁ = Softmax(ext_scores₁)                # [batch, seq_len, top_k]
    ext_out₁ = ext_attn₁ @ ext_vals₁                # [batch, seq_len, head_dim]
    ext_out₁ = ExternalMemoryProjection(ext_out₁)   # [batch, seq_len, hidden_dim]
else:
    ext_out₁ = 0                                     # No retrieval

# Gating Network (Softmax Fusion)
gate_logits₁ = GateNetwork(h₁)                      # [batch, seq_len, 2]
g₁ = Softmax(gate_logits₁)
# g₁[0] = weight for self-attention
# g₁[1] = weight for external memory

# Fused Output
fused₁ = g₁[:,:,0:1] * a₁ + g₁[:,:,1:2] * ext_out₁
x₃ = x₂ + fused₁

# Feed Forward
h₂ = LayerNorm₂(x₃)
f₁ = FeedForward(h₂)
x₄ = x₃ + f₁

# Layer 2 (Non-Retrieval Layer)
h₃ = LayerNorm₃(x₄)
q₂, k₂, v₂ = SelfAttention.QKV(h₃)
attn₂ = Softmax(q₂ @ k₂ᵀ / √d)
a₂ = attn₂ @ v₂

# No retrieval for this layer
ext_out₂ = 0

gate_logits₂ = GateNetwork(h₃)
g₂ = Softmax(gate_logits₂)
fused₂ = g₂[:,:,0:1] * a₂ + g₂[:,:,1:2] * ext_out₂  # ext_out₂ = 0
x₅ = x₄ + fused₂

h₄ = LayerNorm₄(x₅)
f₂ = FeedForward(h₄)
x₆ = x₅ + f₂

# ... (repeat, only retrieve at specified layers)

# Layer N (Retrieval Layer)
h₂ₙ₋₁ = LayerNorm₂ₙ₋₁(x₂ₙ₋₂)
qₙ, kₙ, vₙ = SelfAttention.QKV(h₂ₙ₋₁)
attnₙ = Softmax(qₙ @ kₙᵀ / √d)
aₙ = attnₙ @ vₙ

if layer_idx in retrieval_layers:
    queryₙ = h₂ₙ₋₁.view(batch, seq_len, num_heads, head_dim)
    chunk_indicesₙ = FAISSSearch(queryₙ, k=top_k)
    ext_keysₙ = ExternalMemory.keys[chunk_indicesₙ]
    ext_valsₙ = ExternalMemory.values[chunk_indicesₙ]
    ext_scoresₙ = queryₙ @ ext_keysₙ.transpose(-2,-1) / √d
    ext_attnₙ = Softmax(ext_scoresₙ)
    ext_outₙ = ext_attnₙ @ ext_valsₙ
    ext_outₙ = ExternalMemoryProjection(ext_outₙ)
else:
    ext_outₙ = 0

gate_logitsₙ = GateNetwork(h₂ₙ₋₁)
gₙ = Softmax(gate_logitsₙ)
fusedₙ = gₙ[:,:,0:1] * aₙ + gₙ[:,:,1:2] * ext_outₙ
x₂ₙ₋₁ = x₂ₙ₋₂ + fusedₙ

h₂ₙ = LayerNorm₂ₙ(x₂ₙ₋₁)
fₙ = FeedForward(h₂ₙ)
x₂ₙ = x₂ₙ₋₁ + fₙ

# Output
h_final = FinalLayerNorm(x₂ₙ)
logits = LMHead(h_final)
```

### Key External Memory Functions:
- `FAISSSearch(query, k)`: Similarity search in FAISS index
- `ExternalMemory.keys[indices]`: Retrieve stored keys
- `ExternalMemory.values[indices]`: Retrieve stored values
- `ExternalMemoryProjection(x)`: Project retrieved context
- `GetRetrievalLayers(N)`: Determine which layers retrieve

---

## 4. Hybrid Transformer (Internal + External Memory) ⭐

### Complete Layer-by-Layer Functions

```python
# Input Processing
x₀ = input_tokens                                    # [batch, seq_len]
x₁ = Embedding(x₀)                                   # [batch, seq_len, hidden_dim]
x₂ = PositionalEncoding(x₁)                          # [batch, seq_len, hidden_dim]

# Initialize Internal Memory
M₀ = Eye(mem_slots)                                  # [mem_slots, mem_slots]
M₀ = M₀.expand(batch, -1, -1)                       # [batch, mem_slots, mem_slots]

# External Memory Bank (Pre-populated)
# ExternalMemory = {...}

# Retrieval Layers
retrieval_layers = GetRetrievalLayers(num_layers)    # e.g., [0, 6, 12, 18, 24]

# ============================================================
# Layer 1 (Retrieval Layer)
# ============================================================
h₁ = LayerNorm₁(x₂)                                  # Pre-normalization

# -------------------- Self-Attention (A_t) --------------------
q₁, k₁, v₁ = SelfAttention.QKV(h₁)                  # [batch, seq_len, hidden_dim]
attn₁ = Softmax(q₁ @ k₁ᵀ / √d)                      # [batch, seq_len, seq_len]
a₁ = attn₁ @ v₁                                      # [batch, seq_len, hidden_dim]

# -------------------- Internal Memory (M_int) --------------------
m_query₁ = MemoryQuery(a₁)                          # [batch, seq_len, mem_slots]
m_key₁ = MemoryKey(M₀)                              # [batch, mem_slots, head_size]
m_value₁ = MemoryValue(M₀)                          # [batch, mem_slots, head_size]

m_attn₁ = Softmax(m_query₁ @ m_key₁ᵀ)              # [batch, seq_len, mem_slots]
m_out₁ = m_attn₁ @ m_value₁                         # [batch, seq_len, head_size]
m_out₁ = MemoryProjection(m_out₁)                   # [batch, seq_len, hidden_dim]

# Update Memory State
M₁ = UpdateMemory(M₀, a₁, m_out₁)                   # [batch, mem_slots, mem_slots]

# -------------------- External Memory (M_ext) with Learnable Gate --------------------
if layer_idx in retrieval_layers:
    # Learnable Gate Check
    gate_value₁ = Sigmoid(g_retrieve₁)              # Scalar parameter
    threshold = ExternalMemory.get_threshold()       # Default: 0.5
    
    if training_mode:
        # Training: Always retrieve, weight by gate
        query₁ = a₁.view(batch, seq_len, num_heads, head_dim)
        chunk_indices₁ = FAISSSearch(query₁, k=top_k)
        ext_keys₁ = ExternalMemory.keys[chunk_indices₁]
        ext_vals₁ = ExternalMemory.values[chunk_indices₁]
        ext_scores₁ = query₁ @ ext_keys₁.transpose(-2,-1) / √d
        ext_attn₁ = Softmax(ext_scores₁)
        ext_context₁ = ext_attn₁ @ ext_vals₁
        ext_context₁ = ExternalMemoryProjection(ext_context₁)
        
        ext_out₁ = gate_value₁ * ext_context₁       # Weighted by gate
    
    elif gate_value₁ > threshold:
        # Inference: Retrieve only if gate > threshold
        query₁ = a₁.view(batch, seq_len, num_heads, head_dim)
        chunk_indices₁ = FAISSSearch(query₁, k=top_k)
        ext_keys₁ = ExternalMemory.keys[chunk_indices₁]
        ext_vals₁ = ExternalMemory.values[chunk_indices₁]
        ext_scores₁ = query₁ @ ext_keys₁.transpose(-2,-1) / √d
        ext_attn₁ = Softmax(ext_scores₁)
        ext_out₁ = ext_attn₁ @ ext_vals₁
        ext_out₁ = ExternalMemoryProjection(ext_out₁)
    
    else:
        # Inference: Skip retrieval (gate too low)
        ext_out₁ = 0                                 # Fast path!
else:
    ext_out₁ = 0                                     # Non-retrieval layer

# -------------------- Gating Network (3-way Softmax Fusion) --------------------
gate_logits₁ = GateNetwork(h₁)                      # [batch, seq_len, 3]
g₁ = Softmax(gate_logits₁)                          # [batch, seq_len, 3]
# g₁[:,:,0] = weight for self-attention (A_t)
# g₁[:,:,1] = weight for internal memory (M_int)
# g₁[:,:,2] = weight for external memory (M_ext)

# Fused Output: H_t = g[0]·A_t + g[1]·M_int + g[2]·M_ext
fused₁ = (g₁[:,:,0:1] * a₁ + 
          g₁[:,:,1:2] * m_out₁ + 
          g₁[:,:,2:3] * ext_out₁)                   # [batch, seq_len, hidden_dim]

x₃ = x₂ + fused₁                                     # Residual connection

# -------------------- Feed Forward --------------------
h₂ = LayerNorm₂(x₃)
f₁ = FeedForward(h₂)                                 # W₂·ReLU(W₁·h₂)
x₄ = x₃ + f₁

# ============================================================
# Layer 2 (Non-Retrieval Layer)
# ============================================================
h₃ = LayerNorm₃(x₄)

# Self-Attention
q₂, k₂, v₂ = SelfAttention.QKV(h₃)
attn₂ = Softmax(q₂ @ k₂ᵀ / √d)
a₂ = attn₂ @ v₂

# Internal Memory (using M₁ from previous layer)
m_query₂ = MemoryQuery(a₂)
m_key₂ = MemoryKey(M₁)                              # Memory propagates
m_value₂ = MemoryValue(M₁)
m_attn₂ = Softmax(m_query₂ @ m_key₂ᵀ)
m_out₂ = m_attn₂ @ m_value₂
m_out₂ = MemoryProjection(m_out₂)

M₂ = UpdateMemory(M₁, a₂, m_out₂)

# External Memory (no retrieval for this layer)
ext_out₂ = 0

# Gating
gate_logits₂ = GateNetwork(h₃)
g₂ = Softmax(gate_logits₂)
fused₂ = (g₂[:,:,0:1] * a₂ + 
          g₂[:,:,1:2] * m_out₂ + 
          g₂[:,:,2:3] * ext_out₂)                   # ext_out₂ = 0

x₅ = x₄ + fused₂

h₄ = LayerNorm₄(x₅)
f₂ = FeedForward(h₄)
x₆ = x₅ + f₂

# ... (repeat for middle layers)

# ============================================================
# Layer N (Retrieval Layer)
# ============================================================
h₂ₙ₋₁ = LayerNorm₂ₙ₋₁(x₂ₙ₋₂)

# Self-Attention
qₙ, kₙ, vₙ = SelfAttention.QKV(h₂ₙ₋₁)
attnₙ = Softmax(qₙ @ kₙᵀ / √d)
aₙ = attnₙ @ vₙ

# Internal Memory
m_queryₙ = MemoryQuery(aₙ)
m_keyₙ = MemoryKey(Mₙ₋₁)
m_valueₙ = MemoryValue(Mₙ₋₁)
m_attnₙ = Softmax(m_queryₙ @ m_keyₙᵀ)
m_outₙ = m_attnₙ @ m_valueₙ
m_outₙ = MemoryProjection(m_outₙ)

Mₙ = UpdateMemory(Mₙ₋₁, aₙ, m_outₙ)

# External Memory with Learnable Gate
if layer_idx in retrieval_layers:
    gate_valueₙ = Sigmoid(g_retrieveₙ)
    threshold = ExternalMemory.get_threshold()
    
    if training_mode:
        queryₙ = aₙ.view(batch, seq_len, num_heads, head_dim)
        chunk_indicesₙ = FAISSSearch(queryₙ, k=top_k)
        ext_keysₙ = ExternalMemory.keys[chunk_indicesₙ]
        ext_valsₙ = ExternalMemory.values[chunk_indicesₙ]
        ext_scoresₙ = queryₙ @ ext_keysₙ.transpose(-2,-1) / √d
        ext_attnₙ = Softmax(ext_scoresₙ)
        ext_contextₙ = ext_attnₙ @ ext_valsₙ
        ext_contextₙ = ExternalMemoryProjection(ext_contextₙ)
        ext_outₙ = gate_valueₙ * ext_contextₙ
    
    elif gate_valueₙ > threshold:
        queryₙ = aₙ.view(batch, seq_len, num_heads, head_dim)
        chunk_indicesₙ = FAISSSearch(queryₙ, k=top_k)
        ext_keysₙ = ExternalMemory.keys[chunk_indicesₙ]
        ext_valsₙ = ExternalMemory.values[chunk_indicesₙ]
        ext_scoresₙ = queryₙ @ ext_keysₙ.transpose(-2,-1) / √d
        ext_attnₙ = Softmax(ext_scoresₙ)
        ext_outₙ = ext_attnₙ @ ext_valsₙ
        ext_outₙ = ExternalMemoryProjection(ext_outₙ)
    
    else:
        ext_outₙ = 0
else:
    ext_outₙ = 0

# Gating
gate_logitsₙ = GateNetwork(h₂ₙ₋₁)
gₙ = Softmax(gate_logitsₙ)
fusedₙ = (gₙ[:,:,0:1] * aₙ + 
          gₙ[:,:,1:2] * m_outₙ + 
          gₙ[:,:,2:3] * ext_outₙ)

x₂ₙ₋₁ = x₂ₙ₋₂ + fusedₙ

h₂ₙ = LayerNorm₂ₙ(x₂ₙ₋₁)
fₙ = FeedForward(h₂ₙ)
x₂ₙ = x₂ₙ₋₁ + fₙ

# ============================================================
# Output
# ============================================================
h_final = FinalLayerNorm(x₂ₙ)                        # Final normalization
logits = LMHead(h_final)                             # [batch, seq_len, vocab_size]
```

### Complete Function Inventory for Hybrid Model:

#### Core Attention Functions:
- `SelfAttention.QKV(x)`: Linear projections → Q, K, V
- `Softmax(scores)`: Attention weights normalization
- `LayerNorm(x)`: Layer normalization

#### Internal Memory Functions:
- `MemoryQuery(x)`: W_q @ x → memory query
- `MemoryKey(M)`: W_k @ M → memory keys
- `MemoryValue(M)`: W_v @ M → memory values
- `MemoryProjection(x)`: W_proj
