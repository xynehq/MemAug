# Learnable Retrieval Gate Feature

## Overview

The learnable retrieval gate is a feature that allows the model to adaptively decide when to use external memory retrieval during inference, improving both speed and reasoning efficiency.

## Implementation Status

### ✅ Completed
1. **External Memory Threshold** - Added configurable threshold to `ExternalMemoryBank`
   - `retrieval_threshold` parameter (default: 0.5)
   - `set_threshold(threshold)` method to adjust threshold (0.0 to 1.0)
   - `get_threshold()` method to query current threshold

### 🚧 To Be Implemented

The following components need to be added to complete the feature:

#### 1. Learnable Gate Parameter in HybridMemoryAttention

Add to `src/mem_aug/components/memory/hybrid_model.py` in the `HybridMemoryAttention.__init__()`:

```python
# Learnable retrieval gate (if enabled)
use_gated_retrieval = getattr(config, 'use_gated_retrieval', False)
if use_gated_retrieval and config.use_external_memory:
    self.g_retrieve = nn.Parameter(torch.tensor(0.5))
    self.use_gated_retrieval = True
else:
    self.g_retrieve = None
    self.use_gated_retrieval = False
```

#### 2. Gated Retrieval Logic in Forward Pass

Modify the `HybridMemoryAttention.forward()` method:

```python
# External memory (M_ext) with gating
ext_out = None
if self.use_external_memory and external_kv is not None:
    if self.use_gated_retrieval and self.g_retrieve is not None:
        # Compute gate value
        gate = torch.sigmoid(self.g_retrieve)
        
        # Check against threshold
        if self.training:
            # During training: always retrieve but weight the output
            ext_context = self._compute_external_memory(
                hidden_states, external_kv, attention_mask
            )
            ext_out = gate * ext_context
        else:
            # During inference: skip retrieval if gate < threshold
            threshold = external_memory.get_threshold()
            if gate.item() > threshold:
                ext_out = self._compute_external_memory(
                    hidden_states, external_kv, attention_mask
                )
            # else: ext_out remains None, skipping external memory
    else:
        # Standard retrieval without gating
        ext_out = self._compute_external_memory(
            hidden_states, external_kv, attention_mask
        )
```

#### 3. Configuration Flag

Add to `HybridTransformerConfig`:

```python
class HybridTransformerConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        # ... existing parameters ...
        
        # Learnable retrieval gate
        self.use_gated_retrieval = kwargs.pop("use_gated_retrieval", False)
        
        super().__init__(**kwargs)
```

#### 4. Gate Logging

Add logging method to track gate values during training:

```python
def log_gate_value(self, step: int):
    """Log retrieval gate value for debugging."""
    if self.use_gated_retrieval and self.g_retrieve is not None:
        gate_value = torch.sigmoid(self.g_retrieve).item()
        if step % 1000 == 0:
            print(f"Step {step}: Retrieval gate = {gate_value:.4f}")
        return gate_value
    return None
```

#### 5. Ensure Gate is Trainable

The gate parameter should automatically be included in model parameters since it's created with `nn.Parameter()`. Verify with:

```python
# Check trainable parameters
for name, param in model.named_parameters():
    if 'g_retrieve' in name:
        print(f"Gate parameter: {name}, requires_grad={param.requires_grad}")
```

## Usage Examples

### Basic Usage

```python
# 1. Enable gated retrieval in config
config = HybridTransformerConfig(
    use_external_memory=True,
    use_gated_retrieval=True,  # Enable learnable gate
    # ... other parameters
)

# 2. Create model
model = HybridTransformerModel(config, tokenizer)

# 3. Set retrieval threshold (optional, default is 0.5)
model.external_memory.set_threshold(0.7)  # More selective retrieval

# 4. Training - gate learns when to retrieve
for batch in dataloader:
    logits, loss, memory_info = model(batch['input_ids'])
    loss.backward()
    optimizer.step()
    
    # Log gate value every 1000 steps
    if step % 1000 == 0:
        for layer in model.model.layers:
            if hasattr(layer, 'hybrid_attn'):
                gate_val = layer.hybrid_attn.log_gate_value(step)

# 5. Inference - gate controls retrieval
model.eval()
with torch.no_grad():
    logits, loss, memory_info = model(test_input)
    # Retrieval only happens if sigmoid(g_retrieve) > threshold
```

### Adjusting Threshold

```python
# Make retrieval more selective (higher threshold)
model.external_memory.set_threshold(0.8)

# Make retrieval less selective (lower threshold)
model.external_memory.set_threshold(0.3)

# Disable retrieval completely
model.external_memory.set_threshold(1.0)

# Always retrieve (if gate > 0)
model.external_memory.set_threshold(0.0)
```

## Benefits

1. **Adaptive Retrieval** - Model learns when external memory is helpful
2. **Speed Improvement** - Skip unnecessary retrievals during inference
3. **Better Reasoning** - Focus retrieval on complex queries
4. **Configurable** - Threshold can be adjusted per use case

## Training Behavior with Context Caching

- **Training Mode**: Always performs retrieval, but weights output by gate value
  - Allows gradient flow to learn optimal gate value
  - Formula: `output = sigmoid(g_retrieve) * external_memory_output`
  - Caches retrieved context: `cached_context = external_memory_output.detach()`

- **Inference Mode**: Adaptive retrieval with context caching
  - If `sigmoid(g_retrieve) > threshold`: 
    - Perform fresh retrieval
    - Update cache: `cached_context = new_retrieval.detach()`
  - Else if `cached_context` exists:
    - Reuse cached context (model learns when context is still valid)
    - No retrieval overhead!
  - Else:
    - Skip retrieval (no cache available yet)

### Context Caching Benefits

1. **Smarter Inference**: Model learns when to update vs reuse context
2. **Reduced Latency**: Avoid retrieval when cached context is still relevant
3. **Adaptive Updates**: Gate value indicates when context becomes outdated
4. **Memory Efficient**: Only stores last retrieved context per layer

## Implementation Checklist

- [x] Add threshold parameter to ExternalMemoryBank
- [x] Add set_threshold() and get_threshold() methods
- [ ] Add g_retrieve parameter to HybridMemoryAttention
- [ ] Add use_gated_retrieval config flag
- [ ] Implement gated retrieval logic in forward pass
- [ ] Add gate logging method
- [ ] Test with training loop
- [ ] Verify gate is trainable
- [ ] Update documentation

## Notes

- The gate value is initialized at 0.5 (sigmoid(0.5) ≈ 0.62)
- During training, the gate learns to increase for helpful retrievals and decrease for unhelpful ones
- The threshold provides a hard cutoff during inference for speed
- This feature is optional and can be disabled by setting `use_gated_retrieval=False`
