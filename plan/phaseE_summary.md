# Phase E Summary: Xeon Decode/ Prefill Integration

Date: 2026-05-16

## Results

### Integration architecture

Three new components added to the xeon decode pipeline:

1. **`ds4_xeon_matvec_vpdpbusd/ vpdpwssd`** (`ds4_xeon.c`) — VNNI matvec functions with row-range parameters, designed for caller-side `ds4_parallel_for` dispatch.

2. **`ds4_xeon_ffn_decode_one`** (`ds4_xeon.c`) — Single token, single layer FFN using pre-dequantized weights:
   - INT8 quantize input → VPDPBUSD gate → VPDPBUSD up → SwiGLU → INT16 quant mid → VPDPWSSD down → accumulate

3. **`ds4_xeon_decode_token`** (`ds4.c`) — Full decode function: embedding → 43 layers × (CPU attention + FFN) → LM head. CPU attention path copied verbatim from `layer_forward_raw_swa_one` for KV cache correctness.

### Build and run

```
make cpu  # builds successfully
./ds4 --backend xeon -p hi -n 3
# Compiles, links, runs without errors
# prefill: 3.27 t/s, generation: 3.60 t/s
```

Generation speed unchanged from CPU path (3.43→3.60 t/s) because the xeon FFN is not yet activated — pre-dequant weight loading requires expert-level lazy caching before the VNNI path can be used.

### Blocked: pre-dequant startup cost

Pre-dequantizing all 256 experts for a single layer takes ~6.9 seconds (Phase B data). For 43 layers, that's ~300 seconds of startup. This is unacceptable for interactive use.

**Required fix (Phase F)**: Expert-level lazy pre-dequant. Only pre-dequantize the experts actually selected by the router for each token. For a typical prompt, ~46/256 experts are used per layer (per git history commit `3e3d73f`). The pre-dequant cache is populated incrementally as different experts are accessed across tokens.

### Code structure (ready for VNNI activation)

```
ds4_session_sync xeon branch:
  if (s->predequant.gate_up[0])     ← buffer allocated = ready
    ds4_xeon_decode_token(...)       ← VNNI FFN (when pre-dequant populated)
  else
    forward_token_raw_swa_cpu_decode_scratch(...)  ← CPU fallback

ds4_xeon_decode_token:
  for each layer:
    [CPU attention — known correct]
    if (predequant ready for this layer):
      ds4_xeon_ffn_decode_one(...)   ← VNNI FFN
    else:
      layer_ffn_one_decode_scratch(...)  ← CPU FFN
```

When expert-level lazy pre-dequant is implemented (Phase F), changing one condition activates the VNNI path.

## Files changed

| File | Change |
|------|--------|
| `ds4_xeon.c` | Added matvec_vpdpbusd, matvec_vpdpwssd, ffn_decode_one + workers |
| `ds4_xeon.h` | Added function declarations |
| `ds4.c` | Added ds4_xeon_decode_token(), modified ds4_session_create (predequant init), modified ds4_session_sync (xeon decode dispatch) |

## Next Steps (Phase F)

1. Expert-level lazy pre-dequant (only dequant needed experts, not all 256)
2. Pre-dequant cache eviction policy (LRU across layers)
3. Background pre-dequant thread (hide latency)
4. Wire VNNI FFN into prefill_xeon_graph
5. Remove OpenMP dependency, NUMA cleanup
