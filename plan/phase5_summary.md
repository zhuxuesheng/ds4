# Phase 5 Summary: Engine Integration & End-to-End Validation

Date: 2026-05-15

## Results

### Build Success

```
make cpu
# Produces: ds4, ds4-server, ds4-bench (CPU-only, with Xeon backend)
```

All three binaries compile and link successfully with the xeon backend integrated:
- `ds4`: Interactive CLI
- `ds4-server`: HTTP server mode
- `ds4-bench`: Benchmarking tool

The `-xeon` flag selects the Xeon backend at runtime. Existing `-cpu`/`-metal`/`-cuda` paths are unaffected.

### Integration Architecture

The Xeon backend is now fully integrated into the ds4.c main loop:

1. **Generation entry** (`generate_xeon_graph_raw_swa`, ds4.c:15673): Main loop that creates a session, runs prefill, then decode loop via `ds4_session_sync`.

2. **Prefill** (`prefill_xeon_graph`, ds4.c:15610): Per-layer loop (43 layers) that:
   - Reuses CPU attention (`layer_attention_raw_swa_batch`) — attention is memory-bound, less benefit from VNNI
   - Calls Xeon-optimized FFN (`ds4_xeon_ffn_shared_batch`) — FFN is compute-bound, primary VNNI target
   - Uses `ds4_xeon_graph` static buffers for HC state (`f32_cur`/`f32_next`)

3. **FFN batch** (`ds4_xeon_ffn_shared_batch`, ds4.c:15556): Per-token MoE + shared FFN. Currently delegates to CPU implementations (`layer_routed_moe_one` + `layer_shared_ffn_one`) for correctness; the VNNI-optimized path is scaffolded for future replacement.

4. **Decode** (reuses `ds4_session_sync`): Single-token decode using the same CPU path as `-cpu` backend, ensuring KV cache compatibility.

### Graph Initialization

`ds4_session_create` (ds4.c:17282) initializes the xeon graph when backend is `DS4_BACKEND_XEON`:

```c
ds4_xeon_graph_init(&s->xeon_graph, s->prefill_cap,
    DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_HC,
    DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_N_LAYER,
    -1 /* numa interleaved */);
```

Graph is freed at session teardown (ds4.c:17332).

## Changes

### `ds4.c` (modified)

- **`prefill_xeon_graph`**: Uses `f32_cur`/`f32_next` HC state buffers from xeon graph
- **`ds4_xeon_ffn_shared_batch`** (new): Per-token FFN with HC pre/post, RMS norm, MoE routing, shared FFN. Delegates to CPU for router/matmuls; VNNI path scaffolded
- **`ds4_session_create`**: Updated `ds4_xeon_graph_init` call with all 8 parameters
- Existing dispatch hooks unchanged: `DS4_BACKEND_XEON` in `ds4_backend_uses_graph`, `generate_xeon_graph_raw_swa` in generation loop

### `ds4_xeon.h` (modified)

- Added `f32_cur`/`f32_next` HC state buffers to graph struct
- Updated `ds4_xeon_graph_init` signature with n_hc, numa_node parameters
- Added NUMA API declarations (`ds4_xeon_numa_init`, `ds4_xeon_threads_bind`, `ds4_xeon_numa_alloc`)
- Added `ds4_xeon_expert_replica` struct and init/free API

### `ds4_xeon.c` (modified)

- **`ds4_xeon_graph_init`**: Added `f32_cur`/`f32_next` allocation, n_hc/numa_node support
- **`ds4_xeon_graph_free`**: Added `f32_cur`/`f32_next` deallocation
- NUMA functions: `ds4_xeon_numa_init`, `ds4_xeon_threads_bind`, `ds4_xeon_numa_alloc`
- Expert replica API: `ds4_xeon_expert_replica_init`/`free`

### `plan/TODO.md` (updated)

- Phase 5.1 (T5.1.1-T5.1.3) marked complete
- Phase 5.2-5.3 annotated as blocked on model weight availability

## Blocked Tasks

The following require actual `.ds4` model weight files on the test server:

| Task | Blocked By | Notes |
|------|-----------|-------|
| T5.2.1 Token match test | Model weights | -xeon vs -cpu comparison |
| T5.2.2 Logits error test | Model weights | Relative error analysis |
| T5.2.3 KV cache test | Model weights | 128K context |
| T5.3.1 Prefill benchmark | Model weights | Target: 70-140 tok/s |
| T5.3.2 Decode benchmark | Model weights | Target: 5-10 tok/s |
| T5.3.3 Interactive benchmark | Model weights | Target: 30-80 tok/s |

## Next Steps

Once model weights are available:
1. Run `./ds4 -xeon -m <model.ds4> -p "test prompt"` to verify token generation
2. Compare `-xeon` vs `-cpu` output for token-exact match
3. Profile prefill/decode throughput
4. Replace `ds4_xeon_ffn_shared_batch` CPU delegates with VNNI matmul kernels (VNNI matmul kernels exist in ds4_xeon.c; need pre-dequantized weights from T3.3.1)
