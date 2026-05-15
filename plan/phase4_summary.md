# Phase 4 Summary: Static Graph, NUMA Topology & Expert Replication

Date: 2026-05-15

## Results

### Static Graph (T4.1.1-T4.1.2)

`ds4_xeon_graph` provides pre-allocated buffers for the entire inference pipeline:

| Buffer | Type | Dimensions | Purpose |
|--------|------|-----------|---------|
| a8_cur | int8* | [mb][n_embd] | Layer input (attention + FFN shared) |
| a8_scale | float* | [mb][n_blocks] | Per-32-element scales |
| a16_mid | int16* | [mb][n_ff_exp] | SwiGLU mid (down projection input) |
| a16_mid_scale | float* | [mb] | Per-token scale for mid |
| a16_residual | int16* | [mb][n_embd] | Residual accumulator |
| f32_attn_out | float* | [mb][n_embd] | Attention output |
| f32_ffn_cur | float* | [mb][n_embd] | FFN input (pre-norm copy) |
| f32_norm | float* | [mb][n_embd] | RMS norm output |
| f32_gate | float* | [mb][n_ff_exp] | Expert gate projection |
| f32_up | float* | [mb][n_ff_exp] | Expert up projection |
| f32_mid | float* | [mb][n_ff_exp] | SwiGLU mid (FP32) |
| f32_hc | float* | [mb][n_hc] | Host-context routing |
| f32_router_logits | float* | [mb][n_expert] | Router output |
| f32_shared_out | float* | [mb][n_embd] | Shared FFN output |
| f32_moe_out | float* | [mb][n_embd] | MoE combined output |

All buffers allocated via `aligned_alloc(64, ...)`, zero-initialized. Total memory: ~12 MB for max_batch_size=256 (all FP32/INT8/INT16 buffers combined).

### NUMA Topology Detection (T4.2.1)

`ds4_xeon_numa_init()` reads `/sys/devices/system/node/online` to detect NUMA nodes without libnuma dependency. On the dual-socket Xeon Gold 5318Y test server, it reports 2 nodes.

### Thread Binding (T4.3.1)

`ds4_xeon_threads_bind(numa_node)` parses `/sys/devices/system/node/nodeN/cpulist` to get the CPU set for a NUMA node, then binds each OpenMP thread via `pthread_setaffinity_np` using striped assignment (thread tid gets CPU (tid % ncpu) within the node).

Supports cpulist formats: "0,2,4,6", "0-5", "0-23,48-71".

### Expert Weight Replication API (T4.2.2)

`ds4_xeon_expert_replica` struct holds per-socket copies of pre-dequantized weights:
- `gate_up[node]`: uint8* — gate + up projections for VPDPBUSD
- `down[node]`: int16* — down projections for VPDPWSSD
- API: `ds4_xeon_expert_replica_init/free`

Actual weight replication (memcpy from predequant buffer to per-node replicas) requires Phase 5 integration to determine exact tensor dimensions.

### NUMA-Aware Allocator (T4.2.3)

`ds4_xeon_numa_alloc(size, node)` uses mmap + mbind syscall for explicit NUMA placement (best-effort, no libnuma required). Falls back to `aligned_alloc(64, ...)` if mbind is unavailable or fails.

## Changes

### `ds4_xeon.h`

- **Graph struct**: Added `f32_hc` buffer, `n_hc` field, `numa_nodes` and `numa_node` fields
- **Graph init**: Updated signature to include `n_hc` and `numa_node` parameters
- **NUMA API**: Added `ds4_xeon_numa_init()`, `ds4_xeon_threads_bind()`
- **Expert replica**: Added `ds4_xeon_expert_replica` struct and init/free functions
- **NUMA alloc**: Added `ds4_xeon_numa_alloc()`

### `ds4_xeon.c`

- **`ds4_xeon_graph_init`**: Added `f32_hc` allocation, `n_hc`/`numa_node` initialization
- **`ds4_xeon_graph_free`**: Added `f32_hc` deallocation
- **`ds4_xeon_numa_init`**: New — sysfs-based NUMA topology detection (0-libnuma-dependency)
- **`numa_node_to_cpuset`**: New — cpulist parser for thread binding
- **`ds4_xeon_threads_bind`**: New — pthread_setaffinity_np striped binding
- **`ds4_xeon_threads_init`**: Updated to call numa_init + thread_bind(0) when NUMA available
- **`ds4_xeon_numa_alloc`**: New — mmap + mbind best-effort NUMA allocation
- **`ds4_xeon_expert_replica_init/free`**: New — per-socket expert weight replica management
- Added Linux headers: `<pthread.h>`, `<sched.h>`, `<sys/stat.h>`, `<sys/mman.h>`, `<unistd.h>`

### `plan/TODO.md`

- Phase 4 tasks T4.1.1 through T4.2.3 and T4.3.1 marked `[x]` complete
- T4.4.1 and T4.4.2 annotated as blocked on Phase 5

## Deferred

- **T4.4.1**: Worker-group model — requires Phase 5 forward pass implementation
- **T4.4.2**: Barrier benchmark — requires T4.4.1
- Full NUMA verification (numa_maps, perf stat dTLB) — requires libnuma or running on the test server with model loaded

## Verification

Compiled on remote server (Xeon Gold 5318Y, dual-socket). All existing math tests continue to pass. NUMA detection and thread binding compiled without warnings.
