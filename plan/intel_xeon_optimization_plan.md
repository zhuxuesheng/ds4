# DeepSeek V4 Flash - Intel Xeon Dual-Socket Optimization Plan

## 1. Background & Motivation
The `ds4` inference engine currently targets macOS (Metal) and Linux (CUDA). The CPU path is currently an unoptimized reference implementation. However, for a dual-socket server like the Intel Xeon Gold 5318Y (Ice Lake-SP) with 48 physical cores, 512GB RAM, and AVX-512 VNNI support, the CPU is capable of acting as a high-performance inference engine for DeepSeek V4 Flash.

DeepSeek V4 heavily relies on MLA and MoE architectures, which produce extreme activation outliers — notably in the SwiGLU intermediate activations (heavy-tailed distribution). A naive INT8 activation path risks severe precision degradation at these outlier points.

However, most activation tensors in the model are well-behaved: every sub-layer input is bounded by a preceding RMS Norm, and the reference CPU path already uses per-block INT8 quantization (`quantize_q8_0_activation`) extensively for attention projections. The outlier problem is concentrated in a single location per FFN block: the **SwiGLU mid activation** (`sigmoid(gate) × up`).

We therefore adopt a **Hybrid Precision VNNI Static Graph Architecture**:

- **W4A8 / W2A8** for all matrix multiplications where the activation input is RMS-Norm-bounded
- **W4A16 / W2A16** only for the FFN down projection (SwiGLU mid input)
- **FP32** for control layers (MLA compressor, router, HC sinkhorn)

## 2. Hardware Capabilities & Theoretical Limits (Corrected)

### 2.1 Intel Xeon Gold 5318Y (Ice Lake-SP) Dual-Socket

| Metric | Value |
|---|---|
| Cores (total) | 48 physical (24 × 2 sockets), 96 threads |
| Base / All-Core Frequency | 2.1 GHz / ~2.5 GHz |
| AVX-512 Sustained Frequency | ~1.5-1.8 GHz (throttling under heavy 512-bit load) |
| Memory Bandwidth (sustained) | ~260-300 GB/s (8-ch DDR4-2933 × 2 sockets) |
| L3 Cache | ~36 MB per socket |
| NUMA | 2 nodes, cross-socket via UPI (~11.2 GT/s × 3 links) |

### 2.2 Compute Throughput (Corrected)

The plan originally estimated ~32 TFLOPS equivalent. This was incorrect — it conflated INT16 VNNI throughput with GPU-style tensor-core numbers. **Corrected math:**

#### VPDPWSSD (INT16 VNNI) — the W4A16/W2A16 workhorse

```
Instruction: _mm512_dpwssd_epi32
  → 2 multiply-add per lane × 16 lanes = 32 INT16 MACs per instruction
Throughput: 1 instruction / cycle / core (Ice Lake)
Per core at 2.1 GHz: 32 × 2.1e9 = 67.2 GMACs/s
48 cores at base freq: 48 × 67.2 = 3.2 TMACs/s
× 2 ops per MAC: 6.4 TOPS
```

| Scenario | Effective TOPS |
|---|---|
| Theoretical peak (2.1 GHz, no throttle) | **6.4 TOPS** |
| Realistic (1.5-1.8 GHz AVX-512 freq) | **4.0-5.0 TOPS** |

#### VNNI Throughput Comparison (48 cores, base 2.1 GHz)

| Instruction | INT8 MACs/inst | Throughput/core | Total TMACs | Total TOPS |
|---|---|---|---|---|
| `VPDPBUSD` (INT8 VNNI) | 64 | 134.4 GMACs | 6.45 | **12.9** |
| `VPDPWSSD` (INT16 VNNI) | 32 | 67.2 GMACs | 3.23 | **6.4** |
| FMA (FP32) | 16 | 67.2 GFLOPS | 3.23 | 3.2 TFLOPS |

**Key insight**: INT8 VNNI offers 2× throughput over INT16 VNNI. For Q8_0 attention layers, using `VPDPBUSD` with INT8 activations can double the attention projection throughput compared to the INT16 path.

### 2.3 Model Architecture

From `ds4.c`:
```
DS4_N_LAYER            = 43
DS4_N_EMBD             = 4096
DS4_N_VOCAB            = 129280
DS4_N_EXPERT           = 256
DS4_N_EXPERT_USED      = 6
DS4_N_EXPERT_SHARED    = 1
DS4_N_FF_EXP           = 2048
DS4_N_HC               = 4
```

**Activated parameters per token:**

| Component | Per-layer | ×43 layers | Weight Format |
|---|---|---|---|
| MoE FFN (6 experts × gate/up/down) | 6 × 3 × (4096×2048) = 151M | 6.5B | Q4_K (~4.5 bit) |
| Attention (Q_A, KV, Output_B, compressors) | ~25M | 1.1B | Q8_0 / F16 |
| Shared FFN (gate, up, down) | ~25M | 1.1B | Q8_0 |
| Router | ~1M | 43M | F16 |
| Embedding + Output head | — | 1.0B | F16 / Q8_0 |
| **Total activated** | **~202M** | **~9.7B params** | |

**GGUF format sizes** (from `antirez/deepseek-v4-gguf`):
- **Q4KExperts**: 165 GB (4-bit routed experts, Q8_0 attention projections, F16 control layers)
- **IQ2XXS**: 86.7 GB (2-bit routed experts, same otherwise)

### 2.4 Activation Precision: Why Most Activations Can Be INT8

The key safety guarantee is **RMS Norm reset**: every sub-layer (Attention, FFN) begins with an RMS Norm that bounds the activation distribution to unit variance. The activation flowing into each matmul is reliably bounded, making per-block INT8 quantization safe.

The **only** activation in the model with a heavy-tailed, unbounded distribution is the SwiGLU intermediate (`mid = sigmoid(gate) × up`), which feeds the FFN down projection. This is the singular point where INT16 is required.

**Per-layer activation precision assignment (Prefill path):**

```
RMS Norm output ──→ INT8 per-block ──→ × Q/K/V weights (Q8_0)  → VPDPBUSD
RMS Norm output ──→ INT8 per-block ──→ × gate/up weights (Q4_K) → VPDPBUSD
SwiGLU mid ───────→ INT16 per-token ─→ × down weights (Q4_K)   → VPDPWSSD
Residual path ────→ INT16 accumulator (FP32 → quantize back to INT8)
Router / HC ──────→ FP32 (control layers, small dims)
```

| Activation Position | Quantization | VNNI Instruction | Why |
|---|---|---|---|
| RMS Norm output (attn_norm, ffn_norm) | **INT8 per-block** | VPDPBUSD | Norm guarantees bounded distribution |
| Attention Q/K/V/O projection input | **INT8 per-block** | VPDPBUSD | Norm output, linear layer input |
| Attention internal (QK^T, softmax) | **FP32** | FMA | Small dims, non-linear ops, skip quant |
| FFN gate/up projection input | **INT8 per-block** | VPDPBUSD | Norm output, before SwiGLU |
| **SwiGLU mid (sigmoid(gate)×up)** | **INT16 per-token** | VPDPWSSD | Heavy-tailed, the only INT16 point |
| FFN down projection output | **INT8 per-block** | VPDPBUSD | Linear transform of mid, safe |
| Router gate logits | **FP32** | FMA | Small (4096×256), precision-critical |
| HC sinkhorn | **FP32** | FMA | Control layer, 4-dim |
| Residual accumulator | **INT16** | — | Accumulates over 43 layers |
| Embedding / LM head | **FP32** | FMA | First/last layer, high precision |

**Compute instruction mix per FFN block** (per layer, 6 experts):

| Operation | Weights | Activations | Instruction | % of MACs |
|---|---|---|---|---|
| gate projection | Q4_K (unpack→u8) | INT8 | VPDPBUSD | 30% |
| up projection | Q4_K (unpack→u8) | INT8 | VPDPBUSD | 30% |
| down projection | Q4_K (unpack→u16) | INT16 | VPDPWSSD | 30% |
| Attention projections | Q8_0 | INT8 | VPDPBUSD | 10% |

~70% of MACs use INT8 VNNI (2× throughput), ~30% use INT16 VNNI. Effective throughput:

```
Weighted throughput: 0.7 × 12.9 TOPS + 0.3 × 6.4 TOPS ≈ 9.0 + 1.9 = ~10.9 TOPS (base)
Realistic (AVX-512 throttle): ~0.7 × ~9 TOPS + 0.3 × ~4.5 TOPS ≈ ~7.6 TOPS
```

### 2.5 Realistic Prefill Performance

```
GFLOPs per token: 9.7B MACs × 2 ops = ~19.4 GFLOPs
Effective mixed VNNI throughput: ~7.5 TOPS (realistic, AVX-512 throttled)
```

| Variant | Compute Path | Prefill Speed |
|---|---|---|
| W4A16 (pure INT16, baseline) | All layers via VPDPWSSD | **~230 tok/s** |
| **W4A8 mixed (INT8 + INT16 mid)** | VPDPBUSD for gate/up/attn, VPDPWSSD for down | **~500-600 tok/s** |
| W2A8 mixed (IQ2XXS + INT8/INT16) | Same compute, higher dequant overhead | **~400-500 tok/s** |

### 2.6 Realistic Decode Performance

Decode (single token generation) is **memory-bandwidth-bound**. Each token must read activated weight bytes from DRAM. Activation precision (INT8 vs INT16) has negligible impact on decode — the bottleneck is weight reads, not activation traffic.

**Weight bytes read per token (Q4KExperts, 165GB):**

| Component | Per-layer bytes | ×43 layers |
|---|---|---|
| MoE FFN (6 experts, Q4_K) | 151M × 0.56B = 85 MB | 3.65 GB |
| Attention (Q8_0/F16) | ~20 MB | 0.86 GB |
| Shared FFN (Q8_0) | ~25 MB | 1.08 GB |
| Embedding + Output + Router | — | ~2.0 GB |
| **Total** | **~132 MB** | **~7.6 GB** |

**Weight bytes read per token (IQ2XXS, 86.7GB):**

| Component | Per-layer bytes | ×43 layers |
|---|---|---|
| MoE FFN (6 experts, IQ2XXS) | 151M × 0.28B = 42 MB | 1.81 GB |
| Same non-expert overhead | ~47 MB | 3.94 GB |
| **Total** | **~89 MB** | **~5.75 GB** |

At sustained 260 GB/s, with ~70% effective utilization after NUMA overhead (~180 GB/s):

| Variant | Weight Read/Token | Decode Speed | Note |
|---|---|---|---|
| Q4KExperts (165GB) | ~7.6 GB | **~24 tok/s** | Compute precision irrelevant, memory-bound |
| IQ2XXS (86.7GB) | ~5.75 GB | **~31 tok/s** | Memory-bound |

**Decode is not affected by activation precision choice**: reading ~7.6 GB of weights dominates the ~0.1 GB of activation data. W4A8 vs W4A16 changes only activation bytes, not weight bytes.

### 2.7 Performance Summary: Target vs. Reality

| Metric | Original Plan Target | Pure W4A16 (INT16) | **W4A8 Mixed (INT8 + INT16 mid)** |
|---|---|---|---|
| Peak VNNI throughput | ~32 TFLOPS | 4-5 TOPS | **~7-8 TOPS** |
| Prefill (Q4_K experts) | 500-800 tok/s | ~230 tok/s | **~500-600 tok/s** |
| Prefill (IQ2XXS experts) | 1000-1500 tok/s | ~200 tok/s | **~400-500 tok/s** |
| Decode (Q4KExperts) | not specified | ~24 tok/s | ~24 tok/s |
| Decode (IQ2XXS) | not specified | ~31 tok/s | ~31 tok/s |

**Key takeaway**: The W4A8/W2A8 mixed precision approach brings prefill throughput back into the plan's original target range on the current Ice Lake hardware, without requiring an AMX upgrade. The conservative INT16 for SwiGLU mid prevents precision degradation at the only risky activation point.

## 3. Scheme Design: Hybrid-Precision VNNI Graph

We will replace the dynamic memory allocation (`ds4_cpu_decode_scratch`) with a static, NUMA-aware execution graph (`ds4_xeon_graph`).

### Phase 1: Static Graph & NUMA Architecture
- **`ds4_xeon_graph`:** Pre-allocate all intermediate activation tensors for the maximum batch size (e.g., 1024 or 2048). Buffers are typed by precision:
  - `int8_t` buffers for RMS Norm outputs, gate/up inputs, attention inputs (~90% of activation volume)
  - `int16_t` buffers for SwiGLU mid, residual accumulator (~10% of activation volume)
  - `float` buffers for router logits, HC sinkhorn (tiny, FP32)
- **NUMA-Aware Expert Parallelism:** For the 256 MoE experts, the graph will statically partition the workload across the 96 threads. Threads bound to Socket 0 will strictly process experts resident in Socket 0's local memory, completely eliminating cross-socket UPI traffic during the heavy MoE routing phase.

### Phase 2: Mixed VNNI Compute Kernels (VPDPBUSD + VPDPWSSD)

The core data path uses hybrid integer precision, selecting the VNNI instruction based on the activation tensor's risk profile:

**Primary kernel — VPDPBUSD (INT8 VNNI, ~70% of MACs):**
- **Activations:** `int8_t` with per-32-element block scaling (Q8_0 style, same as `quantize_q8_0_activation`)
- **Weights (Q4_K/IQ2XXS):** Dequantized on-the-fly from 4-bit/2-bit → `uint8_t` via shuffle/shift in registers
- **Weights (Q8_0):** Already `uint8_t`, direct load
- **Compute:** `_mm512_dpbusd_epi32` — 4 MACs/lane × 16 lanes = 64 MACs/instruction
- **Scale:** `float` rescale (weight_d * weight_scale * activation_block_scale) applied to INT32 accumulator

**Fallback kernel — VPDPWSSD (INT16 VNNI, ~30% of MACs):**
- **Activations:** `int16_t` with per-token scaling (A16 style, same as `ds4_xeon_quantize_a16`)
- **Weights (Q4_K):** Dequantized 4-bit → `int16_t`
- **Compute:** `_mm512_dpwssd_epi32` — 2 MACs/lane × 16 lanes = 32 MACs/instruction
- **Used only for:** FFN down projection (SwiGLU mid input)

**Control kernel — FP32 FMA:**
- Router gate (4096×256), HC sinkhorn (4-dim), embedding/LM head
- Standard AVX-512 FMA for maximum precision

**Critical performance note**: For Q4_K/IQ2XXS weights, dequantization uses shuffle/unpack/convert instructions that compete for execution ports 0/5 with VNNI. If the W4A8 kernel's dequant overhead is too high, **pre-dequantizing expert weights to contiguous INT8 buffers** at load time eliminates this overhead. Expert weights expand from ~15 GB (Q4_K) to ~30 GB (INT8), well within the 512GB RAM budget.

### Phase 3: Per-Layer Precision Dispatch

The graph execution dispatches each matmul to the correct kernel based on activation type:

```
For each layer il = 0..42:
  ┌─ Attention ─────────────────────────────────────────┐
  │ RMS Norm(input) → f32                                │
  │ quantize_a8(norm_out)      → int8 per-block          │
  │ × attn_q_a (Q8_0)         → VPDPBUSD → f32          │
  │ × attn_kv  (Q8_0)         → VPDPBUSD → f32          │
  │ [MLA compute in FP32: QK^T, softmax, weighted sum]   │
  │ quantize_a8(attn_out)      → int8 per-block          │
  │ × attn_output_b (Q8_0)    → VPDPBUSD → f32          │
  └──────────────────────────────────────────────────────┘
  ┌─ MoE FFN ───────────────────────────────────────────┐
  │ RMS Norm(residual) → f32                             │
  │ quantize_a8(norm_out)      → int8 per-block          │
  │ Router: × gate_inp (F16)   → FMA → f32 (softmax)    │
  │                                                      │
  │ Shared FFN:                                          │
  │   × ffn_gate_shexp (Q8_0) → VPDPBUSD → f32          │
  │   × ffn_up_shexp   (Q8_0) → VPDPBUSD → f32          │
  │   SwiGLU → f32                                       │
  │   × ffn_down_shexp (Q8_0) → VPDPBUSD → f32 (shared) │
  │                                                      │
  │ Routed Experts (6 of 256, per selected expert):      │
  │   × ffn_gate_exp (Q4_K) → VPDPBUSD → f32 (gate)     │
  │   × ffn_up_exp   (Q4_K) → VPDPBUSD → f32 (up)       │
  │   SwiGLU(gate, up) → mid (heavy-tailed)              │
  │   quantize_a16(mid)        → int16 per-token ◄─ ONLY INT16 │
  │   × ffn_down_exp (Q4_K)   → VPDPWSSD → f32 (down)   │
  │                                                      │
  │ residual = input + shared + Σ(down_i × expert_weight) │
  └──────────────────────────────────────────────────────┘
```

This approach guarantees that only the SwiGLU mid activation — the single heavy-tailed distribution in the model — uses INT16. All other activations are RMS-Norm-bounded and safe for per-block INT8 quantization.

## 4. Development Roadmap

To ensure the original engine's reference implementation remains completely untouched while maximizing server performance, we will implement this via a dedicated `DS4_BACKEND_XEON`. Development will proceed in the following test-driven phases:

### Step 1: Backend Isolation & INT8/INT16 VNNI Micro-Benchmarks
**Goal:** Verify both VPDPBUSD (INT8) and VPDPWSSD (INT16) VNNI throughput before rewriting the engine.
1. **Define the Backend:** Add `DS4_BACKEND_XEON` to `ds4_backend` enum. Create an isolation layer (e.g., `ds4_xeon.h` / `ds4_xeon.c`).
2. **Build VNNI Micro-kernels:** Write pure AVX-512 intrinsics for both instructions:
   - `matmul_w4a8_vnni`: `_mm512_dpbusd_epi32` (INT8 VNNI, 64 MACs/inst)
   - `matmul_w4a16_vnni`: `_mm512_dpwssd_epi32` (INT16 VNNI, 32 MACs/inst)
3. **Micro-Benchmark Test:** Update `tests/ds4_xeon_matmul_bench.c` to test both.
   - *Test criteria:* Achieve **>9 TOPS** (VPDPBUSD) and **>4.5 TOPS** (VPDPWSSD), i.e., >70% of respective theoretical peaks.
   - *Compilation:* Ensure `-march=native -mprefer-vector-width=512` is used.

### Step 2: Activation Quantization Kernels
**Goal:** Implement per-block INT8 and per-token INT16 activation quantization.
1. **Per-block INT8 quant (`quantize_a8_per_block`):** Port `quantize_q8_0_activation` from `ds4.c` to AVX-512 — use `_mm512_reduce_max_ps` for block max finding instead of scalar loop.
2. **Per-token INT16 quant (`quantize_a16_per_token`):** Optimize existing `ds4_xeon_quantize_a16` with AVX-512 max reduction.
3. **Dequant + rescale:** Implement `dequant_a8_a16_add` for residual accumulation (INT8 per-block × scale → FP32 → accumulate → re-quantize).
4. **Validation Test:** `tests/ds4_xeon_math_test.c`.
   - *Test criteria:* Roundtrip fidelity: quantize FP32 → INT8/INT16 → dequant → FP32, assert cosine similarity >0.999. Validate SwiGLU mid INT16 vs INT8 SNR difference.

### Step 3: Unpacking & Q4_K/IQ2XXS Dequant Kernels
**Goal:** Efficiently unpack 4-bit/2-bit weights to uint8_t/int16_t in AVX-512 registers for VNNI consumption.
1. **Q4_K → uint8_t unpack:** AVX-512 shuffle/shift to expand nibbles to bytes. Output feeds `VPDPBUSD`.
2. **Q4_K → int16_t unpack:** Same, with sign extension. Output feeds `VPDPWSSD` (down projection only).
3. **IQ2XXS → uint8_t unpack:** AVX-512 gather for grid lookup + sign mask application. Currently scalar — vectorization is critical for W2A8 path.
4. **Scale extraction:** `ds4q_get_scale_min_k4` optimization — pre-decode packed 6-bit scales into a flat lookup table.
5. **Validation Test:** Compare dequantized weight values against reference `ds4.c` scalar dequant, assert bit-exact match.

### Step 4: The Static Graph & NUMA Topology (`ds4_xeon_graph`)
**Goal:** Pre-allocate all memory and map MoE experts to specific CPU sockets.
1. **Graph Definition:** Define the `ds4_xeon_graph` struct with pre-allocated buffers typed by precision (int8_t, int16_t, float) for max batch size.
2. **NUMA Binding:** Use `pthread_setaffinity_np` with `libnuma`. Pin threads 0-47 to Socket 0, threads 48-95 to Socket 1.
3. **Expert Partitioning:** Statically assign experts 0-127 to Socket 0 memory, experts 128-255 to Socket 1. Socket 0 threads only process Socket 0 experts.
4. **First-touch memory policy:** Allocate expert weight copies via `numa_alloc_onnode` to ensure local NUMA placement.

### Step 5: Engine Integration & End-to-End Validation
**Goal:** Wire the Xeon backend into the main `ds4_engine` execution flow (Prefill & Decode).
1. **Hook into Prefill/Decode:** Implement `prefill_xeon_graph` and `decode_xeon_graph`. Route inference calls when `DS4_BACKEND_XEON` is selected.
2. **End-to-End Integration Test:** Run a full prompt through the model using `tests/ds4_test.c`.
   - *Test criteria:* The `-xeon` backend must produce the exact same token stream as the `-cpu` reference backend for a complex reasoning prompt. Logits within 1e-3 relative tolerance.

### Step 6: IQ2XXS Vectorization & Pre-Dequant Optimization
**Goal:** Eliminate scalar bottlenecks in W2A8 path and optionally pre-dequant to skip runtime overhead.
1. **IQ2XXS vectorization:** Replace `ds4_xeon_vec_dot_iq2_xxs_vnni` scalar inner loop with AVX-512 gather + `VPDPBUSD`.
2. **Pre-dequant option:** Pre-dequantize all expert weights to `uint8_t` (for W4A8) or `int16_t` (for W2A8 down projection) at model load time. Memory cost: ~15 GB → ~30 GB (Q4_K→INT8), well within 512GB. Eliminates all dequant in the hot inference loop.

## 5. Architectural Shift: The "Plugin" Graph Model

To minimize modifications to the core `ds4.c` engine and facilitate seamless upstream merges, the Xeon backend will be treated as a first-class, "plugin-style" Graph Backend, mirroring the architectural pattern established by Metal and CUDA.

### The "Shadow" Execution Pattern
1. **Opaque Coupling:** `ds4.c` will interact with the Xeon backend via opaque pointers (`ds4_xeon_graph *`) defined in `ds4_xeon.h`. It will not export its internal structures (`ds4_model`, `ds4_weights`, `ds4_kv_cache`) to a shared private header.
2. **Minimal Hooks in `ds4.c`:**
   - **Initialization:** `ds4_session_create` will conditionally initialize the `ds4_xeon_graph` if the backend is `DS4_BACKEND_XEON`.
   - **Dispatch:** `ds4_session_sync` (or the equivalent prefill/decode loops) will intercept the execution path and dispatch to `ds4_xeon_graph_prefill` or `ds4_xeon_graph_eval_token`.
3. **Encapsulated Execution:** The entirety of the inference loop—including embedding lookup, layer iterations (Attention and MoE FFN), and KV cache management—will be self-contained within `ds4_xeon.c`.
4. **Context Passing:** A lightweight `ds4_xeon_model_context` struct will pass essential pointers (e.g., base memory map, raw weight pointers) and dimensions from `ds4.c` to `ds4_xeon.c` at runtime, ensuring the Xeon backend has the necessary data without requiring `#include` sharing of complex internal types.

This approach guarantees that `ds4.c` remains largely untouched (modifications kept to ~30-50 lines of dispatch logic), preserving the reference implementation's integrity while unlocking maximum performance on Xeon hardware.

## 6. Hardware Upgrade Path

The W4A8/W2A8 mixed precision approach brings prefill into the **~500-600 tok/s** range on the current Ice Lake hardware — matching the plan's original lower-bound target without requiring new hardware.

If higher throughput (1000+ tok/s) is needed:

1. **Sapphire Rapids / Granite Rapids upgrade**: AMX (Advanced Matrix Extensions) provides tile-based matrix multiply at 2048 INT8 ops/cycle/tile. A dual-socket SPR Xeon with AMX could achieve **50-100 TOPS** in practice, enabling **2000-4000 tok/s prefill**.

2. **Pre-dequantize expert weights**: On Ice Lake, pre-dequantizing Q4_K expert weights to INT8 at load time eliminates all dequant overhead. Cost: ~15 GB → ~30 GB for expert weights. Combined with W4A8, this could push prefill to **~600-700 tok/s**.

3. **Batch-fusion prefills**: For large batch prefill (1024+ tokens), group tokens by expert assignment and fuse expert computation, improving L3 cache hit rate and reducing redundant weight reads from DRAM.
