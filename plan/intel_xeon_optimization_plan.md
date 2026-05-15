# DeepSeek V4 Flash - Intel Xeon Dual-Socket Optimization Plan

## 1. Background & Motivation

The `ds4` inference engine currently targets macOS (Metal) and Linux (CUDA). The CPU path is currently an unoptimized reference implementation. However, for a dual-socket server like the Intel Xeon Gold 5318Y (Ice Lake-SP) with 48 physical cores, 512GB RAM, and AVX-512 VNNI support, the CPU is capable of acting as a high-performance inference engine for DeepSeek V4 Flash.

DeepSeek V4 heavily relies on MLA and MoE architectures, which produce extreme activation outliers — notably in the SwiGLU intermediate activations (heavy-tailed distribution). A naive INT8 activation path risks severe precision degradation at these outlier points.

However, most activation tensors in the model are well-behaved: every sub-layer input is bounded by a preceding RMS Norm, and the reference CPU path already uses per-block INT8 quantization (`quantize_q8_0_activation`) extensively for attention projections. The outlier problem is concentrated in a single location per FFN block: the **SwiGLU mid activation** (`sigmoid(gate) × up`).

We therefore adopt a **Hybrid Precision VNNI Static Graph Architecture**:

- **W4A8 / W2A8** for all matrix multiplications where the activation input is RMS-Norm-bounded
- **W4A16 / W2A16** only for the FFN down projection (SwiGLU mid input)
- **FP32** for control layers (MLA compressor, router, HC sinkhorn)

This is a **full Xeon-native inference architecture** — not merely a set of AVX-512 kernel patches. The value of this design lies in the system architecture (static graph, NUMA-aware scheduling, backend isolation, hybrid precision dispatch) as much as in the individual kernels. Real-world token throughput will be determined by end-to-end system bottlenecks (cache, NUMA, scheduling, dequant) rather than by theoretical VNNI peak throughput alone.

## 2. Hardware Capabilities & Theoretical Limits

### 2.1 Intel Xeon Gold 5318Y (Ice Lake-SP) Dual-Socket

| Metric | Value |
|---|---|
| Cores (total) | 48 physical (24 × 2 sockets), 96 threads |
| Base / All-Core Frequency | 2.1 GHz / ~2.5 GHz |
| AVX-512 Sustained Frequency | ~1.5-1.8 GHz (throttling under heavy 512-bit load) |
| Memory Bandwidth (theoretical) | ~375 GB/s (8-ch DDR4-2933 × 2 sockets) |
| Memory Bandwidth (sustained sequential) | ~260-300 GB/s |
| Memory Bandwidth (random access, MoE) | ~30-80 GB/s (see Section 2.8) |
| L3 Cache (LLC) | ~36 MB per socket (1.5 MB/core × 24) |
| L2 Cache | 1.25 MB per core |
| TLB (STLB) | 2048 entries (4K), 1024 entries (2M) |
| NUMA | 2 nodes, cross-socket via UPI (~11.2 GT/s × 3 links) |
| Page size | 4 KB default; 2 MB (transparent hugepage); 1 GB (explicit hugetlbfs) |

### 2.2 Compute Throughput

#### VPDPWSSD (INT16 VNNI)

```
Instruction: _mm512_dpwssd_epi32
  → 2 multiply-add per lane × 16 lanes = 32 INT16 MACs per instruction
Throughput: 1 instruction / cycle / core (Ice Lake)
Per core at 2.1 GHz: 32 × 2.1e9 = 67.2 GMACs/s
48 cores at base freq: 48 × 67.2 = 3.2 TMACs/s
× 2 ops per MAC: 6.4 TOPS
```

#### VPDPBUSD (INT8 VNNI)

```
Instruction: _mm512_dpbusd_epi32
  → 4 multiply-add per lane × 16 lanes = 64 INT8 MACs per instruction
Per core at 2.1 GHz: 64 × 2.1e9 = 134.4 GMACs/s
48 cores at base freq: 48 × 134.4 = 6.45 TMACs/s
× 2 ops per MAC: 12.9 TOPS
```

| Scenario | VPDPBUSD (INT8) | VPDPWSSD (INT16) |
|---|---|---|
| Theoretical peak (2.1 GHz) | **12.9 TOPS** | **6.4 TOPS** |
| Realistic (1.5-1.8 GHz AVX-512 freq) | **~9 TOPS** | **~4.5 TOPS** |
| With dequant overhead (~50% port contention) | **~4.5 TOPS** | **~2.3 TOPS** |

**Critical note**: the "~9 TOPS" realistic figure assumes a pure dense GEMM with pre-quantized INT8 weights and activations. In the real inference loop, on-the-fly Q4_K dequantization (shuffle, unpack, convert) competes for the same execution ports (0, 5) as VNNI, typically reducing effective throughput by 30-50%. This is why **pre-dequantizing weights at load time** is one of the highest-impact optimizations available.

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

**Caveat**: While per-block INT8 quantization is mathematically safe for RMS-Norm-bounded activations, DeepSeek models are known to be sensitive to quantization noise, especially in MoE router paths and long-context reasoning chains. Even cos-sim >0.999 does not guarantee that token generation won't drift — LLM errors undergo recursive amplification through:
- **Routing drift**: small perturbations in router logits → different expert selection → cascading trajectory divergence
- **Entropy collapse**: accumulated quantization noise along 43 layers can collapse the token distribution

The conservative strategy (INT16 for mid, FP32 for router) mitigates these risks. **Full INT8 for all activations should be validated end-to-end** before deployment — a single reasoning prompt producing 100+ tokens without divergence is the minimum acceptance test.

### 2.5 Real-world Bottleneck Analysis

This section addresses the gap between theoretical VNNI throughput and real-world MoE inference performance on CPU.

#### Bottleneck 1: MoE Random Memory Access

The theoretically calculated decode performance (24 tok/s) assumes streaming sequential access at full DRAM bandwidth. In reality, MoE expert access patterns are highly irregular:

```
Per token, per layer: 6 experts selected by router
  → 6 Q4_K blocks read (gate + up + down = 3 projections per expert)
  → These 6 experts are scattered across 256 experts in memory
  → Adjacent tokens typically select different expert subsets
  → Zero spatial locality, minimal temporal reuse
```

**Effective bandwidth under MoE random access: ~30-80 GB/s** (not the 260 GB/s sequential peak). This is the dominant factor limiting decode throughput.

#### Bottleneck 2: On-the-fly Dequantization

In the W4A8 kernel, Q4_K weights must be unpacked from nibbles to uint8_t before feeding `VPDPBUSD`. AVX-512 unpack instructions (shuffle, shift, permute, convert) compete with VNNI for execution ports 0 and 5:

```
Q4_K unpack per 256-element block:
  vpsrlw / vpand          → nibble extraction (port 5 only)
  vpshufb / vpermw        → reordering (port 5 only)  
  vpmovzxbw               → zero-extend to int16 (port 5)

VNNI: vpdpbusd            → also port 0/5

→ Significant port 5 contention
→ Real-world dequant overhead: 30-50% of total compute time
```

This is why many production CPU inference systems pre-dequantize weights at load time — trading memory for throughput is the right tradeoff when RAM is abundant.

#### Bottleneck 3: Cache Hierarchy

DeepSeek V4 Flash's expert working set per token is massive and cache-unfriendly:

- Per layer: 6 experts × 3 projections × ~4.2M weight elements = ~75M weight elements
- Total (all layers): ~3.2B weight elements accessed per token (scattered)
- L3 per socket: 36 MB — can hold only a tiny fraction of active weights
- Expert reuse between tokens: **very low** (router dynamically selects different experts)

The CPU never enters a steady-state GEMM regime. Instead, it's in a perpetual cold-start state where each matmul begins with weights missing from cache.

#### Bottleneck 4: Thread Synchronization (Barrier Overhead)

MoE inference involves multiple synchronization points per layer:

```
Per layer synchronization:
  1. Router compute (all threads) → barrier
  2. Expert dispatch (scatter tokens to expert workers) → barrier  
  3. Expert compute (per-expert worker groups) → barrier
  4. Shared FFN (all threads) → barrier
  5. Attention (all threads) → barrier
  6. Residual add (all threads) → barrier
```

At 43 layers × ~6 barriers/layer × ~5μs/barrier = ~1.3ms per token just in barrier overhead. On a 48-core dual-socket system, cross-socket barriers are particularly expensive (cache coherency protocol latency across UPI).

#### Bottleneck 5: TLB Pressure

With 9.7B activated parameters per token spread across 165GB of address space:

- 4 KB pages: 165GB / 4KB = ~41M potential pages → TLB thrashing inevitable
- 2 MB hugepages: 165GB / 2MB = ~82,500 pages → still far exceeds 2048-entry STLB
- 1 GB hugepages: 165GB / 1GB = 165 pages → **fits in STLB**

**Using 1GB hugetlbfs pages for model weights is essential** — without it, TLB miss rates will be catastrophic (>30% overhead).

#### Bottleneck 6: KV Cache in Long Context

As context length grows, KV cache access becomes a first-order cost in decode:

```
Per decode step, per layer:
  - Read all previous K/V pairs for attention
  - KV cache size: 2 × seq_len × n_kv_heads × head_dim × sizeof(fp16)
  - At 128K context: ~2 × 131072 × 1 × 128 × 2 bytes ≈ 64 MB per layer
  - 43 layers: ~2.75 GB KV cache read per decode step
```

At sustained effective bandwidth (~50 GB/s for mixed access), KV cache alone consumes ~55ms per decode step — this can exceed the weight-reading time and become the dominant decode bottleneck at long context.

### 2.6 Realistic Performance Estimates

**Methodology**: Start from the current scalar reference implementation (5.67 tok/s prefill, 3.70 tok/s decode), then apply conservative multipliers for each optimization.

**Prefill — from scalar reference to optimized VNNI:**

| Optimization | Multiplier | Rationale |
|---|---|---|
| Scalar → VNNI INT8 (gate/up/attention, ~70% of MACs) | ×3-5 | VPDPBUSD vs. scalar fp32 dot product |
| Scalar → VNNI INT16 (down projection, ~30% of MACs) | ×2-3 | VPDPWSSD vs. scalar |
| Static graph (eliminate malloc/free, pre-allocate) | ×1.2-1.4 | Removes per-token allocator overhead |
| NUMA locality (expert replication, local access only) | ×1.2-1.5 | Eliminates cross-socket UPI stalls |
| Pre-dequantize weights (INT8 preload) | ×1.3-1.6 | Eliminates runtime unpack port contention |
| INT8 activations (better cache footprint) | ×1.1-1.3 | Halved activation buffer size |
| 1GB hugepages (TLB miss elimination) | ×1.1-1.3 | Removes TLB thrash on 165GB model |
| **Combined (with interaction discount ~30%)** | **×12-25** | Sub-multipliers compound sub-linearly |
| **From 5.67 baseline** | | **~70-140 tok/s** |

**Decode — memory-bandwidth-bound:**

| Factor | Estimate |
|---|---|
| Activated weight bytes per token (Q4KExperts) | ~7.6 GB |
| Effective MoE random-access bandwidth | ~30-80 GB/s |
| Decode speed (bandwidth ÷ weight-read) | **~4-10 tok/s** |
| KV cache overhead (128K context) | subtract ~20-40% |
| With IQ2XXS (5.75 GB/read) | **~5-15 tok/s** |

**Summary — conservative, realistic targets on this hardware:**

| Metric | Scalar Reference (current) | Optimized VNNI Target |
|---|---|---|
| Prefill (Q4KExperts, batch=1024) | 5.67 tok/s | **70-140 tok/s** |
| Prefill (IQ2XXS, batch=1024) | — | **50-110 tok/s** |
| Decode (Q4KExperts, batch=1) | 3.70 tok/s | **5-10 tok/s** |
| Decode (IQ2XXS, batch=1) | — | **7-15 tok/s** |
| Decode (long context, 128K) | — | **3-7 tok/s** |

These are already very strong numbers for CPU-only inference on a ~300B-parameter MoE model. To go beyond this range (e.g., 500+ tok/s prefill), the hardware must be upgraded to a Sapphire Rapids / Granite Rapids platform with AMX.

### 2.7 Performance Targets: Theory vs. Engineering

| Layer of Analysis | Prefill | Decode | Key Assumption |
|---|---|---|---|
| Pure VNNI TOPS math (Section 2.2) | 500-600 tok/s | 24 tok/s | 100% VNNI occupancy, no memory stall |
| With dequant port contention | 300-400 tok/s | 16 tok/s | 50% dequant overhead |
| With MoE random-access bandwidth | 150-250 tok/s | 5-12 tok/s | 30-80 GB/s effective |
| With cache miss + barrier + TLB | **70-140 tok/s** | **4-10 tok/s** | Full system bottlenecks |
| With KV cache (long context) | — | **3-7 tok/s** | KV read dominates |

**The gap between row 1 and row 4 is where most inference optimization projects spend their time.** The architecture design (Sections 3-5) systematically addresses each degradation layer.

## 3. Scheme Design: Hybrid-Precision VNNI Graph

We will replace the dynamic memory allocation (`ds4_cpu_decode_scratch`) with a static, NUMA-aware execution graph (`ds4_xeon_graph`).

### Phase 1: Static Graph, NUMA Architecture & Expert Replication

**Static graph buffers** — `ds4_xeon_graph` pre-allocates all intermediate activation tensors for the maximum batch size. Buffers are typed by precision:

- `int8_t` buffers for RMS Norm outputs, gate/up inputs, attention inputs (~90% of activation volume)
- `int16_t` buffers for SwiGLU mid, residual accumulator (~10% of activation volume)
- `float` buffers for router logits, HC sinkhorn (tiny, FP32)

Zero `malloc`/`free` calls during the inference loop. This eliminates allocator overhead, which is especially expensive across NUMA nodes under multi-threading.

**Expert weight replication** — the server has 512GB RAM vs. the 165GB model. This surplus enables a key optimization:

```
Socket 0 RAM: full copy of all 256 experts + shared layers (~165 GB)
Socket 1 RAM: full copy of all 256 experts + shared layers (~165 GB)
Total: ~330 GB (well within 512 GB)
```

By replicating all expert weights into each socket's local memory, every token→expert lookup is guaranteed to be a **local NUMA access**. This eliminates the fundamental conflict between dynamic token routing and static NUMA partitioning — no expert-to-socket assignment, no token migration, no cross-socket UPI traffic during MoE routing.

**Shared layer placement**: Attention weights, router, shared FFN, embeddings, and output head are also replicated (or, for read-only shared weights, allocated interleaved across sockets with first-touch policy). Since these layers are accessed by all threads regardless of token-to-expert mapping, replication or careful interleaving avoids hot-spot NUMA contention.

**Thread pinning**: Threads 0-47 bound to Socket 0, threads 48-95 bound to Socket 1 via `pthread_setaffinity_np`. Each socket's threads process a subset of the batch, accessing only that socket's local copy of expert weights. Expert compute is statically partitioned: Socket 0 threads handle experts 0-127, Socket 1 threads handle experts 128-255 (the partitioning serves to distribute work, not to avoid cross-NUMA access — replication already solves that).

### Phase 2: Mixed VNNI Compute Kernels (VPDPBUSD + VPDPWSSD)

The core data path uses hybrid integer precision:

**Primary kernel — VPDPBUSD (INT8 VNNI, ~70% of MACs):**
- **Activations:** `int8_t` with per-32-element block scaling (Q8_0 style)
- **Weights:** Pre-dequantized to `uint8_t` at model load time (Section 6, Option 2)
- **Compute:** `_mm512_dpbusd_epi32` — 4 MACs/lane × 16 lanes = 64 MACs/instruction
- **Scale:** `float` rescale applied to INT32 accumulator

**Fallback kernel — VPDPWSSD (INT16 VNNI, ~30% of MACs):**
- **Activations:** `int16_t` with per-token scaling (A16 style)
- **Weights:** Pre-dequantized to `int16_t` for down projection only
- **Compute:** `_mm512_dpwssd_epi32` — 2 MACs/lane × 16 lanes = 32 MACs/instruction
- **Used only for:** FFN down projection (SwiGLU mid input)

**Control kernel — FP32 FMA:**
- Router gate (4096×256), HC sinkhorn (4-dim), embedding/LM head
- Standard AVX-512 FMA

### Phase 3: Per-Layer Precision Dispatch

```
For each layer il = 0..42:
  ┌─ Attention ─────────────────────────────────────────┐
  │ RMS Norm(input) → f32                                │
  │ quantize_a8(norm_out)      → int8 per-block          │
  │ × attn_q_a (INT8 preload)  → VPDPBUSD → f32         │
  │ × attn_kv  (INT8 preload)  → VPDPBUSD → f32         │
  │ [MLA compute in FP32: QK^T, softmax, weighted sum]   │
  │ quantize_a8(attn_out)      → int8 per-block          │
  │ × attn_output_b (INT8)     → VPDPBUSD → f32         │
  └──────────────────────────────────────────────────────┘
  ┌─ MoE FFN ───────────────────────────────────────────┐
  │ RMS Norm(residual) → f32                             │
  │ quantize_a8(norm_out)      → int8 per-block          │
  │ Router: × gate_inp (F16)   → FMA → f32 (softmax)    │
  │                                                      │
  │ Shared FFN:                                          │
  │   × ffn_gate_shexp (INT8) → VPDPBUSD → f32          │
  │   × ffn_up_shexp   (INT8) → VPDPBUSD → f32          │
  │   SwiGLU → f32                                       │
  │   × ffn_down_shexp (INT8) → VPDPBUSD → f32 (shared) │
  │                                                      │
  │ Routed Experts (6 of 256, per selected expert):      │
  │   × ffn_gate_exp (INT8 preload) → VPDPBUSD → f32    │
  │   × ffn_up_exp   (INT8 preload) → VPDPBUSD → f32    │
  │   SwiGLU(gate, up) → mid (heavy-tailed)              │
  │   quantize_a16(mid)           → int16 per-token ◄─ ONLY INT16 │
  │   × ffn_down_exp (INT16 preload) → VPDPWSSD → f32   │
  │                                                      │
  │ residual = input + shared + Σ(down_i × expert_weight) │
  └──────────────────────────────────────────────────────┘
```

### Phase 4: Thread Synchronization Model

MoE inference involves multiple synchronization points per layer. The key insight: **expert compute can proceed without a global barrier between gate and down projections**.

**Lock-free expert dispatch per layer:**

```
1. Router compute → global barrier (all threads compute router logits)
2. Expert top-k selection → per-token, no barrier (independent per token)
3. gate/up matmul → per-expert worker group, no cross-group barrier
4. SwiGLU → local to each worker group
5. quantize_a16(mid) → local
6. down matmul → per-expert worker group  
7. Shared FFN + residual add → global barrier (accumulate results)
```

Worker group model: each expert (or small contiguous expert group) is assigned to a fixed thread pool pinned to the socket where that expert's replicated weights reside. Tokens are dispatched to worker groups based on their selected experts. Since weights are replicated per socket, dispatch is always local.

This reduces barriers from ~6 per layer (fully synchronous) to ~3 per layer (router, attention, shared FFN). The expert compute phase operates with work-stealing between expert groups within each socket.

## 4. Development Roadmap

Implementation via a dedicated `DS4_BACKEND_XEON` backend, following test-driven phases.

### Step 1: Backend Isolation & VNNI Micro-Benchmarks
**Goal:** Verify VPDPBUSD (INT8) and VPDPWSSD (INT16) VNNI throughput.
1. Add `DS4_BACKEND_XEON` to `ds4_backend` enum. Create `ds4_xeon.h` / `ds4_xeon.c`.
2. Write pure AVX-512 intrinsics kernels:
   - `matmul_w4a8_vnni`: `_mm512_dpbusd_epi32` (64 MACs/inst)
   - `matmul_w4a16_vnni`: `_mm512_dpwssd_epi32` (32 MACs/inst)
3. Micro-benchmark: `tests/ds4_xeon_matmul_bench.c`.
   - Criteria: >9 TOPS (VPDPBUSD), >4.5 TOPS (VPDPWSSD) — >70% of theoretical peaks.
   - Compilation: `-march=native -mprefer-vector-width=512`.

### Step 2: Activation Quantization Kernels
1. **Per-block INT8 quant (`quantize_a8_per_block`):** Port `quantize_q8_0_activation` to AVX-512 (`_mm512_reduce_max_ps`).
2. **Per-token INT16 quant (`quantize_a16_per_token`):** Optimize `ds4_xeon_quantize_a16` with AVX-512 max reduction.
3. **Precision validation:** Roundtrip fidelity test — quantize → dequant, cosine similarity >0.999. Measure SNR difference between INT8 and INT16 on SwiGLU mid activations from a real forward pass.

### Step 3: Weight Dequantization & Pre-dequant Infrastructure
1. **Q4_K → uint8_t unpack:** AVX-512 shuffle/shift for nibble expansion. Output feeds `VPDPBUSD`.
2. **Q4_K → int16_t unpack:** Same with sign extension. Output feeds `VPDPWSSD` (down projection only).
3. **IQ2XXS unpack:** Vectorize the scalar grid lookup using AVX-512 gather + `VPDPBUSD`.
4. **Pre-dequant loader:** At model load time, expand all expert weights from Q4_K/IQ2XXS → contiguous INT8/INT16 buffers. Validate weight fidelity: bit-exact match against reference scalar dequant.
5. **1GB hugepage allocation:** Use `hugetlbfs` or `mmap(MAP_HUGETLB)` for the pre-dequantized weight buffers — eliminate TLB pressure on the 165GB+ address space.

### Step 4: Static Graph, NUMA Topology & Expert Replication
1. **Graph definition:** `ds4_xeon_graph` struct with pre-allocated buffers per precision type for max batch size.
2. **Expert weight replication:** Duplicate all expert weights into each NUMA node's local memory at load time. Total RAM: ~330 GB (2 × 165 GB) within the 512 GB budget.
3. **Thread pinning:** `pthread_setaffinity_np` — threads 0-47 to Socket 0, threads 48-95 to Socket 1.
4. **First-touch allocation:** All dynamic buffers allocated with `numa_alloc_onnode` to guarantee local memory placement.
5. **Lock-free expert dispatch:** Per-layer worker-group model with work-stealing, reducing barriers from ~6 to ~3 per layer.

### Step 5: Engine Integration & End-to-End Validation
1. **Prefill/Decode hooks:** `prefill_xeon_graph` and `decode_xeon_graph` dispatch when `DS4_BACKEND_XEON` is selected.
2. **End-to-end test:** Full prompt through the model via `tests/ds4_test.c`.
   - Criteria: identical token stream as `-cpu` reference backend for a reasoning prompt. Logits within 1e-3 relative tolerance.
   - Long-context test: 128K token prompt — validate KV cache correctness, no accuracy degradation vs. reference.

### Step 6: KV Cache & Long-Context Optimization
1. **KV cache placement:** Allocate KV cache in socket-local memory with first-touch policy. Partition by layer and head across sockets.
2. **Attention prefetching:** Prefetch next-layer KV blocks while computing current layer's FFN, using software prefetch (`_mm_prefetch`) to hide DRAM latency.
3. **FP8 KV cache:** Explore FP8 quantization for KV cache (`dsv4_fp8_kv_quantize_row_inplace_cpu` already exists in `ds4.c`) — halves KV cache memory traffic at long context.

## 5. Architectural Shift: The "Plugin" Graph Model

To minimize modifications to the core `ds4.c` engine and facilitate seamless upstream merges, the Xeon backend will be treated as a first-class, "plugin-style" Graph Backend, mirroring the architectural pattern established by Metal and CUDA.

### The "Shadow" Execution Pattern
1. **Opaque Coupling:** `ds4.c` interacts with the Xeon backend via opaque pointers (`ds4_xeon_graph *`) defined in `ds4_xeon.h`. No internal structures (`ds4_model`, `ds4_weights`, `ds4_kv_cache`) are exported to a shared private header.
2. **Minimal Hooks in `ds4.c`:**
   - **Initialization:** `ds4_session_create` conditionally initializes `ds4_xeon_graph` for `DS4_BACKEND_XEON`.
   - **Dispatch:** `ds4_session_sync` intercepts execution and dispatches to `ds4_xeon_graph_prefill` or `ds4_xeon_graph_eval_token`.
3. **Encapsulated Execution:** The entirety of the inference loop — embedding lookup, layer iterations (Attention and MoE FFN), and KV cache management — is self-contained within `ds4_xeon.c`.
4. **Context Passing:** A lightweight `ds4_xeon_model_context` struct passes essential pointers (base memory map, raw weight pointers) and dimensions at runtime.

This approach guarantees that `ds4.c` remains largely untouched (~30-50 lines of dispatch logic), preserving the reference implementation's integrity while unlocking maximum performance on Xeon hardware.

## 6. Key Engineering Decisions

### Decision 1: Expert Weight Replication (NUMA)

**Problem**: Dynamic token routing makes it impossible to guarantee token→expert affinity to a specific socket.

**Solution**: Replicate all expert weights into both sockets' local memory. Cost: ~165 GB extra RAM (total ~330 GB). Benefit: zero cross-socket UPI traffic during MoE routing. This is the single most impactful NUMA optimization and is feasible because the server's 512 GB RAM has ample headroom.

### Decision 2: Pre-dequantize Weights at Load Time

**Problem**: On-the-fly Q4_K/IQ2XXS dequantization incurs 30-50% overhead due to execution port contention with VNNI.

**Solution**: Expand expert weights to contiguous INT8/INT16 buffers at model load time. Memory cost: ~15 GB (Q4_K experts) → ~30 GB (INT8), and ~7 GB (IQ2XXS experts) → ~28 GB (INT16). Eliminates all dequant in the hot inference loop. This tradeoff (memory for throughput) is correct for CPU inference where RAM is abundant and latency is the scarce resource.

### Decision 3: 1GB Hugepages

**Problem**: A 165 GB model mapped with 4 KB pages creates ~41M TLB entries. The STLB has only 2048 entries — TLB miss rate will be catastrophic.

**Solution**: Allocate weight buffers (and static graph buffers) using 1 GB hugepages via `hugetlbfs`. This reduces TLB entries to under 200, all fitting comfortably in the STLB. Expected overhead reduction: 15-30% on memory-intensive decode.

### Decision 4: Lock-Free Expert Dispatch

**Problem**: MoE inference requires synchronization between router, expert compute, and result aggregation. With 48+ threads, global barriers at every step create significant idle time.

**Solution**: Worker-group model with per-expert thread pools. Expert compute (gate/up → SwiGLU → down) proceeds asynchronously within each worker group, with a single global barrier at result aggregation. This reduces per-layer barriers from ~6 to ~3.

## 7. Hardware Upgrade Path

The W4A8/W2A8 mixed precision architecture on Ice Lake targets **70-140 tok/s prefill** and **5-15 tok/s decode** — strong results for a CPU-only ~300B MoE model.

For higher throughput targets:

1. **Sapphire Rapids / Granite Rapids with AMX**: AMX tile-based matrix multiply (2048 INT8 ops/cycle/tile) delivers **50-100 TOPS** in practice. Combined with the same architecture (static graph, NUMA replication, pre-dequant, mixed precision), this enables **500-1500 tok/s prefill**.

2. **Speculative decoding**: For CPU-only deployments, speculative decoding (draft model + verification) is the most effective way to improve perceived decode latency. A small draft model (~1B params) running on a few cores can generate candidate tokens at high speed, with the main model verifying in parallel.

3. **Batch-prefill optimization**: For large batch prefill (1024+ tokens), group tokens by expert assignment and fuse expert computation across tokens. This improves L3 cache hit rate by ensuring each expert's weights, once loaded, process multiple tokens before eviction.

4. **FP8 KV cache**: Compress KV cache to FP8 (infrastructure already present in `ds4.c`) to reduce KV memory traffic by 50% at long context lengths.
