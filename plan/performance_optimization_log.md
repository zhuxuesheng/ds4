# VNNI MoE Performance Optimization Log

## Baseline

| Metric | CPU Baseline | Initial VNNI | Target |
|--------|-------------|-------------|--------|
| Generation | 4.60 t/s | 0.01 t/s (broken) | 5-10 t/s |
| Prefill | 4.27 t/s | — | 70-140 t/s |

## Optimization Attempts

### 1. Kernel Bug Fixes (6 bugs) — ✅ Success

Before any performance work, 6 critical kernel bugs were found and fixed
in IQ2XXS, Q2_K, and Q8_K VNNI functions. These were correctness blockers,
not performance issues.

**Bugs fixed:**
- IQ2XXS sign mask: `sign_idx` should use `ksigns_iq2xs[sign_idx]` (permutation table)
- Q2_K scale nibble: `sc[j]` splits into `sc & 0x0F` (dot) + `sc >> 4` (bsums)
- Q8_K quantization: CPU uses `iscale = -127/max`, d can be negative
- Q2_K down interleaving: match CPU's maddubs pattern
- IQ2XXS weight format: 8-entry groups with interleaved grid/sign/scale, not `{grid, sign}` pairs
- d0 scaling position: `dw * q8_d` applied once per block (was 4× inside ib loop)

**Result:** VNNI MoE produces correct output.

---

### 2. ds4_parallel_for Down Projection — ✅ 0.01 → 3.63 t/s

Replaced per-row scalar loop (4096 sequential VNNI calls per expert) with
`down_q8k_worker` dispatched via `ds4_parallel_for` over 4096 rows.

**Result:** Generation 3.63 t/s (from unmeasurable <0.01 t/s).

---

### 3. Pure VNNI Gate/Up (replace CPU native) — ❌ Slower

Replaced CPU `matvec_iq2_xxs_expert_pair_prequant` with VNNI
`ds4_xeon_vec_dot_iq2_xxs_q8k_vnni` per row, parallelized via
`gateup_q8k_worker` with `ds4_parallel_for`.

**Result:** 3.63 → 3.30 t/s (10% slower). VNNI IQ2XXS on-the-fly
decompression is slower than CPU's hand-optimized AVX-512 pair function.

---

### 4. Hybrid MoE (CPU gate/up + VNNI down) — ✅ 3.30 → 3.66 t/s

Reverted gate/up to CPU native `matvec_iq2_xxs_expert_pair_prequant`.
Kept VNNI Q8_K down. This is the current architecture.

**Result:** 3.66 t/s. CPU gate/up is faster than VNNI for on-the-fly decode.

---

### 5. Batched Down Dispatch — ✅ 3.66 → 3.83 t/s

Replaced 6 per-expert `ds4_parallel_for` calls for down projection with
one `down_batch_worker` that processes all 6 experts in a single dispatch.
Saves 5 thread-pool wake-up cycles per layer.

**Result:** 3.83 t/s (+5%). Profile: routed from 5.7ms → 3.2ms/layer.

---

### 6. `ds4_parallel_for_min_rows(128)` — ❌ Slower

Changed batched down dispatch to use `ds4_parallel_for_min_rows(n, fn, ctx, 128)`
to reduce dispatch overhead. Only 32 work items created for 4096 rows → severe
underutilization of 48 threads.

**Result:** Much slower, >45s to generate. Reverted.

---

### 7. Batched Gate/Up via `matvec_iq2_xxs_experts_mid_prequant` — ❌ Extremely Slow

Replaced 6 per-expert `matvec_iq2_xxs_expert_pair_prequant` calls with
one batched `matvec_iq2_xxs_experts_mid_prequant` that uses 1 dispatch
for all 6 experts. Function dispatches 12288 tasks (n_expert × n_rows).

**Result:** >120s to generate a single response. The fine-grained task
granularity (12288 tasks for ~40M operations total) creates extreme
dispatch overhead. Reverted.

---

### 8. Pre-dequant Q2_K Down (3 attempts) — ❌❌❌ All Failed

**Attempt 1 — Scale nibble:** `ds4_xeon_dequant_q2k_block_to_i16` used
full `sc[j]` byte. Fixed to `sc[j] & 0x0F`. Output: garbage.

**Attempt 2 — Nibble interleaving:** Dequant function extracted nibbles
in linear order but matvec loaded activation in interleaved order.
Fixed dequant to produce interleaved weights matching activation loading.
Output: garbage.

**Attempt 3 — Scale mapping:** Dequant used linear scale-to-element mapping
(`sc[e/16]`) but on-the-fly function uses interleaved mapping
(`si = (e%4)*2 + (e/128)*8 + ((e/4)%32)/16`). Fixed to use correct mapping.
Output: garbage.

**Root cause:** Q2_K format has multiple layers of interleaving (nibble
position, scale mapping, upper/lower halves) that must all match exactly.
Each fix revealed another layer of mismatch. Pre-dequant requires perfect
replication of the on-the-fly function's access pattern.

---

### 9. IQ2XXS 8× Scale Factor — 🔍 Root Cause NOT Found

VNNI gate/up results are exactly 8× the CPU native results (per-block and
total). Per-batch self-check confirms VNNI == scalar computation. But
scalar computation also gives 8× CPU native.

`_mm512_reduce_add_epi32` verified correct (sum 1..16 = 136).

**Workaround:** `sumf *= (1.0f/8.0f)` in `ds4_xeon_vec_dot_iq2_xxs_q8k_vnni`.
Works correctly for model output.

**Candidate causes eliminated:**
- Reduce intrinsic (verified correct)
- d0 scaling position (fixed, was 4×)
- Per-pair ls scale factor (same in both CPU and VNNI)
- Activation data (same xq used for both)

---

### 10. Inject VNNI Down into `layer_routed_moe_one_prealloc` — ❌❌ Build/Link Issues

**Goal:** Keep CPU's batched gate/up (1 dispatch via `matvec_iq2_xxs_experts_mid_prequant`)
+ replace only the down projection with VNNI (1 dispatch via `down_batch_worker`).
This would give 2 dispatches total for MoE (vs current 7).

**Attempt 1:** Added VNNI code with `aligned_alloc` for per-expert Q8_K buffers.
Result: extremely slow (>60s/token). Allocation overhead dominated.

**Attempt 2:** Replaced alloc with stack buffers (`mq8[6][2048]`).
Result: build/link errors from forward declaration and duplicate definitions.
Reverted after multiple sed/edit cycles.

**Challenge:** Requires adding typedef + forward decl before the function,
injecting VNNI code inside the function body, adding worker implementation
after the function, AND removing old VNNI block from `layer_ffn_one_decode_scratch`.
Remote editing with build-test cycle makes this error-prone.

---

### 11. Attention VNNI — ❓ Not Attempted (analysis showed limited gain)

Profiled attention path: Q projection 0.55ms, KV 0.08ms, Out 0.85ms, attn_rows 0.45ms.
All projections use Q8_0 weights. CPU already uses AVX-512 `_mm512_dpbusd_epi32`
(same instruction as VNNI would use). Attention projections are already parallelized
via `ds4_parallel_for`.

**Conclusion:** VNNI would use the same AVX-512 instructions as CPU, no speedup
expected. Pre-dequantizing Q8_0 weights could help but Q8_0 is already simple
(int8 + fp16 scale, no nibble extraction needed).

---

### 12. Code Cleanup — ✅

Removed unused workers: `gateup_q8k_ctx`/`gateup_q8k_worker`, `down_q8k_ctx`/
`down_q8k_worker`, `dq_down_dequant_ctx`, `dq_down_matvec_ctx`, and pre-dequant
extern declarations. Removed old VNNI MoE block from `layer_ffn_one_decode_scratch`.

---

## Final State (commit `7874391`)

| Metric | Value | vs CPU |
|--------|-------|--------|
| Generation | 3.71 t/s | 81% |
| Prefill | 4.30 t/s | 101% |
| Correctness | ✅ | — |

**Architecture:** CPU gate/up (6 dispatches) → SwiGLU → mid Q8_K (6 scalar) →
VNNI down (1 batched dispatch). Total: 7 dispatches + 6 scalar quant per layer.

**Remaining gap to 5-10 t/s target:** 35-170%.

**Key insights:**
1. On-the-fly VNNI is slower than CPU native for IQ2XXS and comparable for Q2_K
2. Pre-dequantization is the only path to beating CPU, but format complexity
   makes it very difficult
3. Dispatch overhead (ds4_parallel_for wake-up) is significant — reducing from
   7 to 2 dispatches per MoE layer would help
4. Attention optimization has limited potential since CPU already uses AVX-512
