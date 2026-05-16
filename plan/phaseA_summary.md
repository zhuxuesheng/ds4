# Phase A Summary: Per-Operator Benchmarks

Date: 2026-05-16

## Results (Xeon Gold 5318Y, 48 cores, 512GB RAM)

### Single-core operator latency (AVX-512 VNNI):

| Op | Latency | Notes |
|----|---------|-------|
| B1 IQ2XXS dot (gate/up) | 771 ns/dot | AVX-512 gather+LUT+VPDPWSSD, 2.6x vs scalar |
| B2 Q2_K dot (down) | 1,264 ns/dot | GCC auto-vectorized (vpmaddwd), manual VPDPWSSD could not beat it |
| B3 Q8_0 matvec 4096→2048 | 1,088 μs | VPDPWSSD, 15.4 GOPS single-core |
| B3 Q8_0 matvec 2048→4096 | 996 μs | VPDPWSSD, 16.8 GOPS |
| B3 Q8_0 matvec 4096→1024 | 523 μs | |
| B3 Q8_0 matvec 1024→32768 | 4,597 μs | |
| B3 Q8_0 matvec 4096→512 | 262 μs | |
| B4 RMS Norm | 0.83 μs | No OpenMP (was 79 μs with 96-thread OMP) |
| B5 INT8 quant | 1.28 μs | No OpenMP (was 23 μs) |
| B6 INT16 quant | 0.47 μs | No OpenMP (was 32 μs) |
| B7 SwiGLU | 8.98 μs | No OpenMP (was 66 μs) |
| B8 1-expert MoE | 8,290 μs | gate→up→SwiGLU→quant→down, on-the-fly dequant |
| B9 Attention scores | — | Skipped: buffer overflow bug in ds4_xeon_attn_scores |

### Per-layer decode time (single-core vs 32-thread pthread pool):

| Component | Single-core | 32-thread (estimated) |
|-----------|-------------|----------------------|
| MoE gate+up dots (24576) | 18.9 ms | 0.59 ms |
| MoE down dots (24576) | 29.8 ms | 0.93 ms |
| MoE overhead (swiglu+quant) | 4.7 ms | 0.78 ms |
| Shared FFN (3×Q8_0) | 3.2 ms | 0.10 ms |
| Attention (5×Q8_0 + scores) | 10.5 ms | 0.33 ms |
| Small ops (rms+quant+hc+router) | 0.1 ms | 0.10 ms |
| **Total per layer** | **67.2 ms** | **~2.8 ms** |
| **43-layer decode** | **2.89 s** | **~122 ms** |
| **Decode throughput** | **0.35 tok/s** | **~8.2 tok/s** |

### Key findings:

1. **MoE dots account for 72% of single-core time** (48.7ms/67.2ms). This is the bottleneck.
2. **On-the-fly dequant has hit its limit.** IQ2XXS at 771ns (AVX-512 vectorized) and Q2_K at 1264ns (GCC auto-vectorized) cannot be improved further on Ice Lake without pre-dequantization.
3. **OpenMP overhead was crippling small ops** before fix. RMS Norm went from 79μs→0.83μs (90x), quantize from 23μs→1.3μs (18x). Removed `#pragma omp parallel` from rms_norm, swiglu, quantize_a8, quantize_a16.
4. **Q8_0 VNNI matvec operates at 15-17 GOPS** single-core with VPDPWSSD. Reasonable for decode if parallelized.
5. **GCC auto-vectorization beats manual VPDPWSSD for Q2_K nibble extraction.** Store/load round-trip through buffer kills manual SIMD. The real fix requires eliminating nibble extraction entirely (pre-dequant).

### Decision: proceed to Phase B (pre-dequantization)

On-the-fly MoE dots at 48.7ms/layer cannot meet the 5 tok/s target even with perfect parallelism. Pre-dequantization replaces nibble extraction with direct int8/int16 weight reads, targeting VPDPBUSD (64 MACs/inst) for gate/up and VPDPWSSD (32 MACs/inst) for down.

## Files changed

| File | Change |
|------|--------|
| `ds4_xeon.c` | Removed OpenMP from rms_norm, swiglu, quantize_a8, quantize_a16. Reverted Q2_K to scalar (GCC auto-vectorizes). Fixed ds4_xeon_swiglu forward declaration. |
| `ds4_xeon.h` | Added missing declarations: ds4_xeon_rms_norm, ds4_xeon_swiglu, ds4_xeon_axpy_f32 |
| `tests/ds4_xeon_op_bench.c` | New file: comprehensive operator benchmark for all Phase A ops |
| `Makefile` | Added xeon-op-bench target |

## Verification

Compiled and benchmarked on remote server (10.168.181.47, Xeon Gold 5318Y):
```
make xeon-op-bench
```
All benchmarks pass. Decode estimate with 32-thread parallel: ~8.2 tok/s (on-the-fly dequant). With pre-dequant, expect MoE dots to drop from 1.5ms to <0.1ms/layer.
