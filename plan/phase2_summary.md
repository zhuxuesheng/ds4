# Phase 2 Summary: Activation Quantization Kernels

Date: 2026-05-15

## Results

### Roundtrip Fidelity (FP32 → quantize → dequant → FP32)

| Distribution | INT8 cos-sim | INT8 SNR | INT16 cos-sim | INT16 SNR |
|-------------|-------------|----------|---------------|-----------|
| Gaussian | 0.99998 | 44.5 dB | 1.00000 | 89.8 dB |
| Uniform | 0.99999 | 48.3 dB | 1.00000 | 96.3 dB |
| Heavy-tailed | 0.99993 | 38.4 dB | 1.00000 | 81.7 dB |

All AVX-512 paths bit-exact match scalar reference (0 mismatches in quantized values and scales).

### SwiGLU Mid INT8 vs INT16

| Quantization | SNR | Advantage |
|-------------|-----|-----------|
| INT8 per-block | 38.38 dB | — |
| INT16 per-token | 81.42 dB | **+43.04 dB** |

Confirms plan assumption: SwiGLU mid is heavy-tailed and needs INT16 (>10 dB gap observed as 43 dB).

### Quantization Throughput

| Kernel | Batch | Throughput | Target |
|--------|-------|-----------|--------|
| INT8 per-block | 8 tok | 8.6 GB/s | — |
| INT8 per-block | 256 tok | **190.4 GB/s** | >100 GB/s |
| INT16 per-token | 8 tok | 8.4 GB/s | — |
| INT16 per-token | 256 tok | **196.9 GB/s** | >100 GB/s |

Small batch throttled by OpenMP overhead on 48 threads; large batch saturates at ~73% of theoretical memory bandwidth (260 GB/s).

## Changes

### `ds4_xeon.c`

- **`ds4_xeon_quantize_a8_per_block`**: Replaced scalar max-find with `_mm512_reduce_max_ps`, replaced scalar quantize with AVX-512 round+pack (`_mm512_roundscale_ps` → `_mm512_cvtps_epi32` → `_mm512_cvtsepi32_epi16` → `_mm256_cvtsepi16_epi8`). Handles arbitrary block_size (multiples of 16). 16x throughput improvement over scalar.
- **`ds4_xeon_quantize_a8_per_block_avx512`**: Now delegates to the main function (alias, kept for compatibility).
- **`ds4_xeon_quantize_a16_per_token`**: Added AVX-512 max-find via `_mm512_reduce_max_ps`, vectorized quantize with `_mm512_min_ps`/`_mm512_max_ps` clamping and `_mm256_storeu_si256` packing. ~20x throughput improvement.
- Fixed IQ2XXS table weak definitions for standalone linking.
- Fixed `tail` unused variable warning.

### `tests/ds4_xeon_math_test.c`

Full rewrite with 4 test suites:
1. Roundtrip fidelity (3 distributions × 2 quantization types)
2. SwiGLU mid INT8 vs INT16 SNR comparison
3. AVX-512 bit-exact match vs scalar reference
4. Throughput benchmarks (small/large batch for both INT8 and INT16)

### Bugs Fixed

- **Heap corruption in int16 quantization**: `_mm512_storeu_si512` writes 32 int16 (64 bytes) but pointer advanced by only 16 int16 (32 bytes), writing past the buffer. Fixed to `_mm256_storeu_si256` (16 int16 = 256 bits).
- **IQ2XXS linker error**: `extern` declarations caused undefined references when linking without `ds4.c`. Fixed with `__attribute__((weak))` definitions.

## Verification

Compiled and tested on remote server (10.168.181.60):
```
make xeon-math-test
ALL TESTS PASSED
```
