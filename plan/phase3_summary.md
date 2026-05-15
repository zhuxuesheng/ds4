# Phase 3 Summary: Weight Dequantization & Pre-dequant Infrastructure

Date: 2026-05-15

## Results

### Q4_K Dequant Overhead

| Operation | Throughput | Overhead |
|-----------|-----------|----------|
| Raw unpack (nibble→int16) | 20-21 GB/s | — |
| Full dequant (nibble→float→scale*q-min→clamp→int16) | 8-9 GB/s | **57-60%** |

Dequant overhead is 57-60%, well above the 30-50% assumed in the plan. This strongly validates the pre-dequant strategy: doing the expensive float arithmetic once at load time rather than on every inference pass.

### IQ2XXS Vectorization

| Implementation | Throughput | Speedup |
|---------------|-----------|---------|
| Scalar reference | 0.98 GOPS | — |
| AVX-512 vectorized | 2.50 GOPS | **2.6x** |

The vectorized kernel produces bit-exact identical results to the scalar reference (rel_err=0.00e+00).

Key techniques used:
- `_mm512_i32gather_epi64` for 8-way parallel grid table lookup
- 256-entry LUT (`iq2xxs_sign_mask_lut`) mapping sign byte → 8 negation mask bytes, avoiding per-element scalar bit extraction
- `sub(xor(w, mask), mask)` idiom for conditional two's-complement negation (no branching)
- `_mm512_dpwssd_epi32` (VPDPWSSD) for int16×int16 dot product
- 8 qs entries processed per inner loop iteration (64 weights × 64 activations)

## Changes

### `ds4_xeon.c`

- **`ds4_xeon_unpack_q4_k_to_u8`**: Scalar nibble extraction + scale factor computation. Extracts 256 uint8 values (0-15 range) and 8 scale/min pairs from a Q4_K block.
- **`ds4_xeon_unpack_q4_k_to_i16`**: Scalar nibble extraction to int16 (0-15 range, no dequant formula). For use as intermediate input to VNNI matmul.
- **`ds4_xeon_dequant_q4_k_to_i16`**: Full AVX-512 dequant: nibble extraction → float → scale*q-min → clamp → round → int16. Already had AVX-512 version; new scalar fallback.
- **`ds4_xeon_vec_dot_iq2_xxs_vnni`**: Complete rewrite from scalar (~30 lines) to AVX-512 vectorized (~80 lines). Uses gather + LUT + VPDPWSSD.
- **`iq2xxs_sign_mask_lut`**: New 256-entry uint64_t LUT for sign byte → negation mask expansion (2KB table).
- No-AVX512 stubs added for `ds4_xeon_unpack_q4_k_to_i16`.

### `ds4_xeon.h`

- Added declarations for `ds4_xeon_unpack_q4_k_to_i16`.

### `tests/ds4_xeon_math_test.c`

Extended with 4 new test suites:
- **Test 4b**: Q4_K unpack to int16 correctness (0 mismatches vs scalar)
- **Test 5**: Q4_K dequant overhead benchmark (full dequant vs raw unpack throughput)
- **Test 7**: IQ2XXS vectorized dot product correctness (bit-exact vs scalar reference, 256 blocks)
- **Test 8**: IQ2XXS throughput benchmark (scalar vs vectorized speedup)

Also added non-static `ksigns_iq2xs[128]` and `iq2xxs_grid[256]` table overrides with deterministic test patterns (necessary because ds4_xeon.c uses weak zero-filled definitions for standalone linking).

### Bugs Fixed

- **IQ2XXS sign byte extraction**: `_mm_storel_epi64` only stored first 4 uint16 elements (8 bytes), missing bytes for entries qi=4..7. Fixed to `_mm_storeu_si128` with `si16[qi*2]` indexing.
- **Static table override**: IQ2XXS correctness initially FAILED because test tables were `static`, not overriding weak definitions in ds4_xeon.c. Fixed by removing `static` from table definitions.
- **Intrinsic name**: `_mm512_xor_epi8` does not exist in AVX-512BW; corrected to `_mm512_xor_si512`.

## Pending (T3.3.1 - T3.3.3)

Pre-dequant weight loader implementation requires understanding internal `ds4_weights` struct layout from `ds4.c` (expert weight storage format, per-layer tensor indexing). Deferred to Phase 4 where engine integration provides access to these internals.

## Verification

All tests compiled and passed on remote server (dual-socket Intel Xeon Gold 5318Y, Ice Lake-SP):
```
make xeon-math-test
ALL TESTS PASSED
```
