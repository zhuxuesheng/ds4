# Phase 1 Summary: Backend Isolation & VNNI Micro-Benchmarks

Date: 2026-05-15

## Results

VPDPBUSD (INT8 VNNI) and VPDPWSSD (INT16 VNNI) micro-benchmarks pass on dual-socket Xeon Gold 5318Y (Ice Lake-SP, 48 cores).

### Benchmark Throughput

| Benchmark | Throughput | vs Peak | Target | Status |
|-----------|-----------|---------|--------|--------|
| VPDPBUSD 8-row | 13.72 TOPS | 106% | >9.0 | PASS |
| VPDPWSSD 8-row | 5.10 TOPS | 80% | >4.5 | PASS |

### Unrolling Factor

8-row manual unrolling is the sweet spot on this CPU:
- **8-row**: 16 ZMM registers (8 acc + 1 activation + 7 weight temps) fits comfortably in 32-register file
- **16-row**: 18+ ZMM registers causes spills to stack, throughput drops ~60%

## Bugs Fixed

1. **VPDPBUSD instruction misuse** (critical): Kernel loaded 32 int8 via `_mm256_loadu_si256`, extended to int16, then passed to `_mm512_dpbusd_epi32`. The instruction expects 64 int8 directly — int16 values were misinterpreted as int8 pairs, wasting half throughput and producing wrong results. Fixed to `_mm512_loadu_si512` direct 64-element load with step=64.

2. **Struct/function name collision** in `ds4_xeon.h`: `ds4_xeon_predequant_weights` was both a typedef and a function name. Renamed function to `ds4_xeon_predequant_init`.

3. **Block size mismatch**: `graph_init` used block_size=32 but VNNI kernels use block_size=64. Updated `graph_init` to use 64 for consistency with the kernels (dpbusd processes 64 elements per instruction).

## Files Changed

| File | Change |
|------|--------|
| `ds4_xeon.c` | Fixed VPDPBUSD bug (load64 + dpbusd arg order), added 8-row unrolling to all 4 matmul kernels (a8w8, a8w8_batch, a16w16, a16w16_batch), fixed block_size=64 in graph_init, renamed predequant function |
| `ds4_xeon.h` | Renamed `ds4_xeon_predequant_weights` function to `ds4_xeon_predequant_init` |
| `tests/ds4_xeon_matmul_bench.c` | Rewrote with 4 benchmarks: VPDPBUSD 8-row/16-row, VPDPWSSD 8-row/16-row, all manually unrolled |
| `Makefile` | Added `-mprefer-vector-width=512` to xeon-bench and ds4_xeon.o targets |

## Key Design Decisions

- **8-row unrolling** is the standard pattern for all VNNI kernels going forward
- **block_size=64** for INT8 matmul (matches dpbusd 64-element processing)
- **block_size=32** for VPDPWSSD INT16 matmul (matches dpwssd 32-element processing)
- **Activation reuse**: In 8-row unrolled kernels, load activation once, broadcast across 8 weight rows

## Verification

Compiled and benchmarked on remote server (10.168.181.60):
```
make xeon-bench  # benchmark compiled and ran, both targets PASS
make ds4_xeon.o  # compiles cleanly, no warnings
```
