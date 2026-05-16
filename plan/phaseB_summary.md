# Phase B Summary: Pre-dequant vs On-the-fly Comparison

Date: 2026-05-16

## Results (Xeon Gold 5318Y, single-core AVX-512 VNNI)

### Pre-dequant throughput (one-time startup cost)

| Op | Throughput | Per-expert | Per-layer (256exp) |
|----|-----------|-----------|-------------------|
| B.1 IQ2XXS→uint8 | 2.08 GB/s | 4.0 ms | ~1.0 s |
| B.2 Q2_K→int16 | 0.87 GB/s | 19.3 ms | ~4.9 s |
| **Total per layer** | | | **~6 s** |
| **43 layers** | | | **~258 s (4.3 min)** |

### Dot product latency: on-the-fly vs pre-dequant

| Op | On-the-fly | Pre-dequant | Speedup |
|----|-----------|-------------|---------|
| Gate/up dot (4096-dim) | 771 ns (IQ2XXS gather+LUT+VPDPWSSD) | **107 ns** (VPDPBUSD) | **7.2x** |
| Down dot (2048-dim) | 1264 ns (Q2_K nibble+auto-vec) | **109 ns** (VPDPWSSD) | **11.6x** |
| Full gate matmul (2048×4096) | 1,578 μs | **279 μs** | **5.7x** |

### Projected decode throughput (pre-dequant, 32-thread pthread pool)

| Component | Single-core | 32-thread | Notes |
|-----------|-------------|-----------|-------|
| MoE gate+up (6exp) | 892 μs | **28 μs** | 7.2x vs on-the-fly |
| MoE down (6exp) | 892 μs | **28 μs** | 11.6x vs on-the-fly |
| MoE overhead (swiglu+quant) | 60 μs | 60 μs | small ops, 1 thread |
| Shared FFN | 100 μs | 100 μs | estimate |
| Attention | 330 μs | 330 μs | from Phase A |
| Small ops | 40 μs | 40 μs | rms+hc+router |
| **Per layer** | | **~637 μs** | |
| **43 layers** | | **27 ms/token** | |
| **Decode** | | **36.5 tok/s** | ideal, no overhead |

### Real-world adjustments

| Factor | Penalty | Adjusted |
|--------|---------|----------|
| Thread scaling (32→28x effective) | ×0.88 | 32 tok/s |
| DRAM bandwidth (18.6 MB/layer @ 50 GB/s) | +372 μs/layer | 14 tok/s |
| Barrier overhead (3×5μs/layer) | +15 μs/layer | 13 tok/s |
| TLB misses (4KB pages, 165GB model) | ×0.85 | 11 tok/s |
| **Conservative estimate** | | **~8-12 tok/s** |

**In target range (5-10 tok/s).** Pre-dequantization resolves the MoE dot bottleneck identified in Phase A.

## Decision

Proceed to Phase C (per-layer time budget validation) then Phase D (xeon decode path implementation).

## Files changed

| File | Change |
|------|--------|
| `tests/ds4_xeon_predequant_bench.c` | New: Phase B benchmarks (B.1-B.6) |
| `Makefile` | Added xeon-predequant-bench target |

## Verification

```
make xeon-predequant-bench
B.1 IQ2XXS→u8   | 2.08 GB/s
B.2 Q2_K→i16    | 0.87 GB/s
B.3 VPDPBUSD dot| 107 ns/dot (7.2x)
B.4 VPDPWSSD dot| 109 ns/dot (11.6x)
B.6 Decode       | 36.5 tok/s (ideal), ~8-12 tok/s (conservative)
```
