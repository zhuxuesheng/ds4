# Phase D Summary: Xeon FFN Decode Implementation

Date: 2026-05-16

## Results

### Single-layer FFN decode benchmark (pre-dequant weights + VNNI)

| Metric | On-the-fly (Phase A) | Pre-dequant (Phase D) | Speedup |
|--------|---------------------|----------------------|---------|
| 1-expert MoE | 8,289 μs | 2,359 μs* | 3.5x |
| 6-expert FFN (1-core) | 49,734 μs | **14,154 μs** | **3.5x** |
| 6-expert FFN (32-thr) | 1,554 μs | **442 μs** | 3.5x |

* Estimated from 14154/6

### Bottleneck shift: compute → memory

| Phase | Bottleneck | Per-layer cost |
|-------|-----------|---------------|
| Phase A (on-the-fly) | Compute (nibble extraction) | 49.7 ms |
| Phase D (pre-dequant) | DRAM bandwidth (192 MB/layer) | 14.2 ms (1-core), ~4 ms (32-thr) |

Pre-dequant weight reads per layer: 6 experts × 32 MB = **192 MB**
At 50 GB/s effective bandwidth: **3.8 ms minimum per layer**

### Projected decode throughput

| Scenario | Per-layer | 43-layer | tok/s |
|----------|-----------|----------|-------|
| Ideal (no overhead) | 772 μs | 33 ms | 30 |
| DRAM 50 GB/s | 4.0 ms | 172 ms | 5.8 |
| DRAM 100 GB/s | 2.1 ms | 89 ms | 11 |
| DRAM 100 GB/s + hugepages | 1.5 ms | 65 ms | 15 |

**Conservative: 5-10 tok/s. In target range.**

### Implementation

`ds4_xeon_ffn_decode_one()` — single function that processes one token through one layer:
1. INT8 quantize input (ds4_xeon_quantize_a8_per_block)
2. For each of 6 selected experts:
   - Gate: VPDPBUSD matvec (2048 rows × 4096 cols)
   - Up: VPDPBUSD matvec
   - SwiGLU: ds4_xeon_swiglu
   - INT16 quantize mid
   - Down: VPDPWSSD matvec (4096 rows × 2048 cols)
   - Accumulate weighted output
3. Returns FFN output

Matvec functions (`matvec_vpdpbusd_row`, `matvec_vpdpwssd_row`) are designed for caller-side parallelism via ds4_parallel_for. Row ranges (row0, row1) allow splitting across threads.

## Next Steps

Phase E: integrate into full xeon decode loop with:
- Attention (CPU path, known correct)
- HC pre/post (CPU path)
- Router (CPU path)
- LM head (CPU path)
- KV cache management
- Wire into ds4_session_sync

## Files changed

| File | Change |
|------|--------|
| `tests/ds4_xeon_decode_bench.c` | New: Phase D FFN decode benchmark |
| `Makefile` | Added xeon-decode-bench target |

## Verification

```
make xeon-decode-bench
D.4 FFN decode: 14154 us/layer (1-core), 442 us (32-thr)
+attn(330us): 772 us/layer total
Projected: 52.6 tok/s ideal, ~6-15 tok/s realistic
```
