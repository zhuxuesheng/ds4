/* ds4_xeon_predequant_bench.c — Phase B: Pre-dequant vs on-the-fly comparison
 *
 * Benchmarks:
 *   B.1  IQ2XXS → uint8  pre-dequant throughput (blocks/s, GB/s)
 *   B.2  Q2_K   → int16  pre-dequant throughput
 *   B.3  VPDPBUSD matmul 1-row dot latency (pre-dequant uint8 weights)
 *   B.4  VPDPWSSD matmul 1-row dot latency (pre-dequant int16 weights)
 *   B.5  Per-expert end-to-end: on-the-fly vs pre-dequant (single core)
 *   B.6  Summary: projected per-layer decode time with pre-dequant
 *
 * Build: make xeon-predequant-bench
 * Run:   ./tests/ds4_xeon_predequant_bench
 */
#include "../ds4_xeon.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if !defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__)
#error "Requires AVX-512F + AVX-512VNNI + AVX-512BW"
#endif

/* ==========================================================================
   Model dimensions
   ========================================================================== */
enum {
    N_EMBD       = 4096,
    N_FF_EXP     = 2048,
    N_HEAD       = 64,
    N_HEAD_DIM   = 512,
    N_EXPERT_USED = 6,
    N_LAYER      = 43,
    QK_K         = 256,
};

/* IQ2XXS override tables (must be non-static to override weak defs in ds4_xeon.c) */
const uint8_t ksigns_iq2xs[128];
const uint64_t iq2xxs_grid[256];

/* ==========================================================================
   Helpers
   ========================================================================== */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_iq2xxs_blocks(ds4_xeon_block_iq2_xxs *blk, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        blk[i].d = 0x3C00; /* fp16 1.0 */
        for (int j = 0; j < 32; j++)
            blk[i].qs[j] = (uint16_t)(rand() & 0xFFFF);
    }
}

static void fill_q2k_blocks(ds4_xeon_block_q2_K *blk, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        blk[i].d    = 0x3C00;
        blk[i].dmin = 0x0000;
        for (int j = 0; j < 16; j++) blk[i].scales[j] = (uint8_t)(rand() & 0x3F);
        for (int j = 0; j < 64; j++) blk[i].qs[j]     = (uint8_t)(rand() & 0xFF);
    }
}

/* ==========================================================================
   B.1: IQ2XXS → uint8 pre-dequant throughput
   ==========================================================================
   ds4_xeon_dequant_iq2xxs_block_to_u8: 1 block (66 bytes) → 256 uint8
   One expert gate: 2048 rows × (4096/256)=16 blocks/row = 32768 blocks
*/
static double bench_iq2xxs_dequant_throughput(int n_blocks) {
    ds4_xeon_block_iq2_xxs *src = aligned_alloc(64, (size_t)n_blocks * sizeof(*src));
    uint8_t *dst = aligned_alloc(64, (size_t)n_blocks * QK_K * sizeof(uint8_t));
    fill_iq2xxs_blocks(src, n_blocks);

    /* warmup */
    for (int i = 0; i < 100; i++)
        ds4_xeon_dequant_iq2xxs_block_to_u8(dst + i * QK_K, src + i);

    double t0 = now_sec();
    for (int i = 0; i < n_blocks; i++)
        ds4_xeon_dequant_iq2xxs_block_to_u8(dst + (uint64_t)i * QK_K, src + i);
    double dt = now_sec() - t0;

    double gb = (double)n_blocks * (double)QK_K / 1e9;
    double gbps = gb / dt;
    double blocks_per_us = (double)n_blocks / (dt * 1e6);

    free(src); free(dst);
    return gbps;
}

static void bench_B1(void) {
    enum { N_BLOCKS = 32768 }; /* one expert gate = 2048 rows × 16 blocks */
    double gbps = bench_iq2xxs_dequant_throughput(N_BLOCKS);
    double us_per_expert = (double)N_BLOCKS / (gbps * 1e9 / (double)QK_K) * 1e6;
    /* More directly: */
    us_per_expert = (double)N_BLOCKS / (gbps * 1e9 / (double)QK_K / 1e6);
    /* Actually: n_blocks × 256 bytes / gbps / 1e9 * 1e6 */
    double bytes_per_expert = (double)N_BLOCKS * (double)QK_K;
    us_per_expert = bytes_per_expert / (gbps * 1e9) * 1e6;

    fprintf(stderr, "  B.1 IQ2XXS→u8   | %7.2f GB/s | %7.0f blocks | %7.1f us/expert-gate\n",
           gbps, (double)N_BLOCKS, us_per_expert);
}

/* ==========================================================================
   B.2: Q2_K → int16 pre-dequant throughput
   ========================================================================== */
static double bench_q2k_dequant_throughput(int n_blocks) {
    ds4_xeon_block_q2_K *src = aligned_alloc(64, (size_t)n_blocks * sizeof(*src));
    int16_t *dst = aligned_alloc(64, (size_t)n_blocks * QK_K * sizeof(int16_t));
    fill_q2k_blocks(src, n_blocks);

    for (int i = 0; i < 100; i++)
        ds4_xeon_dequant_q2k_block_to_i16(dst + (uint64_t)i * QK_K, src + i);

    double t0 = now_sec();
    for (int i = 0; i < n_blocks; i++)
        ds4_xeon_dequant_q2k_block_to_i16(dst + (uint64_t)i * QK_K, src + i);
    double dt = now_sec() - t0;

    double gb = (double)n_blocks * (double)QK_K * 2.0 / 1e9; /* ×2 for int16 */
    double gbps = gb / dt;

    free(src); free(dst);
    return gbps;
}

static void bench_B2(void) {
    enum { N_BLOCKS = 32768 }; /* one expert down = 4096 rows × 8 blocks */
    double gbps = bench_q2k_dequant_throughput(N_BLOCKS);
    double bytes_per_expert = (double)N_BLOCKS * (double)QK_K * 2.0;
    double us_per_expert = bytes_per_expert / (gbps * 1e9) * 1e6;

    fprintf(stderr, "  B.2 Q2_K→i16    | %7.2f GB/s | %7.0f blocks | %7.1f us/expert-down\n",
           gbps, (double)N_BLOCKS, us_per_expert);
}

/* ==========================================================================
   B.3: VPDPBUSD matmul dot latency (pre-dequant uint8 weights)
   ==========================================================================
   Measures a SINGLE row dot: INT8 activation[4096] × uint8 weight[4096] → float
   This corresponds to one output row of gate/up projection.
   With VPDPBUSD: 4096/64 = 64 instructions @ 1/cycle → ~30ns theoretical.
*/
#include <immintrin.h>

static double bench_vpdpbusd_dot(int in_dim) {
    enum { WARMUP = 500, ITERS = 5000 };
    int n_blocks = in_dim / 64;

    int8_t  *act  = aligned_alloc(64, (size_t)in_dim * sizeof(int8_t));
    uint8_t *wgt  = aligned_alloc(64, (size_t)in_dim * sizeof(uint8_t));
    for (int i = 0; i < in_dim; i++) {
        act[i] = (int8_t)((rand() & 0xFF) - 128);
        wgt[i] = (uint8_t)(rand() & 0xFF);
    }

    /* warmup */
    volatile float sum = 0;
    for (int w = 0; w < WARMUP; w++) {
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 64));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wgt + b * 64));
            acc = _mm512_dpbusd_epi32(acc, wv, av);
        }
        sum += (float)_mm512_reduce_add_epi32(acc);
    }

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 64));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wgt + b * 64));
            acc = _mm512_dpbusd_epi32(acc, wv, av);
        }
        sum += (float)_mm512_reduce_add_epi32(acc);
    }
    double dt = (now_sec() - t0) / (double)ITERS;
    double ns_per_dot = dt * 1e9;

    fprintf(stderr, "  B.3 VPDPBUSD dot | %7.1f ns/dot (%d-dim) | vs B1 on-the-fly: 771 ns → %.0fx\n",
           ns_per_dot, in_dim, 771.0 / ns_per_dot);

    free(act); free(wgt);
    return ns_per_dot;
}

/* ==========================================================================
   B.4: VPDPWSSD matmul dot latency (pre-dequant int16 weights)
   ========================================================================== */
static double bench_vpdpwssd_dot(int in_dim) {
    enum { WARMUP = 500, ITERS = 5000 };
    int n_blocks = in_dim / 32;

    int16_t *act  = aligned_alloc(64, (size_t)in_dim * sizeof(int16_t));
    int16_t *wgt  = aligned_alloc(64, (size_t)in_dim * sizeof(int16_t));
    for (int i = 0; i < in_dim; i++) {
        act[i] = (int16_t)((rand() & 0xFFFF) - 32768);
        wgt[i] = (int16_t)((rand() & 0xFFFF) - 32768);
    }

    volatile float sum = 0;
    for (int w = 0; w < WARMUP; w++) {
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 32));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wgt + b * 32));
            acc = _mm512_dpwssd_epi32(acc, wv, av);
        }
        sum += (float)_mm512_reduce_add_epi32(acc);
    }

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 32));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wgt + b * 32));
            acc = _mm512_dpwssd_epi32(acc, wv, av);
        }
        sum += (float)_mm512_reduce_add_epi32(acc);
    }
    double dt = (now_sec() - t0) / (double)ITERS;
    double ns_per_dot = dt * 1e9;

    fprintf(stderr, "  B.4 VPDPWSSD dot | %7.1f ns/dot (%d-dim) | vs B2 on-the-fly: 1264 ns → %.0fx\n",
           ns_per_dot, in_dim, 1264.0 / ns_per_dot);

    free(act); free(wgt);
    return ns_per_dot;
}

/* ==========================================================================
   B.5: Full-matmul latency (single-core, pre-dequant weights)
   ==========================================================================
   Simpler: measure full gate matmul (2048 rows × 4096 cols) with raw
   VPDPBUSD loop.  Combine with B.3/B.4 dot data for expert projection.
*/
static void bench_B5_full_matmul(void) {
    enum { WARMUP = 3, ITERS = 10 };
    const int gate_rows = N_FF_EXP;   /* 2048 */
    const int in_dim    = N_EMBD;     /* 4096 */

    int8_t  *act   = aligned_alloc(64, (size_t)in_dim * sizeof(int8_t));
    uint8_t *wgt   = aligned_alloc(64, (size_t)gate_rows * in_dim);
    float   *out   = aligned_alloc(64, (size_t)gate_rows * sizeof(float));
    for (int i = 0; i < in_dim; i++) act[i] = (int8_t)((rand() & 0xFF) - 128);
    for (int i = 0; i < gate_rows * in_dim; i++) wgt[i] = (uint8_t)(rand() & 0xFF);

    /* warmup */
    for (int w = 0; w < WARMUP; w++) {
        for (int r = 0; r < gate_rows; r++) {
            const uint8_t *wr = wgt + (uint64_t)r * in_dim;
            __m512i acc = _mm512_setzero_si512();
            for (int b = 0; b < in_dim / 64; b++) {
                __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 64));
                __m512i wv = _mm512_loadu_si512((const __m512i*)(wr + b * 64));
                acc = _mm512_dpbusd_epi32(acc, wv, av);
            }
            out[r] = (float)_mm512_reduce_add_epi32(acc);
        }
    }

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        for (int r = 0; r < gate_rows; r++) {
            const uint8_t *wr = wgt + (uint64_t)r * in_dim;
            __m512i acc = _mm512_setzero_si512();
            for (int b = 0; b < in_dim / 64; b++) {
                __m512i av = _mm512_loadu_si512((const __m512i*)(act + b * 64));
                __m512i wv = _mm512_loadu_si512((const __m512i*)(wr + b * 64));
                acc = _mm512_dpbusd_epi32(acc, wv, av);
            }
            out[r] = (float)_mm512_reduce_add_epi32(acc);
        }
    }
    double dt = (now_sec() - t0) / (double)ITERS;

    double ns_per_row = dt * 1e9 / (double)gate_rows;
    fprintf(stderr, "  B.5 Full matmul  | %7.1f us (2048x4096) | %7.1f ns/row | vs dot: %.0f ns\n",
           dt * 1e6, ns_per_row, dt * 1e9 / (double)gate_rows);

    free(act); free(wgt); free(out);
}

/* ==========================================================================
   B.6: Summary — projected per-layer time with pre-dequant
   ========================================================================== */
static void bench_B6_summary(double ns_vpdpbusd, double ns_vpdpwssd) {
    /* Single-core dot latencies → per-expert per-layer */
    double gate_us = ns_vpdpbusd * (double)N_FF_EXP / 1000.0;  /* 2048 rows */
    double up_us   = gate_us;
    double down_us = ns_vpdpwssd * (double)N_EMBD / 1000.0;   /* 4096 rows */
    double swiglu_us = 9.0;   /* from B7 */
    double quant_us  = 0.5;   /* from B6 */

    /* Single expert, single core */
    double expert_us = gate_us + up_us + swiglu_us + quant_us + down_us;

    /* With 32-thread pool: rows-per-thread ≈ total_rows/32, almost linear speedup */
    double expert32_us = expert_us / 32.0;

    /* 6 experts, parallelizable (but bounded by 32 threads total) */
    double moe32_us = expert32_us * 6.0;

    /* Shared FFN + Attention (from Phase A, 32-thread estimate) */
    double shared_ffn_us = 100.0;
    double attn_us = 330.0;
    double small_us = 40.0;

    double layer_us = moe32_us + shared_ffn_us + attn_us + small_us;
    double decode_ms_token = layer_us * (double)N_LAYER / 1000.0;
    double decode_tok_s = 1000.0 / decode_ms_token;

    fprintf(stderr, "\n"
           "======================================================================\n"
           "  B.6 Projected Decode Throughput (pre-dequant, 32-thread)\n"
           "======================================================================\n");
    fprintf(stderr, "  Per-dot latency   : VPDPBUSD %.0f ns, VPDPWSSD %.0f ns\n",
           ns_vpdpbusd, ns_vpdpwssd);
    fprintf(stderr, "  1 expert (1-core) : %.0f us\n", expert_us);
    fprintf(stderr, "  6 experts (32-thr): %.0f us\n", moe32_us);
    fprintf(stderr, "  +Shared FFN       : %.0f us\n", shared_ffn_us);
    fprintf(stderr, "  +Attention        : %.0f us\n", attn_us);
    fprintf(stderr, "  +Small ops        : %.0f us\n", small_us);
    fprintf(stderr, "  Per layer         : %.0f us\n", layer_us);
    fprintf(stderr, "  × %d layers       : %.0f ms/token\n", N_LAYER, decode_ms_token);
    fprintf(stderr, "  Decode            : %.1f tok/s\n", decode_tok_s);
    fprintf(stderr, "\n  Target: 5-10 tok/s\n");
    if (decode_tok_s >= 5.0)
        fprintf(stderr, "  *** IN TARGET RANGE ***\n");
    else
        fprintf(stderr, "  *** BELOW TARGET — memory bandwidth is the next bottleneck ***\n");
}

/* ========================================================================== */
int main(void) {
    srand(12345);
    fprintf(stderr, "=== Phase B: Pre-dequant vs On-the-fly Benchmarks ===\n\n");

    bench_B1();
    bench_B2();
    double ns_busd = bench_vpdpbusd_dot(N_EMBD);   /* 4096-dim dot */
    double ns_wssd = bench_vpdpwssd_dot(N_FF_EXP); /* 2048-dim dot */
    bench_B5_full_matmul();
    bench_B6_summary(ns_busd, ns_wssd);

    fprintf(stderr, "\nDone.\n");
    return 0;
}
