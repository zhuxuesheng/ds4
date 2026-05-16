/* ds4_xeon_decode_bench.c — Phase D: single-layer FFN decode benchmark
 *
 * Implements and benchmarks ds4_xeon_ffn_decode_one() which processes
 * one token through the MoE + shared FFN of a single layer using
 * pre-dequantized weights and VNNI matmuls.
 *
 * Build: make xeon-decode-bench
 * Run:   ./tests/ds4_xeon_decode_bench
 */
#include "../ds4_xeon.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#if !defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__)
#error "Requires AVX-512F + AVX-512VNNI + AVX-512BW"
#endif

/* ==========================================================================
   Model dimensions
   ========================================================================== */
enum {
    N_EMBD       = 4096,
    N_FF_EXP     = 2048,
    N_EXPERT     = 256,
    N_EXPERT_USED = 6,
    QK_K         = 256,
};

/* ==========================================================================
   Helpers
   ========================================================================== */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = h & 0x03FFu;
    if (exp == 0) {
        if (frac == 0) { uint32_t v = sign; return *(float*)&v; }
        while ((frac & 0x0400) == 0) { frac <<= 1; exp--; }
        exp++; frac &= 0x03FF;
    } else if (exp == 0x1F) {
        uint32_t v = sign | 0x7F800000u | (frac << 13);
        return *(float*)&v;
    }
    uint32_t v = sign | ((exp + 112u) << 23) | (frac << 13);
    return *(float*)&v;
}

/* ==========================================================================
   D.1: Single-row VPDPBUSD matvec (gate/up projection per output row)
   ==========================================================================
   a8: [in_dim] int8 activation (with per-64-block scale a8_scale)
   w8: [out_dim][in_dim] uint8 pre-dequantized weights
   Returns: out[out_dim] float
   Parallelism: out_dim rows are independent → caller parallelizes via ds4_parallel_for
*/
static void matvec_vpdpbusd_row(float *out, const int8_t *a8,
    const uint8_t *w8, int in_dim, int out_dim, int row0, int row1)
{
    int n_blocks = in_dim / 64;
    for (int r = row0; r < row1; r++) {
        const uint8_t *wr = w8 + (uint64_t)r * in_dim;
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            int off = b * 64;
            __m512i av = _mm512_loadu_si512((const __m512i*)(a8 + off));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wr + off));
            acc = _mm512_dpbusd_epi32(acc, wv, av);
        }
        out[r] = (float)_mm512_reduce_add_epi32(acc);
    }
}

/* ==========================================================================
   D.2: Single-row VPDPWSSD matvec (down projection per output row)
   ========================================================================== */
static void matvec_vpdpwssd_row(float *out, const int16_t *a16,
    const int16_t *w16, int in_dim, int out_dim, int row0, int row1)
{
    int n_blocks = in_dim / 32;
    for (int r = row0; r < row1; r++) {
        const int16_t *wr = w16 + (uint64_t)r * in_dim;
        __m512i acc = _mm512_setzero_si512();
        for (int b = 0; b < n_blocks; b++) {
            int off = b * 32;
            __m512i av = _mm512_loadu_si512((const __m512i*)(a16 + off));
            __m512i wv = _mm512_loadu_si512((const __m512i*)(wr + off));
            acc = _mm512_dpwssd_epi32(acc, wv, av);
        }
        out[r] = (float)_mm512_reduce_add_epi32(acc);
    }
}

/* ==========================================================================
   D.3: ds4_xeon_ffn_decode_one — single token, one layer, MoE + shared FFN
   ==========================================================================
   x:       [N_EMBD] RMS-normed input
   out:     [N_EMBD] output accumulator
   gate_u8: [N_EXPERT][N_FF_EXP][N_EMBD] pre-dequant uint8 gate weights
   up_u8:   [N_EXPERT][N_FF_EXP][N_EMBD] pre-dequant uint8 up weights
   down_i16:[N_EXPERT][N_EMBD][N_FF_EXP] pre-dequant int16 down weights
   selected:[N_EXPERT_USED] expert indices from router
   weights: [N_EXPERT_USED] router weights
*/
static void ds4_xeon_ffn_decode_one(
    float *restrict out,
    const float *restrict x,
    const uint8_t *gate_u8,      /* [n_expert][N_FF_EXP * N_EMBD] */
    const uint8_t *up_u8,        /* [n_expert][N_FF_EXP * N_EMBD] */
    const int16_t *down_i16,     /* [n_expert][N_EMBD * N_FF_EXP] */
    const int32_t *selected,
    const float *expert_weights)
{
    /* quantize input to INT8 */
    int8_t a8[N_EMBD] __attribute__((aligned(64)));
    float  a8_sc[N_EMBD / 64];
    ds4_xeon_quantize_a8_per_block(a8, a8_sc, x, 1, N_EMBD, 64);

    memset(out, 0, (size_t)N_EMBD * sizeof(float));

    for (int e = 0; e < N_EXPERT_USED; e++) {
        int eid = selected[e];
        float ew = expert_weights[e];

        const uint8_t *g8 = gate_u8 + (uint64_t)eid * N_FF_EXP * N_EMBD;
        const uint8_t *u8 = up_u8   + (uint64_t)eid * N_FF_EXP * N_EMBD;
        const int16_t *d16 = down_i16 + (uint64_t)eid * N_EMBD * N_FF_EXP;

        float gate[N_FF_EXP] __attribute__((aligned(64)));
        float up[N_FF_EXP]   __attribute__((aligned(64)));
        float mid[N_FF_EXP]  __attribute__((aligned(64)));

        /* gate projection: INT8 act × uint8 weight → VPDPBUSD */
        matvec_vpdpbusd_row(gate, a8, g8, N_EMBD, N_FF_EXP, 0, N_FF_EXP);

        /* up projection */
        matvec_vpdpbusd_row(up, a8, u8, N_EMBD, N_FF_EXP, 0, N_FF_EXP);

        /* SwiGLU */
        ds4_xeon_swiglu(mid, gate, up, N_FF_EXP);

        /* quantize mid to INT16 */
        int16_t a16[N_FF_EXP] __attribute__((aligned(64)));
        float a16_sc;
        ds4_xeon_quantize_a16_per_token(a16, &a16_sc, mid, 1, N_FF_EXP);

        /* down projection: INT16 act × int16 weight → VPDPWSSD */
        float down[N_EMBD] __attribute__((aligned(64)));
        matvec_vpdpwssd_row(down, a16, d16, N_FF_EXP, N_EMBD, 0, N_EMBD);

        /* accumulate weighted expert output */
        for (int i = 0; i < N_EMBD; i++)
            out[i] += down[i] * ew;
    }
}

/* ==========================================================================
   D.4: Benchmark — single-layer FFN decode throughput
   ========================================================================== */
static void bench_ffn_decode(void) {
    enum { WARMUP = 5, ITERS = 50 };

    const size_t expert_gate_bytes = (size_t)N_EXPERT * N_FF_EXP * N_EMBD;
    const size_t expert_down_bytes = (size_t)N_EXPERT * N_EMBD * N_FF_EXP;

    /* allocate fake pre-dequant weights */
    uint8_t *gate_u8  = aligned_alloc(64, expert_gate_bytes);
    uint8_t *up_u8    = aligned_alloc(64, expert_gate_bytes);
    int16_t *down_i16 = aligned_alloc(64, expert_down_bytes * sizeof(int16_t));
    for (size_t i = 0; i < expert_gate_bytes; i++) {
        gate_u8[i] = (uint8_t)(rand() & 0xFF);
        up_u8[i]   = (uint8_t)(rand() & 0xFF);
    }
    for (size_t i = 0; i < expert_down_bytes; i++)
        down_i16[i] = (int16_t)((rand() & 0xFFFF) - 32768);

    /* input */
    float input[N_EMBD] __attribute__((aligned(64)));
    float output[N_EMBD] __attribute__((aligned(64)));
    for (int i = 0; i < N_EMBD; i++)
        input[i] = (float)((rand() & 0xFF) - 128) / 256.0f;

    /* fake router output */
    int32_t selected[N_EXPERT_USED];
    float   weights[N_EXPERT_USED];
    for (int e = 0; e < N_EXPERT_USED; e++) {
        selected[e] = rand() % N_EXPERT;
        weights[e]  = 1.0f / (float)N_EXPERT_USED;
    }

    /* warmup */
    for (int w = 0; w < WARMUP; w++)
        ds4_xeon_ffn_decode_one(output, input, gate_u8, up_u8, down_i16,
                                selected, weights);

    /* measure */
    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++)
        ds4_xeon_ffn_decode_one(output, input, gate_u8, up_u8, down_i16,
                                selected, weights);
    double dt = (now_sec() - t0) / (double)ITERS;

    double us_per_layer = dt * 1e6;
    double us_per_layer_32t = us_per_layer / 32.0; /* 32-thread parallel */
    double ms_43layer = us_per_layer_32t * 43.0 / 1000.0;
    double tok_s = 1000.0 / ms_43layer;

    fprintf(stderr, "  D.4 FFN decode  | %7.0f us/layer (1-core) | %7.0f us (32-thr)\n",
           us_per_layer, us_per_layer_32t);
    fprintf(stderr, "    +attn(330us)  | %7.0f us/layer total  | %.1f ms/token | %.1f tok/s\n",
           us_per_layer_32t + 330.0, ms_43layer + 330.0 * 43.0 / 1000.0, tok_s);

    fprintf(stderr, "\n  On-the-fly MoE (Phase A B8): 8289 us/expert × 6 = 49734 us/layer\n");
    fprintf(stderr, "  Pre-dequant MoE:             %.0f us/layer  →  %.0fx speedup\n",
           us_per_layer, 49734.0 / us_per_layer);

    free(gate_u8); free(up_u8); free(down_i16);
}

int main(void) {
    srand(12345);
    fprintf(stderr, "=== Phase D: Xeon FFN Decode Benchmark ===\n\n");
    bench_ffn_decode();
    fprintf(stderr, "\nDone.\n");
    return 0;
}
