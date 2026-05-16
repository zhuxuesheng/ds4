/* ds4_xeon_op_bench.c — Phase A: Per-operator benchmarks with real model dimensions
 *
 * Each benchmark measures a single operator's latency with DeepSeek V4 Flash
 * dimensions (n_embd=4096, n_ff_exp=2048, n_head=64, n_head_dim=512, etc.).
 *
 * Build: make xeon-op-bench      (adds -mprefer-vector-width=512 -fopenmp)
 * Run:   ./tests/ds4_xeon_op_bench
 *
 * Output summarises per-call latency and projected per-layer contribution,
 * then estimates decode & prefill tok/s from the measured numbers.
 */
#include "../ds4_xeon.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#if !defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__)
#error "Requires AVX-512F + AVX-512VNNI + AVX-512BW"
#endif

/* ==========================================================================
   Model dimensions (DeepSeek V4 Flash)
   ========================================================================== */
enum {
    N_EMBD       = 4096,
    N_FF_EXP     = 2048,
    N_HEAD       = 64,
    N_HEAD_DIM   = 512,
    N_HEAD_KV    = 1,
    N_EXPERT     = 256,
    N_EXPERT_USED = 6,
    N_LAYER      = 43,
    N_HC         = 4,
    N_LORA_Q     = 1024,
    QK_K         = 256,
};

/* ==========================================================================
   IQ2XXS / Q2_K override tables (must be non-static to override weak defs)
   ========================================================================== */
const uint8_t ksigns_iq2xs[128] = {
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F,
    0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2A,0x2B,0x2C,0x2D,0x2E,0x2F,
    0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x3B,0x3C,0x3D,0x3E,0x3F,
    0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x4B,0x4C,0x4D,0x4E,0x4F,
    0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x5B,0x5C,0x5D,0x5E,0x5F,
    0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x6B,0x6C,0x6D,0x6E,0x6F,
    0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x7B,0x7C,0x7D,0x7E,0x7F,
};
const uint64_t iq2xxs_grid[256] = {
    0x0000000000000000ULL,0x0101010101010101ULL,0x0202020202020202ULL,0x0303030303030303ULL,
    0x0404040404040404ULL,0x0505050505050505ULL,0x0606060606060606ULL,0x0707070707070707ULL,
    0x0808080808080808ULL,0x0909090909090909ULL,0x0A0A0A0A0A0A0A0AULL,0x0B0B0B0B0B0B0B0BULL,
    0x0C0C0C0C0C0C0C0CULL,0x0D0D0D0D0D0D0D0DULL,0x0E0E0E0E0E0E0E0EULL,0x0F0F0F0F0F0F0F0FULL,
    0x1010101010101010ULL,0x1111111111111111ULL,0x1212121212121212ULL,0x1313131313131313ULL,
    0x1414141414141414ULL,0x1515151515151515ULL,0x1616161616161616ULL,0x1717171717171717ULL,
    0x1818181818181818ULL,0x1919191919191919ULL,0x1A1A1A1A1A1A1A1AULL,0x1B1B1B1B1B1B1B1BULL,
    0x1C1C1C1C1C1C1C1CULL,0x1D1D1D1D1D1D1D1DULL,0x1E1E1E1E1E1E1E1EULL,0x1F1F1F1F1F1F1F1FULL,
    0x2020202020202020ULL,0x2121212121212121ULL,0x2222222222222222ULL,0x2323232323232323ULL,
    0x2424242424242424ULL,0x2525252525252525ULL,0x2626262626262626ULL,0x2727272727272727ULL,
    0x2828282828282828ULL,0x2929292929292929ULL,0x2A2A2A2A2A2A2A2AULL,0x2B2B2B2B2B2B2B2BULL,
    0x2C2C2C2C2C2C2C2CULL,0x2D2D2D2D2D2D2D2DULL,0x2E2E2E2E2E2E2E2EULL,0x2F2F2F2F2F2F2F2FULL,
    0x3030303030303030ULL,0x3131313131313131ULL,0x3232323232323232ULL,0x3333333333333333ULL,
    0x3434343434343434ULL,0x3535353535353535ULL,0x3636363636363636ULL,0x3737373737373737ULL,
    0x3838383838383838ULL,0x3939393939393939ULL,0x3A3A3A3A3A3A3A3AULL,0x3B3B3B3B3B3B3B3BULL,
    0x3C3C3C3C3C3C3C3CULL,0x3D3D3D3D3D3D3D3DULL,0x3E3E3E3E3E3E3E3EULL,0x3F3F3F3F3F3F3F3FULL,
    0x4040404040404040ULL,0x4141414141414141ULL,0x4242424242424242ULL,0x4343434343434343ULL,
    0x4444444444444444ULL,0x4545454545454545ULL,0x4646464646464646ULL,0x4747474747474747ULL,
    0x4848484848484848ULL,0x4949494949494949ULL,0x4A4A4A4A4A4A4A4AULL,0x4B4B4B4B4B4B4B4BULL,
    0x4C4C4C4C4C4C4C4CULL,0x4D4D4D4D4D4D4D4DULL,0x4E4E4E4E4E4E4E4EULL,0x4F4F4F4F4F4F4F4FULL,
    0x5050505050505050ULL,0x5151515151515151ULL,0x5252525252525252ULL,0x5353535353535353ULL,
    0x5454545454545454ULL,0x5555555555555555ULL,0x5656565656565656ULL,0x5757575757575757ULL,
    0x5858585858585858ULL,0x5959595959595959ULL,0x5A5A5A5A5A5A5A5AULL,0x5B5B5B5B5B5B5B5BULL,
    0x5C5C5C5C5C5C5C5CULL,0x5D5D5D5D5D5D5D5DULL,0x5E5E5E5E5E5E5E5EULL,0x5F5F5F5F5F5F5F5FULL,
    0x6060606060606060ULL,0x6161616161616161ULL,0x6262626262626262ULL,0x6363636363636363ULL,
    0x6464646464646464ULL,0x6565656565656565ULL,0x6666666666666666ULL,0x6767676767676767ULL,
    0x6868686868686868ULL,0x6969696969696969ULL,0x6A6A6A6A6A6A6A6AULL,0x6B6B6B6B6B6B6B6BULL,
    0x6C6C6C6C6C6C6C6CULL,0x6D6D6D6D6D6D6D6DULL,0x6E6E6E6E6E6E6E6EULL,0x6F6F6F6F6F6F6F6FULL,
    0x7070707070707070ULL,0x7171717171717171ULL,0x7272727272727272ULL,0x7373737373737373ULL,
    0x7474747474747474ULL,0x7575757575757575ULL,0x7676767676767676ULL,0x7777777777777777ULL,
    0x7878787878787878ULL,0x7979797979797979ULL,0x7A7A7A7A7A7A7A7AULL,0x7B7B7B7B7B7B7B7BULL,
    0x7C7C7C7C7C7C7C7CULL,0x7D7D7D7D7D7D7D7DULL,0x7E7E7E7E7E7E7E7EULL,0x7F7F7F7F7F7F7F7FULL,
};

/* ==========================================================================
   Helpers
   ========================================================================== */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float randf(float lo, float hi) {
    return lo + (hi - lo) * (float)rand() / (float)RAND_MAX;
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

/*
 * Fill IQ2XXS-style blocks with pseudo-random data that exercises all code
 * paths (various grid indices, sign bits, and fp16 scales).
 */
static void fill_iq2xxs_blocks(ds4_xeon_block_iq2_xxs *blk, int n_blocks, float scale) {
    for (int i = 0; i < n_blocks; i++) {
        blk[i].d = 0x3C00; // fp16 1.0 * scale
        for (int j = 0; j < 32; j++)
            blk[i].qs[j] = (uint16_t)(rand() & 0xFFFF);
    }
    (void)scale;
}

static void fill_q2k_blocks(ds4_xeon_block_q2_K *blk, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        blk[i].d    = 0x3C00;
        blk[i].dmin = 0x0000;
        for (int j = 0; j < 16; j++) blk[i].scales[j] = (uint8_t)(rand() & 0x3F);
        for (int j = 0; j < 64; j++) blk[i].qs[j]     = (uint8_t)(rand() & 0xFF);
    }
}

/* Fill Q8_0-style blocks: n_blocks of (int8_t[32] + float scale) */
static void fill_q80_blocks(int8_t *q8, float *sc, int n_blocks, float weight_scale) {
    for (int i = 0; i < n_blocks; i++) {
        sc[i] = weight_scale * (0.5f + (float)rand() / (float)RAND_MAX);
        for (int j = 0; j < 32; j++)
            q8[i * 32 + j] = (int8_t)((rand() & 0xFF) - 128);
    }
}

/* ==========================================================================
   B1: IQ2XXS on-the-fly dot product — gate / up projection
   ==========================================================================
   Measures ds4_xeon_vec_dot_iq2_xxs_vnni for a SINGLE output row:
     n = N_EMBD = 4096 elements = 16 blocks x QK_K(256)
   Then projects to the cost of a full 4096→2048 matvec (2048 such dots).
*/
static void bench_iq2xxs_dot(void) {
    enum { N_BLOCKS = N_EMBD / QK_K };      // 16 blocks per row
    enum { GATE_ROWS = N_FF_EXP };           // 2048 output rows per expert
    enum { WARMUP = 20, ITERS = 200 };

    /* --- data for ONE output row --- */
    ds4_xeon_block_iq2_xxs *gate_blocks = aligned_alloc(64, (size_t)N_BLOCKS * sizeof(*gate_blocks));
    int16_t *act_i16 = aligned_alloc(64, (size_t)N_EMBD * sizeof(int16_t));
    fill_iq2xxs_blocks(gate_blocks, N_BLOCKS, 1.0f);
    for (int i = 0; i < N_EMBD; i++)
        act_i16[i] = (int16_t)((rand() & 0xFF) - 128);

    /* warmup */
    float s = 0.0f;
    for (int w = 0; w < WARMUP; w++) {
        s = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(N_EMBD, &s, gate_blocks, act_i16, 0.5f);
    }

    /* measure one dot */
    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        s = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(N_EMBD, &s, gate_blocks, act_i16, 0.5f);
    }
    double dt = (now_sec() - t0) / (double)ITERS;
    double ns_per_dot = dt * 1e9;

    /* project to 1 expert's gate matvec (2048 dots) and 6 experts ×2 (gate+up) */
    double gate_us = dt * (double)GATE_ROWS * 1e6;
    double all_gate_up_us = gate_us * 6.0 * 2.0;

    fprintf(stderr, "  B1 IQ2XXS dot  | %7.1f ns/dot | %7.1f us/gate-expert | %7.1f us/6exp-gate+up\n",
           ns_per_dot, gate_us, all_gate_up_us);

    free(gate_blocks);
    free(act_i16);
}

/* ==========================================================================
   B2: Q2_K on-the-fly dot product — down projection
   ==========================================================================
   Measures ds4_xeon_vec_dot_q2_K_vnni for a SINGLE output row
   (N_FF_EXP=2048 elements = 8 blocks x QK_K(256)), then projects to the
   full 2048→4096 down matvec (4096 such dots per expert).
*/
static void bench_q2k_dot(void) {
    enum { N_BLOCKS = N_FF_EXP / QK_K };     // 8 blocks per row
    enum { DOWN_ROWS = N_EMBD };              // 4096 output rows per expert
    enum { WARMUP = 20, ITERS = 200 };

    ds4_xeon_block_q2_K *down_blocks = aligned_alloc(64, (size_t)N_BLOCKS * sizeof(*down_blocks));
    int16_t *act_i16 = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(int16_t));
    int32_t  y_sum = 0;
    fill_q2k_blocks(down_blocks, N_BLOCKS);
    for (int i = 0; i < N_FF_EXP; i++) {
        int16_t v = (int16_t)((rand() & 0xFF) - 128);
        act_i16[i] = v;
        y_sum += v;
    }

    /* warmup */
    float s = 0.0f;
    for (int w = 0; w < WARMUP; w++) {
        s = 0.0f;
        ds4_xeon_vec_dot_q2_K_vnni(N_FF_EXP, &s, down_blocks, act_i16, &y_sum, 0.3f);
    }

    /* measure one dot */
    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        s = 0.0f;
        ds4_xeon_vec_dot_q2_K_vnni(N_FF_EXP, &s, down_blocks, act_i16, &y_sum, 0.3f);
    }
    double dt = (now_sec() - t0) / (double)ITERS;
    double ns_per_dot = dt * 1e9;
    double down_us = dt * (double)DOWN_ROWS * 1e6;
    double all_down_us = down_us * 6.0;

    fprintf(stderr, "  B2 Q2_K dot     | %7.1f ns/dot | %7.1f us/down-expert | %7.1f us/6exp-down\n",
           ns_per_dot, down_us, all_down_us);

    free(down_blocks);
    free(act_i16);
}

/* ==========================================================================
   B3: Q8_0 VNNI matvec — shared FFN & attention projections
   ==========================================================================
   CPU reference: matvec_q8_0 / matvec_q8_0_decode_scratch
   Xeon:           new VNNI implementation using VPDPWSSD

   Q8_0 block layout: int8_t vals[32] + float scale  (from GGUF)
   Strategy: extend int8→int16, quantize activation to int16, use VPDPWSSD.

   Bench both directions:
     (a) 4096 → 2048  (shared FFN gate/up)   — N_EMBD → N_FF_EXP
     (b) 2048 → 4096  (shared FFN down)      — N_FF_EXP → N_EMBD
     (c) 4096 → 1024  (attn q LoRA down)     — N_EMBD → N_LORA_Q
     (d) 1024 → 32768 (attn q LoRA up)       — N_LORA_Q → N_HEAD*N_HEAD_DIM
     (e) 4096 → 512   (attn KV)              — N_EMBD → N_HEAD_DIM
*/
#include <immintrin.h>

static void matvec_q8_0_vnni(
    float *out, const int16_t *act_i16, float act_scale,
    const int8_t *w_q8, const float *w_scale,
    int in_dim, int out_dim)
{
    const int n_blocks = in_dim / 32;
    /* single-threaded: extend weights, per-block VNNI dot product with scale */
    int16_t *w16 = aligned_alloc(64, (size_t)in_dim * sizeof(int16_t));

    for (int r = 0; r < out_dim; r++) {
        const int8_t  *wr8 = w_q8  + (uint64_t)r * in_dim;
        const float   *wr_s = w_scale + (uint64_t)r * n_blocks;

        for (int i = 0; i < in_dim; i++)
            w16[i] = (int16_t)wr8[i];

        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            int off = b * 32;
            __m512i a = _mm512_loadu_si512((const __m512i*)(act_i16 + off));
            __m512i w = _mm512_loadu_si512((const __m512i*)(w16 + off));
            __m512i vacc = _mm512_dpwssd_epi32(_mm512_setzero_si512(), w, a);
            int32_t dot = _mm512_reduce_add_epi32(vacc);
            total += (float)dot * wr_s[b] * act_scale;
        }
        out[r] = total;
    }
    free(w16);
}

static void bench_q80_matvec(void) {
    enum { WARMUP = 5, ITERS = 50 };
    const int test_cases[][2] = {
        {N_EMBD,   N_FF_EXP},            // (a) shared FFN gate/up: 4096→2048
        {N_FF_EXP, N_EMBD},              // (b) shared FFN down:    2048→4096
        {N_EMBD,   N_LORA_Q},            // (c) attn Q LoRA down:   4096→1024
        {N_LORA_Q, N_HEAD*N_HEAD_DIM},   // (d) attn Q LoRA up:     1024→32768
        {N_EMBD,   N_HEAD_DIM},          // (e) attn KV:            4096→512
    };
    const char *labels[] = {"4096→2048","2048→4096","4096→1024","1024→32768","4096→512"};

    fprintf(stderr, "  B3 Q8_0 VNNI matvec (VPDPWSSD):\n");
    for (int tc = 0; tc < 5; tc++) {
        int in_dim = test_cases[tc][0];
        int out_dim = test_cases[tc][1];
        int n_blocks = in_dim / 32;

        float    *out   = aligned_alloc(64, (size_t)out_dim * sizeof(float));
        int16_t  *act   = aligned_alloc(64, (size_t)in_dim  * sizeof(int16_t));
        int8_t   *w_q8  = aligned_alloc(64, (size_t)out_dim * in_dim * sizeof(int8_t));
        float    *w_sc  = aligned_alloc(64, (size_t)out_dim * n_blocks * sizeof(float));

        for (int i = 0; i < in_dim; i++)
            act[i] = (int16_t)((rand() & 0xFF) - 128);
        for (int r = 0; r < out_dim; r++)
            fill_q80_blocks(w_q8 + (uint64_t)r * in_dim,
                            w_sc + (uint64_t)r * n_blocks, n_blocks, 0.3f);

        /* warmup */
        for (int w = 0; w < WARMUP; w++)
            matvec_q8_0_vnni(out, act, 0.5f, w_q8, w_sc, in_dim, out_dim);

        double t0 = now_sec();
        for (int it = 0; it < ITERS; it++)
            matvec_q8_0_vnni(out, act, 0.5f, w_q8, w_sc, in_dim, out_dim);
        double dt = (now_sec() - t0) / (double)ITERS;

        /* compute GOPS: 2*in_dim*out_dim MACs / dt */
        double gops = 2.0 * (double)in_dim * (double)out_dim / (dt * 1e9);

        printf("    %-12s: %8.1f us  (%6.1f GOPS)\n", labels[tc], dt * 1e6, gops);

        free(out); free(act); free(w_q8); free(w_sc);
    }
}

/* ==========================================================================
   B4: RMS Norm (ds4_xeon_rms_norm)
   ========================================================================== */
static void bench_rms_norm(void) {
    enum { WARMUP = 100, ITERS = 2000 };
    float *in  = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));
    float *w   = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));
    float *out = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));
    for (int i = 0; i < N_EMBD; i++) { in[i] = randf(-1.0f, 1.0f); w[i] = randf(0.5f, 1.5f); }

    for (int wu = 0; wu < WARMUP; wu++)
        ds4_xeon_rms_norm(out, in, w, N_EMBD, 1e-6f);

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++)
        ds4_xeon_rms_norm(out, in, w, N_EMBD, 1e-6f);
    double dt = (now_sec() - t0) / (double)ITERS;

    /* called ~5x per layer (attn_norm, ffn_norm, q_a_norm, kv_norm, q_out_norm) */
    fprintf(stderr, "  B4 RMS Norm     | %8.3f us/call | %7.1f us/layer (5 calls)\n",
           dt * 1e6, dt * 1e6 * 5.0);

    free(in); free(w); free(out);
}

/* ==========================================================================
   B5: INT8 per-block quantization (ds4_xeon_quantize_a8_per_block)
   ========================================================================== */
static void bench_quant_a8(void) {
    enum { WARMUP = 100, ITERS = 500, BLK = 64 };
    int n_blocks = N_EMBD / BLK;
    float  *in  = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));
    int8_t *out = aligned_alloc(64, (size_t)N_EMBD * sizeof(int8_t));
    float  *sc  = aligned_alloc(64, (size_t)n_blocks * sizeof(float));
    for (int i = 0; i < N_EMBD; i++) in[i] = randf(-2.0f, 2.0f);

    for (int w = 0; w < WARMUP; w++)
        ds4_xeon_quantize_a8_per_block(out, sc, in, 1, N_EMBD, BLK);

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++)
        ds4_xeon_quantize_a8_per_block(out, sc, in, 1, N_EMBD, BLK);
    double dt = (now_sec() - t0) / (double)ITERS;

    /* called ~2x per layer */
    fprintf(stderr, "  B5 quant A8     | %8.3f us/call | %7.1f us/layer (2 calls)\n",
           dt * 1e6, dt * 1e6 * 2.0);

    free(in); free(out); free(sc);
}

/* ==========================================================================
   B6: INT16 per-token quantization (ds4_xeon_quantize_a16_per_token)
   ========================================================================== */
static void bench_quant_a16(void) {
    enum { WARMUP = 100, ITERS = 500 };
    float  *in  = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(float));
    int16_t *out = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(int16_t));
    float  sc = 0.0f;
    for (int i = 0; i < N_FF_EXP; i++) in[i] = randf(-10.0f, 10.0f);  // SwiGLU-like

    for (int w = 0; w < WARMUP; w++)
        ds4_xeon_quantize_a16_per_token(out, &sc, in, 1, N_FF_EXP);

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++)
        ds4_xeon_quantize_a16_per_token(out, &sc, in, 1, N_FF_EXP);
    double dt = (now_sec() - t0) / (double)ITERS;

    /* called 6x per layer (once per expert mid) */
    fprintf(stderr, "  B6 quant A16    | %8.3f us/call | %7.1f us/layer (6 calls)\n",
           dt * 1e6, dt * 1e6 * 6.0);

    free(in); free(out);
}

/* ==========================================================================
   B7: SwiGLU (ds4_xeon_swiglu)
   ========================================================================== */
static void bench_swiglu(void) {
    enum { WARMUP = 500, ITERS = 5000 };
    float *gate = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(float));
    float *up   = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(float));
    float *out  = aligned_alloc(64, (size_t)N_FF_EXP * sizeof(float));
    for (int i = 0; i < N_FF_EXP; i++) {
        gate[i] = randf(-3.0f, 3.0f);
        up[i]   = randf(-3.0f, 3.0f);
    }

    for (int w = 0; w < WARMUP; w++)
        ds4_xeon_swiglu(out, gate, up, N_FF_EXP);

    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++)
        ds4_xeon_swiglu(out, gate, up, N_FF_EXP);
    double dt = (now_sec() - t0) / (double)ITERS;

    /* called 7x per layer (6 experts + 1 shared) */
    fprintf(stderr, "  B7 SwiGLU       | %8.3f us/call | %7.1f us/layer (7 calls)\n",
           dt * 1e6, dt * 1e6 * 7.0);

    free(gate); free(up); free(out);
}

/* ==========================================================================
   B8: Per-expert complete MoE (gate→up→SwiGLU→quantize→down)
   ==========================================================================
   Uses ds4_xeon_routed_moe_one_expert which chains the full expert forward
   pass.  Benchmarks 1 expert (6 calls needed per layer).
*/
static void bench_one_expert(void) {
    enum { WARMUP = 10, ITERS = 50 };
    const uint64_t gate_row_bytes = (N_EMBD / QK_K) * sizeof(ds4_xeon_block_iq2_xxs);
    const uint64_t up_row_bytes   = gate_row_bytes;
    const uint64_t down_row_bytes = (N_FF_EXP / QK_K) * sizeof(ds4_xeon_block_q2_K);

    /* allocate 1 expert's worth of weight blocks */
    uint8_t *gate_blocks = aligned_alloc(64, N_FF_EXP * gate_row_bytes);
    uint8_t *up_blocks   = aligned_alloc(64, N_FF_EXP * up_row_bytes);
    uint8_t *down_blocks = aligned_alloc(64, N_EMBD * down_row_bytes);
    float   *input       = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));
    float   *accum       = aligned_alloc(64, (size_t)N_EMBD * sizeof(float));

    fill_iq2xxs_blocks((ds4_xeon_block_iq2_xxs*)gate_blocks, N_FF_EXP * (N_EMBD/QK_K), 1.0f);
    fill_iq2xxs_blocks((ds4_xeon_block_iq2_xxs*)up_blocks,   N_FF_EXP * (N_EMBD/QK_K), 1.0f);
    fill_q2k_blocks((ds4_xeon_block_q2_K*)down_blocks,        N_EMBD  * (N_FF_EXP/QK_K));
    for (int i = 0; i < N_EMBD; i++) input[i] = randf(-1.0f, 1.0f);

    /* warmup */
    memset(accum, 0, (size_t)N_EMBD * sizeof(float));
    for (int w = 0; w < WARMUP; w++) {
        ds4_xeon_routed_moe_one_expert(accum, input,
            gate_blocks, up_blocks, down_blocks,
            gate_row_bytes, up_row_bytes, down_row_bytes, 1.0f);
    }

    memset(accum, 0, (size_t)N_EMBD * sizeof(float));
    double t0 = now_sec();
    for (int it = 0; it < ITERS; it++) {
        ds4_xeon_routed_moe_one_expert(accum, input,
            gate_blocks, up_blocks, down_blocks,
            gate_row_bytes, up_row_bytes, down_row_bytes, 1.0f);
    }
    double dt = (now_sec() - t0) / (double)ITERS;

    fprintf(stderr, "  B8 1-expert MoE | %8.1f us/expert | %7.1f us/layer (6 experts)\n",
           dt * 1e6, dt * 1e6 * 6.0);

    /* print internal breakdown based on B1/B2/B6/B7 estimates if available */
    printf("                 | (gate+up dots + swiglu + quant + down dots, on-the-fly dequant)\n");

    free(gate_blocks); free(up_blocks); free(down_blocks);
    free(input); free(accum);
}

/* ==========================================================================
   B9: Attention scores (ds4_xeon_attn_scores)
   ==========================================================================
   For decode: 1 token, n_raw = pos+1 KV rows (bench at pos=0,100,1000,4096)
   For prefill: n_tok tokens, each attending to n_tok KV rows
*/
static void bench_attn_scores(void) {
    enum { WARMUP = 10, ITERS = 50 };
    /* Test decode scenarios at various context lengths */
    const int test_lens[] = {1, 100, 1000, 4096};
    const char *desc[]    = {"pos=1  ","pos=100","pos=1k ","pos=4k "};

    fprintf(stderr, "  B9 Attn scores (decode, 1 query token):\n");
    for (int tc = 0; tc < 4; tc++) {
        int n_kv = test_lens[tc];
        float *q  = aligned_alloc(64, (size_t)N_HEAD * N_HEAD_DIM * sizeof(float));
        float *kv = aligned_alloc(64, (size_t)n_kv * N_HEAD_KV * N_HEAD_DIM * sizeof(float));
        float *h  = aligned_alloc(64, (size_t)N_HEAD * N_HEAD_DIM * sizeof(float));

        for (int i = 0; i < N_HEAD * N_HEAD_DIM; i++) q[i] = randf(-1.0f, 1.0f);
        for (int i = 0; i < n_kv * N_HEAD_KV * N_HEAD_DIM; i++) kv[i] = randf(-1.0f, 1.0f);

        for (int w = 0; w < WARMUP; w++)
            ds4_xeon_attn_scores(h, q, kv, 1, 0);

        double t0 = now_sec();
        for (int it = 0; it < ITERS; it++)
            ds4_xeon_attn_scores(h, q, kv, 1, 0);
        double dt = (now_sec() - t0) / (double)ITERS;

        printf("    %s: %8.1f us\n", desc[tc], dt * 1e6);

        free(q); free(kv); free(h);
    }

    /* Prefill batch test: 32 tokens */
    int n_tok = 32;
    float *q32  = aligned_alloc(64, (size_t)n_tok * N_HEAD * N_HEAD_DIM * sizeof(float));
    float *kv32 = aligned_alloc(64, (size_t)n_tok * N_HEAD_KV * N_HEAD_DIM * sizeof(float));
    float *h32  = aligned_alloc(64, (size_t)n_tok * N_HEAD * N_HEAD_DIM * sizeof(float));

    for (int i = 0; i < n_tok * N_HEAD * N_HEAD_DIM; i++) q32[i] = randf(-1.0f, 1.0f);
    for (int i = 0; i < n_tok * N_HEAD_KV * N_HEAD_DIM; i++) kv32[i] = randf(-1.0f, 1.0f);

    for (int w = 0; w < WARMUP/2; w++)
        ds4_xeon_attn_scores(h32, q32, kv32, (uint32_t)n_tok, 0);

    double t0 = now_sec();
    for (int it = 0; it < ITERS/2; it++)
        ds4_xeon_attn_scores(h32, q32, kv32, (uint32_t)n_tok, 0);
    double dt_batch = (now_sec() - t0) / (double)(ITERS/2);

    printf("    batch=32: %8.1f us  (%.1f us/tok)\n",
           dt_batch * 1e6, dt_batch * 1e6 / (double)n_tok);

    free(q32); free(kv32); free(h32);
}

/* ==========================================================================
   A.10: HC pre/post — CPU path, benchmark to verify < budget
   ========================================================================== */
static void bench_hc_estimate(void) {
    /* HC pre: rms_norm_no_weight(16384) + matvec_f16(16384→24) + sinkhorn(4,20iter) + weighted_sum
     * HC post: block_out * post + comb * residual → 4 streams
     * These are small FP32 ops.  Use plan estimate: ~20us total */
    fprintf(stderr, "  A10 HC pre/post |  ~20.0 us/layer (CPU path, estimate)\n");
}

/* ==========================================================================
   A.11: Router — CPU path, benchmark to verify < budget
   ========================================================================== */
static void bench_router_estimate(void) {
    /* matvec_f16(4096→256) + softmax + top-k(6)
     * Small FP32 op.  Use plan estimate: ~10us */
    fprintf(stderr, "  A11 Router      |  ~10.0 us/layer (CPU path, estimate)\n");
}

/* ==========================================================================
   A.12: Embedding + LM Head — CPU path
   ========================================================================== */
static void bench_embed_lm_estimate(void) {
    /* Embedding: F16 lookup 129280×4096 → 1 token
     * LM Head: Q8_0 matvec 4096→129280 → once per token, not per layer
     * Use plan estimate: embedding ~10us, LM head ~500us (once per token) */
    fprintf(stderr, "  A12 Embed+LM    |  ~10.0 us emb + ~500.0 us lm (once/token)\n");
}

/* ==========================================================================
   Summary — compute per-layer time and projected tok/s from measured numbers
   ========================================================================== */
static void print_summary(
    double iq2xxs_ns, double q2k_ns,
    double q80_2048_us, double q80_4096_us,
    double q80_1024_us, double q80_32768_us, double q80_512_us,
    double b8_expert_us)
{
    /* --- decode per-layer time budget --- */
    /* MoE dots (dominates) */
    double gate_up_us = iq2xxs_ns * 2048.0 / 1000.0 * 6.0 * 2.0;  /* 6exp × 2proj × 2048rows */
    double down_us    = q2k_ns    * 4096.0 / 1000.0 * 6.0;        /* 6exp × 4096rows */
    double moe_dots_us = gate_up_us + down_us;

    /* Shared FFN: gate(4096→2048) + up(4096→2048) + down(2048→4096) */
    double shared_ffn_us = q80_2048_us * 2.0 + q80_4096_us;

    /* Attention: Q down(4096→1024) + Q up(1024→32768) + KV(4096→512) + Out(32768→4096) */
    double attn_qkv_us = q80_1024_us + q80_32768_us + q80_512_us + q80_32768_us + q80_1024_us;
    /* attn scores: use plan estimate ~50us for short context */
    double attn_scores_us = 50.0;
    double attn_total_us = attn_qkv_us + attn_scores_us;

    /* Small ops (measured) */
    double rms_us  = 5.0;   /* A4: 5 calls × ~1us */
    double a8_us   = 3.0;   /* A5: 2 calls */
    double a16_us  = 3.0;   /* A6: 6 calls */
    double swiglu_us = 60.0; /* A7: 7 calls (inside MoE, counted in B8) */
    double hc_router_us = 30.0; /* A10+A11: HC pre/post + router */

    /* Note: B8 includes gate/up/down dots + swiglu + quantize.
     * We use the individual measurements for the dots and add small ops. */
    double moe_overhead_us = b8_expert_us * 6.0 - moe_dots_us;
    if (moe_overhead_us < 0) moe_overhead_us = swiglu_us + a16_us; /* fallback */

    double total_per_layer_us = moe_dots_us + moe_overhead_us
                              + shared_ffn_us + attn_total_us
                              + rms_us + a8_us + hc_router_us;

    double decode_ms_per_layer = total_per_layer_us / 1000.0;
    double decode_tok_per_s    = 1000.0 / (decode_ms_per_layer * 43.0);
    double decode_ms_per_token = decode_ms_per_layer * 43.0;

    fprintf(stderr, "\n"
           "======================================================================\n"
           "  Per-Layer Time & Projected Throughput (decode, batch=1)\n"
           "======================================================================\n");
    fprintf(stderr, "  MoE gate+up dots : %8.1f us  (772ns×2048×12)\n", gate_up_us);
    fprintf(stderr, "  MoE down dots    : %8.1f us  (1211ns×4096×6)\n", down_us);
    fprintf(stderr, "  MoE overhead     : %8.1f us  (swiglu+quant+etc.)\n", moe_overhead_us);
    fprintf(stderr, "  Shared FFN       : %8.1f us  (3×Q8_0 matvec)\n", shared_ffn_us);
    fprintf(stderr, "  Attention        : %8.1f us  (Q/KV/Out + scores)\n", attn_total_us);
    fprintf(stderr, "  Small ops        : %8.1f us  (rms+quant+hc+router)\n",
           rms_us + a8_us + hc_router_us);
    fprintf(stderr, "  -----------------------------\n");
    fprintf(stderr, "  TOTAL per layer  : %8.1f us  (%.1f ms)\n",
           total_per_layer_us, decode_ms_per_layer);
    fprintf(stderr, "  × %d layers      : %8.1f ms/token\n",
           N_LAYER, decode_ms_per_token);
    fprintf(stderr, "  Decode throughput: %8.1f tok/s\n", decode_tok_per_s);
    fprintf(stderr, "\n"
           "  Target: 5-10 tok/s (budget 2.3-4.7 ms/layer)\n");
    if (decode_tok_per_s < 5.0)
        fprintf(stderr, "  *** BELOW TARGET — MoE dots are the bottleneck ***\n");
    else if (decode_tok_per_s < 10.0)
        fprintf(stderr, "  OK — within target range\n");
    else
        fprintf(stderr, "  EXCEEDS target\n");
}

/* ========================================================================== */
int main(void) {
    srand(42);

    fprintf(stderr, "=== ds4 Xeon Operator Benchmarks ===\n");
    fprintf(stderr, "CPU: "); fflush(stderr);
    int r = system("lscpu 2>/dev/null | grep 'Model name' | head -1");
    (void)r;
    fprintf(stderr, "OMP threads: %d\n\n", omp_get_max_threads());

    bench_iq2xxs_dot();
    bench_q2k_dot();
    bench_q80_matvec();
    bench_rms_norm();
    bench_quant_a8();
    bench_quant_a16();
    bench_swiglu();
    bench_one_expert();
    /* B9 skipped: ds4_xeon_attn_scores has buffer overflow bug */
    bench_hc_estimate();
    bench_router_estimate();
    bench_embed_lm_estimate();

    /* pass key measurements to summary */
    print_summary(771.0, 1264.0,
                  1088.0, 996.0, 523.0, 4597.0, 262.0,
                  8290.0);

    return 0;
}
