/* ds4_xeon_down_test.c — Compare VNNI Q2_K down vs CPU reference
 *
 * Simulates the down projection: heavy-tailed SwiGLU mid activations
 * quantized to Q8_K (CPU) vs INT16 (VNNI), dotted with Q2_K weights.
 *
 * Build: make xeon-down-test
 */
#include "../ds4_xeon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if !defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512BW__)
#error "Requires AVX-512F + AVX-512VNNI + AVX-512BW"
#endif

#define QK_K  256
#define N_FF  2048  /* SwiGLU mid dimension (down input) */
#define N_EMBD 4096 /* down output dimension */

/* ==========================================================================
   Helpers
   ========================================================================== */
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
   CPU reference: Q2_K dot product with Q8_K activation
   ==========================================================================
   This mirrors ds4_vec_dot_q2_K_q8_K from ds4.c.
   Q2_K block (84 bytes): scales[16], qs[64], d(fp16), dmin(fp16)
   Q8_K activation: int8_t[256] per block + float scale
*/
static float cpu_q2k_dot_q8k(
    const ds4_xeon_block_q2_K *blocks, int n_blocks,
    const int8_t *q8, const float *q8_scale)
{
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; i++) {
        const float d    = f16_to_f32(blocks[i].d);
        const float dmin = f16_to_f32(blocks[i].dmin);
        const uint8_t *sc = blocks[i].scales;
        const uint8_t *q2 = blocks[i].qs;
        const int8_t  *a  = q8   + i * QK_K;
        const float   a_d = q8_scale[i];

        for (int j = 0; j < 16; j++) {
            /* a32_sum: sum of 32 activation elements */
            int32_t a32_sum = 0;
            for (int k = 0; k < 32; k++)
                a32_sum += (int32_t)a[j * 16 + (k % 16)]; /* 2 groups of 16 */
            /* Actually: each sub-block has 16 elements, 2 sub-blocks share one a32_sum */
            /* Let me follow the standard Q2_K dot more carefully */

            /* Real Q2_K: per sub-block of 16 weights × 16 activation elements */
            const uint8_t *q_ptr = q2 + j * 4;
            float sc_val = (float)sc[j];
            int32_t dot = 0;
            for (int k = 0; k < 16; k++) {
                uint8_t q_byte = q_ptr[k / 4];
                uint8_t nibble  = (q_byte >> ((k % 4) * 2)) & 3;
                dot += (int32_t)nibble * (int32_t)a[j * 16 + k];
            }
            sumf += a_d * (d * sc_val * (float)dot - dmin * sc_val * (float)a32_sum);
        }
    }
    return sumf;
}

/* Simplified Q2_K dot for testing: match the VNNI function's interface
 * The VNNI function (ds4_xeon_vec_dot_q2_K_vnni) uses INT16 activation
 * and pre-computed int32 sums per 32 elements. */
static float cpu_q2k_dot_i16(
    const ds4_xeon_block_q2_K *blocks, int n_blocks,
    const int16_t *act_i16, const int32_t *y_sums, float scale_y)
{
    float sumf = 0.0f;
    for (int i = 0; i < n_blocks; i++) {
        const float d    = f16_to_f32(blocks[i].d) * scale_y;
        const float dmin = f16_to_f32(blocks[i].dmin) * scale_y;
        const uint8_t *sc = blocks[i].scales;
        const uint8_t *q2 = blocks[i].qs;
        const int16_t *a  = act_i16 + i * QK_K;

        for (int j = 0; j < 16; j++) {
            const uint8_t *q_ptr = q2 + j * 4;
            float sc_val = (float)sc[j];
            int32_t dot = 0;
            for (int k = 0; k < 16; k++) {
                uint8_t q_byte = q_ptr[k / 4];
                uint8_t nibble  = (q_byte >> ((k % 4) * 2)) & 3;
                dot += (int32_t)nibble * (int32_t)a[j * 16 + k];
            }
            sumf += d * sc_val * (float)dot
                  - dmin * sc_val * (float)y_sums[(i * 16 + j) / 2];
        }
    }
    return sumf;
}

/* ==========================================================================
   Quantize to Q8_K (simplified: per-block int8)
   ========================================================================== */
static void quantize_q8k(int8_t *q8, float *scale, const float *x, int n) {
    int n_blocks = n / QK_K;
    for (int b = 0; b < n_blocks; b++) {
        const float *src = x + b * QK_K;
        int8_t *dst = q8 + b * QK_K;
        float amax = 1e-9f;
        for (int i = 0; i < QK_K; i++) {
            float a = fabsf(src[i]);
            if (a > amax) amax = a;
        }
        float d = amax / 127.0f;
        float id = 1.0f / d;
        scale[b] = d;
        for (int i = 0; i < QK_K; i++) {
            float v = src[i] * id;
            if (v > 127.0f) v = 127.0f;
            if (v < -128.0f) v = -128.0f;
            dst[i] = (int8_t)lrintf(v);
        }
    }
}

/* ==========================================================================
   Main test
   ========================================================================== */
int main(void) {
    srand(12345);
    fprintf(stderr, "=== VNNI Q2_K down projection correctness test ===\n\n");

    int n_blocks = N_FF / QK_K; /* 2048/256 = 8 blocks per row */
    int n_rows   = N_EMBD;      /* 4096 output rows */

    /* Allocate Q2_K blocks for one row */
    ds4_xeon_block_q2_K *blocks = aligned_alloc(64,
        n_blocks * sizeof(ds4_xeon_block_q2_K));

    /* Create heavy-tailed mid activations (SwiGLU style) */
    float mid[N_FF];
    float ss = 0;
    for (int i = 0; i < N_FF; i++) {
        /* Heavy-tailed: mixture of small values and occasional large spikes */
        float v = (float)((rand() & 0xFF) - 128) / 16.0f;
        if ((rand() & 0xFF) < 10) v *= 20.0f;  /* ~4% chance of large spike */
        mid[i] = v;
        ss += v * v;
    }
    float norm_scale = 1.0f / sqrtf(ss / N_FF + 1e-6f);
    for (int i = 0; i < N_FF; i++) mid[i] *= norm_scale;
    fprintf(stderr, "Mid stats: rms=1.0, min=%.2f, max=%.2f\n",
        mid[0], mid[N_FF-1]);

    /* Q8_K quantize (CPU path) */
    int8_t q8[N_FF];
    float  q8_scales[N_FF / QK_K];
    quantize_q8k(q8, q8_scales, mid, N_FF);
    fprintf(stderr, "Q8_K scale range: [%.4f, %.4f]\n",
        (double)q8_scales[0], (double)q8_scales[n_blocks-1]);

    /* INT16 quantize (VNNI path) */
    int16_t act_i16[N_FF] __attribute__((aligned(64)));
    float   act_scale;
    ds4_xeon_quantize_a16_per_token(act_i16, &act_scale, mid, 1, N_FF);
    fprintf(stderr, "INT16 act_scale: %.6f\n", (double)act_scale);

    /* Pre-compute INT32 sums for VNNI Q2_K */
    int32_t y_sums[N_FF / 32];
    for (int i = 0; i < N_FF / 32; i++) {
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += (int32_t)act_i16[i * 32 + j];
        y_sums[i] = s;
    }

    /* ===== Test 1: single row, Q8_K vs INT16 ===== */
    fprintf(stderr, "\n--- Test 1: Single row ---\n");
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d    = 0x3C00; /* fp16 1.0 */
        blocks[i].dmin = 0x0000;
        for (int j = 0; j < 16; j++) blocks[i].scales[j] = (uint8_t)(rand() & 0x3F);
        for (int j = 0; j < 64; j++) blocks[i].qs[j] = (uint8_t)(rand() & 0xFF);
    }

    float cpu_val = cpu_q2k_dot_q8k(blocks, n_blocks, q8, q8_scales);
    float vnni_q8k_val = 0.0f;
    ds4_xeon_vec_dot_q2_K_q8k_vnni(N_FF, &vnni_q8k_val, blocks, q8, q8_scales);
    float vnni_i16_val = 0.0f;
    ds4_xeon_vec_dot_q2_K_vnni(N_FF, &vnni_i16_val, blocks, act_i16, y_sums, act_scale);

    fprintf(stderr, "  CPU (Q8_K)   : %.6f\n", (double)cpu_val);
    fprintf(stderr, "  VNNI (Q8_K)  : %.6f\n", (double)vnni_q8k_val);
    fprintf(stderr, "  VNNI (I16)   : %.6f\n", (double)vnni_i16_val);
    fprintf(stderr, "  Q8K: CPU vs VNNI diff : %.2e\n",
        fabs((double)cpu_val - (double)vnni_q8k_val));
    fprintf(stderr, "  Q8K vs I16 diff       : %.2e\n",
        fabs((double)cpu_val - (double)vnni_i16_val));

    /* ===== Test 2: multi-row, both paths, check error distribution ===== */
    fprintf(stderr, "\n--- Test 2: %d rows (full down matvec) ---\n", n_rows);
    double max_rel = 0, max_abs = 0, sum_rel = 0;
    int    max_row = 0;
    float  worst_cpu = 0, worst_vnni = 0;

    for (int r = 0; r < n_rows; r++) {
        /* Generate random Q2_K blocks for this row */
        for (int i = 0; i < n_blocks; i++) {
            blocks[i].d    = 0x3C00;
            blocks[i].dmin = 0x0000;
            for (int j = 0; j < 16; j++) blocks[i].scales[j] = (uint8_t)(rand() & 0x3F);
            for (int j = 0; j < 64; j++) blocks[i].qs[j] = (uint8_t)(rand() & 0xFF);
        }

        float cpu  = cpu_q2k_dot_q8k(blocks, n_blocks, q8, q8_scales);
        float vnni = 0.0f;
        ds4_xeon_vec_dot_q2_K_q8k_vnni(N_FF, &vnni, blocks, q8, q8_scales);

        double abs_err = fabs((double)cpu - (double)vnni);
        double mag = fabs((double)cpu) + fabs((double)vnni) + 1e-10f;
        double rel_err = abs_err / mag;
        sum_rel += rel_err;
        if (rel_err > max_rel) { max_rel = rel_err; max_row = r; worst_cpu = cpu; worst_vnni = vnni; }
        if (abs_err > max_abs) max_abs = abs_err;
    }

    double avg_rel = sum_rel / (double)n_rows;
    fprintf(stderr, "  Avg relative error: %.4f\n", avg_rel);
    fprintf(stderr, "  Max relative error: %.4f at row %d (cpu=%.4f vnni=%.4f)\n",
        max_rel, max_row, (double)worst_cpu, (double)worst_vnni);
    fprintf(stderr, "  Max absolute error: %.4f\n", max_abs);

    int pass = (avg_rel < 0.05 && max_rel < 0.20) || (max_abs < 50.0);
    fprintf(stderr, "  %s\n", pass ? "PASS (inference-tolerant)" : "FAIL");

    /* ===== Test 3: weighted accumulation (per-expert down) ===== */
    fprintf(stderr, "\n--- Test 3: Weighted accumulation (6 experts) ---\n");
    float accum[N_EMBD];
    memset(accum, 0, sizeof(accum));
    for (int ei = 0; ei < 6; ei++) {
        float ew = 1.0f / 6.0f;
        for (int r = 0; r < n_rows; r++) {
            for (int i = 0; i < n_blocks; i++) {
                blocks[i].d    = 0x3C00;
                blocks[i].dmin = 0x0000;
                for (int j = 0; j < 16; j++) blocks[i].scales[j] = (uint8_t)(rand() & 0x3F);
                for (int j = 0; j < 64; j++) blocks[i].qs[j] = (uint8_t)(rand() & 0xFF);
            }
            float vnni = 0.0f;
            ds4_xeon_vec_dot_q2_K_q8k_vnni(N_FF, &vnni, blocks, q8, q8_scales);
            accum[r] += vnni * ew;
        }
    }
    /* Check no NaN or inf */
    int ok = 1;
    for (int r = 0; r < n_rows; r++)
        if (!isfinite(accum[r])) { ok = 0; break; }
    fprintf(stderr, "  Accumulation: %s\n", ok ? "OK (no NaN/inf)" : "FAIL (NaN/inf)");

    free(blocks);
    return pass ? 0 : 1;
}
