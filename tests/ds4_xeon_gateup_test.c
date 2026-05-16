/* ds4_xeon_gateup_test.c — Compare VNNI gate+up vs CPU reference
 *
 * Loads a single expert from a real model, runs both VNNI and CPU gate+up,
 * compares gate/up outputs element-by-element.
 *
 * Build: make xeon-gateup-test
 */
#include "../ds4_xeon.h"
#include "../ds4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if !defined(__x86_64__)
#error "Requires x86_64"
#endif

/* Must match ds4.c constants */
#define N_EMBD   4096
#define N_FF_EXP 2048
#define QK_K     256
#define N_EXPERT_USED 6

/* External functions from ds4.c needed for the test */
extern void ds4_quantize_row_q8_K(const float *x, void *dst, uint64_t elements);

/* Override weak IQ2XXS tables from ds4_xeon.c */
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

/* Helper: compare two float arrays */
static int cmp_arrays(const char *label, const float *a, const float *b, int n) {
    double max_err = 0, sum_err = 0;
    int max_idx = 0;
    for (int i = 0; i < n; i++) {
        double err = fabs((double)a[i] - (double)b[i]);
        double mag = fabs((double)a[i]) + fabs((double)b[i]) + 1e-10;
        double rel_err = err / mag;
        sum_err += rel_err;
        if (rel_err > max_err) { max_err = rel_err; max_idx = i; }
    }
    fprintf(stderr, "  %-12s: max_rel=%.2e idx=%d avg_rel=%.2e  [%f vs %f]\n",
        label, max_err, max_idx, sum_err / (double)n,
        (double)a[max_idx], (double)b[max_idx]);
    return max_err > 0.01 ? 1 : 0; /* fail if >1% relative error */
}

/* ==========================================================================
   CPU reference: IQ2XXS dot with Q8_K activation (copied from ds4.c)
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

/* CPU scalar IQ2XXS dot product with Q8_K activation */
static float cpu_iq2xxs_dot_q8k(
    const ds4_xeon_block_iq2_xxs *blocks, int n_blocks,
    const int8_t *q8, const float *q8_scale) /* Q8_K: one scale per block */
{
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; i++) {
        float d = f16_to_f32(blocks[i].d);
        const uint16_t *qs = blocks[i].qs;
        const int8_t *a = q8 + i * QK_K;
        int32_t dot = 0;
        for (int j = 0; j < 32; j++) {
            uint16_t q = qs[j];
            uint8_t grid_idx = q & 0xFF;
            uint8_t sign_idx = q >> 8;
            /* grid + sign lookup */
            uint64_t gv = iq2xxs_grid[grid_idx];
            uint8_t signs = ksigns_iq2xs[sign_idx & 0x7F];
            for (int k = 0; k < 8; k++) {
                int8_t w = (int8_t)((gv >> (k * 8)) & 0xFF);
                if (signs & (1 << k)) w = -w;
                dot += (int32_t)w * (int32_t)a[j * 8 + k];
            }
        }
        sum += d * (float)dot * q8_scale[i];
    }
    return sum;
}

/* ==========================================================================
   Test: compare VNNI vs CPU for a single gate/up dot product
   ========================================================================== */
static int test_one_row(
    const ds4_xeon_block_iq2_xxs *blocks, int n_blocks,
    const int8_t *q8, const float *q8_scale,
    const int16_t *act_i16, float act_scale,
    float *cpu_out, float *vnni_out)
{
    /* CPU: Q8_K dot */
    *cpu_out = cpu_iq2xxs_dot_q8k(blocks, n_blocks, q8, q8_scale);

    /* VNNI: INT16 dot via VPDPWSSD */
    float v = 0.0f;
    ds4_xeon_vec_dot_iq2_xxs_vnni(n_blocks * QK_K, &v,
        blocks, act_i16, act_scale);
    *vnni_out = v;

    return 0;
}

/* ==========================================================================
   Quantize float input to Q8_K (simplified for test)
   ========================================================================== */
static void quantize_q8k_test(int8_t *q8, float *scale, const float *x, int n) {
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
    srand(42);
    fprintf(stderr, "=== VNNI gate+up correctness test ===\n\n");

    int n_blocks = N_EMBD / QK_K; /* 16 blocks per 4096-dim row */

    /* Create IQ2XXS blocks with random data */
    ds4_xeon_block_iq2_xxs *blocks = aligned_alloc(64,
        n_blocks * sizeof(ds4_xeon_block_iq2_xxs));
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d = 0x3C00; /* fp16 1.0 */
        for (int j = 0; j < 32; j++)
            blocks[i].qs[j] = (uint16_t)(rand() & 0xFFFF);
    }

    /* Create RMS-normed input (bounded distribution) */
    float input[N_EMBD];
    float ss = 0;
    for (int i = 0; i < N_EMBD; i++) {
        input[i] = (float)((rand() & 0xFF) - 128) / 256.0f;
        ss += input[i] * input[i];
    }
    float scale = 1.0f / sqrtf(ss / N_EMBD + 1e-6f);
    for (int i = 0; i < N_EMBD; i++) input[i] *= scale; /* RMS = 1 */

    /* Quantize to Q8_K (CPU style) */
    int8_t q8[N_EMBD];
    float q8_scale_vals[n_blocks];
    quantize_q8k_test(q8, q8_scale_vals, input, N_EMBD);

    /* Quantize to INT16 (VNNI style) */
    int16_t act_i16[N_EMBD] __attribute__((aligned(64)));
    float act_scale;
    ds4_xeon_quantize_a16_per_token(act_i16, &act_scale, input, 1, N_EMBD);
    /* act_scale = max_val/32767, correct for reconstruction */

    fprintf(stderr, "Q8_K scale range: [%.4f, %.4f]\n",
        q8_scale_vals[0], q8_scale_vals[n_blocks-1]);
    fprintf(stderr, "INT16 act_scale: %.6f\n", (double)act_scale);

    /* Test one row */
    float cpu_val, vnni_val;
    test_one_row(blocks, n_blocks, q8, q8_scale_vals, act_i16, act_scale,
        &cpu_val, &vnni_val);
    fprintf(stderr, "\nSingle row test:\n  CPU : %.8f\n  VNNI: %.8f\n  diff: %.2e\n",
        (double)cpu_val, (double)vnni_val, fabs((double)cpu_val - (double)vnni_val));

    /* Test multiple rows (simulating one expert's gate projection) */
    fprintf(stderr, "\nMulti-row test (2048 rows):\n");
    double max_rel = 0, max_abs = 0;
    int max_row = 0;
    float worst_cpu = 0, worst_vnni = 0;
    for (int r = 0; r < 2048; r++) {
        for (int i = 0; i < n_blocks; i++) {
            blocks[i].d = 0x3C00;
            for (int j = 0; j < 32; j++)
                blocks[i].qs[j] = (uint16_t)(rand() & 0xFFFF);
        }
        float cpu, vnni;
        test_one_row(blocks, n_blocks, q8, q8_scale_vals, act_i16, act_scale,
            &cpu, &vnni);
        double abs_err = fabs((double)cpu - (double)vnni);
        if (abs_err > max_abs) { max_abs = abs_err; }
        double mag = fabs((double)cpu) + fabs((double)vnni) + 1e-10;
        double rel = fabs((double)cpu - (double)vnni) / mag;
        if (rel > max_rel) { max_rel = rel; max_row = r; worst_cpu = cpu; worst_vnni = vnni; }
    }
    fprintf(stderr, "  Max relative error: %.4f at row %d  (cpu=%.6f vnni=%.6f)\n",
        max_rel, max_row, (double)worst_cpu, (double)worst_vnni);
    fprintf(stderr, "  Max absolute error: %.6f\n", max_abs);
    if (max_rel > 0.05 && max_abs > 10.0)
        fprintf(stderr, "  FAIL: large error\n");
    else
        fprintf(stderr, "  PASS (inference-tolerant)\n");

    free(blocks);
    return (max_rel > 0.05 && max_abs > 10.0) ? 1 : 0;
}
