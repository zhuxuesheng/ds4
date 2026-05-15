#include "../ds4_xeon.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

// Override weak zero-filled IQ2XXS lookup tables from ds4_xeon.c with
// deterministic test patterns so the IQ2XXS dot product produces non-zero results.
// Must be non-static to override weak definitions in ds4_xeon.c
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
    0x8080808080808080ULL,0x8181818181818181ULL,0x8282828282828282ULL,0x8383838383838383ULL,
    0x8484848484848484ULL,0x8585858585858585ULL,0x8686868686868686ULL,0x8787878787878787ULL,
    0x8888888888888888ULL,0x8989898989898989ULL,0x8A8A8A8A8A8A8A8AULL,0x8B8B8B8B8B8B8B8BULL,
    0x8C8C8C8C8C8C8C8CULL,0x8D8D8D8D8D8D8D8DULL,0x8E8E8E8E8E8E8E8EULL,0x8F8F8F8F8F8F8F8FULL,
    0x9090909090909090ULL,0x9191919191919191ULL,0x9292929292929292ULL,0x9393939393939393ULL,
    0x9494949494949494ULL,0x9595959595959595ULL,0x9696969696969696ULL,0x9797979797979797ULL,
    0x9898989898989898ULL,0x9999999999999999ULL,0x9A9A9A9A9A9A9A9AULL,0x9B9B9B9B9B9B9B9BULL,
    0x9C9C9C9C9C9C9C9CULL,0x9D9D9D9D9D9D9D9DULL,0x9E9E9E9E9E9E9E9EULL,0x9F9F9F9F9F9F9F9FULL,
    0xA0A0A0A0A0A0A0A0ULL,0xA1A1A1A1A1A1A1A1ULL,0xA2A2A2A2A2A2A2A2ULL,0xA3A3A3A3A3A3A3A3ULL,
    0xA4A4A4A4A4A4A4A4ULL,0xA5A5A5A5A5A5A5A5ULL,0xA6A6A6A6A6A6A6A6ULL,0xA7A7A7A7A7A7A7A7ULL,
    0xA8A8A8A8A8A8A8A8ULL,0xA9A9A9A9A9A9A9A9ULL,0xAAAAAAAAAAAAAAAAULL,0xABABABABABABABABULL,
    0xACACACACACACACACULL,0xADADADADADADADADULL,0xAEAEAEAEAEAEAEAEULL,0xAFAFAFAFAFAFAFAFULL,
    0xB0B0B0B0B0B0B0B0ULL,0xB1B1B1B1B1B1B1B1ULL,0xB2B2B2B2B2B2B2B2ULL,0xB3B3B3B3B3B3B3B3ULL,
    0xB4B4B4B4B4B4B4B4ULL,0xB5B5B5B5B5B5B5B5ULL,0xB6B6B6B6B6B6B6B6ULL,0xB7B7B7B7B7B7B7B7ULL,
    0xB8B8B8B8B8B8B8B8ULL,0xB9B9B9B9B9B9B9B9ULL,0xBABABABABABABABAULL,0xBBBBBBBBBBBBBBBBULL,
    0xBCBCBCBCBCBCBCBCULL,0xBDBDBDBDBDBDBDBDULL,0xBEBEBEBEBEBEBEBEULL,0xBFBFBFBFBFBFBFBFULL,
    0xC0C0C0C0C0C0C0C0ULL,0xC1C1C1C1C1C1C1C1ULL,0xC2C2C2C2C2C2C2C2ULL,0xC3C3C3C3C3C3C3C3ULL,
    0xC4C4C4C4C4C4C4C4ULL,0xC5C5C5C5C5C5C5C5ULL,0xC6C6C6C6C6C6C6C6ULL,0xC7C7C7C7C7C7C7C7ULL,
    0xC8C8C8C8C8C8C8C8ULL,0xC9C9C9C9C9C9C9C9ULL,0xCACACACACACACACAULL,0xCBCBCBCBCBCBCBCBULL,
    0xCCCCCCCCCCCCCCCCULL,0xCDCDCDCDCDCDCDCDULL,0xCECECECECECECECEULL,0xCFCFCFCFCFCFCFCFULL,
    0xD0D0D0D0D0D0D0D0ULL,0xD1D1D1D1D1D1D1D1ULL,0xD2D2D2D2D2D2D2D2ULL,0xD3D3D3D3D3D3D3D3ULL,
    0xD4D4D4D4D4D4D4D4ULL,0xD5D5D5D5D5D5D5D5ULL,0xD6D6D6D6D6D6D6D6ULL,0xD7D7D7D7D7D7D7D7ULL,
    0xD8D8D8D8D8D8D8D8ULL,0xD9D9D9D9D9D9D9D9ULL,0xDADADADADADADADAULL,0xDBDBDBDBDBDBDBDBULL,
    0xDCDCDCDCDCDCDCDCULL,0xDDDDDDDDDDDDDDDDULL,0xDEDEDEDEDEDEDEDEULL,0xDFDFDFDFDFDFDFDFULL,
    0xE0E0E0E0E0E0E0E0ULL,0xE1E1E1E1E1E1E1E1ULL,0xE2E2E2E2E2E2E2E2ULL,0xE3E3E3E3E3E3E3E3ULL,
    0xE4E4E4E4E4E4E4E4ULL,0xE5E5E5E5E5E5E5E5ULL,0xE6E6E6E6E6E6E6E6ULL,0xE7E7E7E7E7E7E7E7ULL,
    0xE8E8E8E8E8E8E8E8ULL,0xE9E9E9E9E9E9E9E9ULL,0xEAEAEAEAEAEAEAEAULL,0xEBEBEBEBEBEBEBEBULL,
    0xECECECECECECECECULL,0xEDEDEDEDEDEDEDEDULL,0xEEEEEEEEEEEEEEEEULL,0xEFEFEFEFEFEFEFEFULL,
    0xF0F0F0F0F0F0F0F0ULL,0xF1F1F1F1F1F1F1F1ULL,0xF2F2F2F2F2F2F2F2ULL,0xF3F3F3F3F3F3F3F3ULL,
    0xF4F4F4F4F4F4F4F4ULL,0xF5F5F5F5F5F5F5F5ULL,0xF6F6F6F6F6F6F6F6ULL,0xF7F7F7F7F7F7F7F7ULL,
    0xF8F8F8F8F8F8F8F8ULL,0xF9F9F9F9F9F9F9F9ULL,0xFAFAFAFAFAFAFAFAULL,0xFBFBFBFBFBFBFBFBULL,
    0xFCFCFCFCFCFCFCFCULL,0xFDFDFDFDFDFDFDFDULL,0xFEFEFEFEFEFEFEFEULL,0xFFFFFFFFFFFFFFFFULL,
};

// Forward declarations (definitions follow below after IQ2XXS table overrides)
static float f16_to_f32(uint16_t h);

// Scalar reference for IQ2XXS dot product (to verify vectorized implementation)
static float iq2xxs_dot_ref(const ds4_xeon_block_iq2_xxs *x,
    const int16_t *y, int n, float scale_y)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d = f16_to_f32(x[i].d) * scale_y;
        const uint16_t *qs = x[i].qs;
        const int16_t *a16 = y + (uint64_t)i * DS4_XEON_QK_K;
        int32_t dot = 0;
        for (int j = 0; j < 32; j++) {
            uint16_t q = qs[j];
            uint64_t grid = iq2xxs_grid[q & 255];
            uint8_t signs = ksigns_iq2xs[q >> 8];
            for (int k = 0; k < 8; k++) {
                int8_t v = (int8_t)((grid >> (k * 8)) & 0xFF);
                if (signs & (1 << k)) v = -v;
                dot += (int32_t)v * (int32_t)a16[j * 8 + k];
            }
        }
        sumf += d * (float)dot;
    }
    return sumf;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// fp32/fp16 conversion (from ds4_xeon_math_test original, used by Q4_K tests)
static uint16_t f32_to_f16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t val = (x & 0x7fffffff) + 0x1000;
    if (val >= 0x47800000) {
        if ((x & 0x7fffffff) >= 0x47800000) {
            if (val < 0x7f800000) return sign | 0x7c00;
            return sign | 0x7c00 | ((x >> 13) & 0x03ff);
        }
        return sign | 0x7bff;
    }
    if (val >= 0x38800000) return sign | ((val - 0x38000000) >> 13);
    if (val < 0x33000000) return sign;
    val = (x & 0x7fffffff) >> 23;
    return sign | ((((x & 0x7fffff) | 0x800000) + (0x800000 >> (val - 102))) >> (126 - val));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t frac = h & 0x03ff;
    if (exp == 0) {
        if (frac == 0) return *(float*)&sign;
        while ((frac & 0x0400) == 0) { frac <<= 1; exp--; }
        exp++; frac &= 0x03ff;
    } else if (exp == 0x1f) {
        uint32_t val = sign | 0x7f800000 | (frac << 13);
        return *(float*)&val;
    }
    uint32_t val = sign | ((exp + 112) << 23) | (frac << 13);
    return *(float*)&val;
}

static inline void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Scalar reference: per-block int8 quantization
static void quantize_a8_per_block_scalar(int8_t *out, float *scale,
    const float *in, int n_tok, int in_dim, int block_size)
{
    const int n_blocks = in_dim / block_size;
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        const float *in_row = in + (uint64_t)t * in_dim;
        int8_t *out_row = out + (uint64_t)t * in_dim;
        float *scale_row = scale + (uint64_t)t * n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            const float *bin = in_row + b * block_size;
            float amax = 1e-9f;
            for (int i = 0; i < block_size; i++) {
                float ax = fabsf(bin[i]);
                if (ax > amax) amax = ax;
            }
            float d = amax / 127.0f;
            float id = 1.0f / d;
            scale_row[b] = d;
            for (int i = 0; i < block_size; i++) {
                float v = bin[i] * id;
                if (v > 127.0f) v = 127.0f;
                if (v < -128.0f) v = -128.0f;
                out_row[b * block_size + i] = (int8_t)lrintf(v);
            }
        }
    }
}

// Dequantize int8 per-block back to float for roundtrip test
static void dequant_a8_per_block(float *out, const int8_t *in,
    const float *scale, int n_tok, int in_dim, int block_size)
{
    const int n_blocks = in_dim / block_size;
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        const int8_t *in_row = in + (uint64_t)t * in_dim;
        const float *scale_row = scale + (uint64_t)t * n_blocks;
        float *out_row = out + (uint64_t)t * in_dim;
        for (int b = 0; b < n_blocks; b++) {
            float d = scale_row[b];
            for (int i = 0; i < block_size; i++) {
                out_row[b * block_size + i] = (float)in_row[b * block_size + i] * d;
            }
        }
    }
}

// Cosine similarity between two float vectors
static float cosine_sim(const float *a, const float *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-9));
}

// Signal-to-noise ratio in dB
static float snr_db(const float *ref, const float *test, int n) {
    double signal = 0.0, noise = 0.0;
    for (int i = 0; i < n; i++) {
        signal += (double)ref[i] * (double)ref[i];
        float diff = ref[i] - test[i];
        noise += (double)diff * (double)diff;
    }
    return 10.0f * log10f((float)(signal / (noise + 1e-12)));
}

// ============================================================================
// Test 1: Roundtrip fidelity â€?FP32 â†?INT8 â†?FP32
// ============================================================================
static int test_roundtrip_a8(const char *dist_name,
    void (*gen)(float *x, int n, int seed),
    int n_tok, int in_dim, int block_size)
{
    const int n_blocks = in_dim / block_size;
    size_t nf = (size_t)n_tok * in_dim;

    float *src = (float*)aligned_alloc(64, nf * sizeof(float));
    int8_t *quant = (int8_t*)aligned_alloc(64, nf * sizeof(int8_t));
    float *scale = (float*)aligned_alloc(64, (size_t)n_tok * n_blocks * sizeof(float));
    float *recon = (float*)aligned_alloc(64, nf * sizeof(float));
    int8_t *quant_ref = (int8_t*)aligned_alloc(64, nf * sizeof(int8_t));
    float *scale_ref = (float*)aligned_alloc(64, (size_t)n_tok * n_blocks * sizeof(float));

    gen(src, nf, 42);
    memset(quant, 0, nf * sizeof(int8_t));
    memset(quant_ref, 0, nf * sizeof(int8_t));

    // AVX-512
    ds4_xeon_quantize_a8_per_block(quant, scale, src, n_tok, in_dim, block_size);
    dequant_a8_per_block(recon, quant, scale, n_tok, in_dim, block_size);

    // Scalar reference
    quantize_a8_per_block_scalar(quant_ref, scale_ref, src, n_tok, in_dim, block_size);

    float cos_sim = cosine_sim(src, recon, nf);
    float snr = snr_db(src, recon, nf);

    // Bit-exact check: AVX-512 output must match scalar
    int mismatches = 0;
    for (size_t i = 0; i < nf; i++) {
        if (quant[i] != quant_ref[i]) mismatches++;
    }
    int scale_mismatches = 0;
    for (int i = 0; i < n_tok * n_blocks; i++) {
        if (fabsf(scale[i] - scale_ref[i]) > 1e-6f) scale_mismatches++;
    }

    printf("  %-20s: cos=%.6f  SNR=%.2fdB  q_mismatch=%d/%zu  s_mismatch=%d/%d  %s\n",
        dist_name, cos_sim, snr, mismatches, nf, scale_mismatches,
        n_tok * n_blocks,
        (mismatches == 0 && scale_mismatches == 0) ? "PASS" : "FAIL");

    free(src); free(quant); free(scale); free(recon); free(quant_ref); free(scale_ref);
    return (mismatches == 0 && scale_mismatches == 0) ? 0 : 1;
}

// ============================================================================
// Test 2: Roundtrip fidelity â€?FP32 â†?INT16 â†?FP32
// ============================================================================
static int test_roundtrip_a16(const char *dist_name,
    void (*gen)(float *x, int n, int seed),
    int n_tok, int in_dim)
{
    size_t nf = (size_t)n_tok * in_dim;

    float *src = (float*)aligned_alloc(64, nf * sizeof(float));
    int16_t *quant = (int16_t*)aligned_alloc(64, nf * sizeof(int16_t));
    float *scale = (float*)aligned_alloc(64, n_tok * sizeof(float));
    float *recon = (float*)aligned_alloc(64, nf * sizeof(float));

    gen(src, nf, 123);

    // AVX-512
    ds4_xeon_quantize_a16_per_token(quant, scale, src, n_tok, in_dim);

    // Dequantize
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        float s = scale[t];
        for (int i = 0; i < in_dim; i++)
            recon[t * in_dim + i] = (float)quant[t * in_dim + i] * s;
    }

    float cos_sim = cosine_sim(src, recon, nf);
    float snr = snr_db(src, recon, nf);

    printf("  %-20s: cos=%.6f  SNR=%.2fdB  %s\n",
        dist_name, cos_sim, snr,
        cos_sim > 0.999f ? "PASS" : "BELOW THRESHOLD");

    free(src); free(quant); free(scale); free(recon);
    return (cos_sim > 0.999f) ? 0 : 1;
}

// ============================================================================
// Distribution generators
// ============================================================================
static void gen_gaussian(float *x, int n, int seed) {
    // Box-Muller using a simple LCG
    uint32_t s = (uint32_t)seed;
    for (int i = 0; i < n; i += 2) {
        s = s * 1103515245u + 12345u;
        float u1 = (float)(s & 0x7FFFFFFFu) / 2147483648.0f + 1e-9f;
        s = s * 1103515245u + 12345u;
        float u2 = (float)(s & 0x7FFFFFFFu) / 2147483648.0f + 1e-9f;
        float r = sqrtf(-2.0f * logf(u1));
        x[i] = r * cosf(2.0f * 3.14159265f * u2);
        if (i + 1 < n) x[i + 1] = r * sinf(2.0f * 3.14159265f * u2);
    }
}

static void gen_uniform(float *x, int n, int seed) {
    uint32_t s = (uint32_t)seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = (float)(s & 0xFFFFFFu) / 16777215.0f * 2.0f - 1.0f;
    }
}

static void gen_heavy_tailed(float *x, int n, int seed) {
    // Laplace-like distribution via inverse CDF of Gaussian^3
    gen_gaussian(x, n, seed);
    for (int i = 0; i < n; i++) {
        // Skew: 10% of samples are 10x larger (simulates SwiGLU mid)
        uint32_t s = (uint32_t)(seed + i) * 1103515245u + 12345u;
        if ((s & 0x3FFu) < 102u) {  // ~10%
            x[i] *= 10.0f;
        }
    }
}

// ============================================================================
// Test 3: SwiGLU mid INT8 vs INT16 comparison
// ============================================================================
static int test_swiglu_mid_comparison(int n_tok, int in_dim) {
    size_t nf = (size_t)n_tok * in_dim;
    const int block_size = 64;
    const int n_blocks = in_dim / block_size;

    float *src = (float*)aligned_alloc(64, nf * sizeof(float));
    int8_t *q8 = (int8_t*)aligned_alloc(64, nf * sizeof(int8_t));
    float *s8 = (float*)aligned_alloc(64, (size_t)n_tok * n_blocks * sizeof(float));
    float *r8 = (float*)aligned_alloc(64, nf * sizeof(float));
    int16_t *q16 = (int16_t*)aligned_alloc(64, nf * sizeof(int16_t));
    float *s16 = (float*)aligned_alloc(64, n_tok * sizeof(float));
    float *r16 = (float*)aligned_alloc(64, nf * sizeof(float));

    gen_heavy_tailed(src, nf, 777);

    // INT8 per-block
    ds4_xeon_quantize_a8_per_block(q8, s8, src, n_tok, in_dim, block_size);
    dequant_a8_per_block(r8, q8, s8, n_tok, in_dim, block_size);

    // INT16 per-token
    ds4_xeon_quantize_a16_per_token(q16, s16, src, n_tok, in_dim);
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        float s = s16[t];
        for (int i = 0; i < in_dim; i++)
            r16[t * in_dim + i] = (float)q16[t * in_dim + i] * s;
    }

    float snr8 = snr_db(src, r8, nf);
    float snr16 = snr_db(src, r16, nf);
    float delta = snr16 - snr8;

    printf("  INT8 per-block  SNR: %.2f dB\n", snr8);
    printf("  INT16 per-token  SNR: %.2f dB\n", snr16);
    printf("  INT16 advantage:  %.2f dB  %s\n", delta,
        delta > 10.0f ? "PASS (confirms plan assumption)" : "BELOW THRESHOLD");

    free(src); free(q8); free(s8); free(r8); free(q16); free(s16); free(r16);
    return (delta > 10.0f) ? 0 : 1;
}

// ============================================================================
// Test 4: Q4_K unpack / dequant correctness
// ============================================================================
static int test_q4k_unpack_u8(void) {
    ds4_xeon_block_q4_K x;
    x.d = f32_to_f16(1.5f);
    x.dmin = f32_to_f16(0.25f);
    for (int i = 0; i < 12; i++) x.scales[i] = (uint8_t)(rand() % 256);
    for (int i = 0; i < 128; i++) x.qs[i] = (uint8_t)(rand() % 256);

    uint8_t u8[256];
    float sc8[8], m8[8];
    ds4_xeon_unpack_q4_k_to_u8(u8, sc8, m8, &x);

    // Verify nibbles are in [0, 15]
    for (int i = 0; i < 256; i++) {
        if (u8[i] > 15) {
            printf("  FAIL: u8[%d] = %u > 15\n", i, u8[i]);
            return 1;
        }
    }

    // Verify against known nibble positions
    // Sub-block 0: low nibbles of qs[0..31]
    for (int k = 0; k < 32; k++) {
        if (u8[0*32 + k] != (x.qs[k] & 0x0F)) {
            printf("  FAIL: sub-block 0 mismatch at k=%d\n", k);
            return 1;
        }
    }
    // Sub-block 1: high nibbles of qs[0..31]
    for (int k = 0; k < 32; k++) {
        if (u8[1*32 + k] != (x.qs[k] >> 4)) {
            printf("  FAIL: sub-block 1 mismatch at k=%d\n", k);
            return 1;
        }
    }

    // Verify scale factors
    float d = f16_to_f32(x.d);
    float dmin = f16_to_f32(x.dmin);
    for (int j = 0; j < 8; j++) {
        uint8_t sc_v, m_v;
        ds4q_get_scale_min_k4(j, x.scales, &sc_v, &m_v);
        if (fabsf(sc8[j] - d * (float)sc_v) > 1e-6f ||
            fabsf(m8[j] - dmin * (float)m_v) > 1e-6f) {
            printf("  FAIL: scale mismatch at j=%d\n", j);
            return 1;
        }
    }

    printf("  PASS\n");
    return 0;
}

static int test_q4k_dequant_i16(void) {
    ds4_xeon_block_q4_K x;
    x.d = f32_to_f16(0.5f);
    x.dmin = f32_to_f16(0.1f);
    for (int i = 0; i < 12; i++) x.scales[i] = (uint8_t)(rand() % 64); // 6-bit
    for (int i = 0; i < 128; i++) x.qs[i] = (uint8_t)(rand() % 256);

    int16_t i16[256];
    ds4_xeon_dequant_q4_k_to_i16(i16, &x);

    // Scalar reference
    float d = f16_to_f32(x.d);
    float dmin = f16_to_f32(x.dmin);
    int mismatches = 0;
    for (int j = 0; j < 8; j++) {
        uint8_t sc_v, m_v;
        ds4q_get_scale_min_k4(j, x.scales, &sc_v, &m_v);
        float f_sc = d * (float)sc_v;
        float f_m = dmin * (float)m_v;
        const uint8_t *q_ptr = x.qs + (j / 2) * 32;
        for (int k = 0; k < 32; k++) {
            uint8_t nib = (j % 2 == 0) ? (q_ptr[k] & 0x0F) : (q_ptr[k] >> 4);
            float wf = f_sc * (float)nib - f_m;
            if (wf > 32767.0f) wf = 32767.0f;
            if (wf < -32768.0f) wf = -32768.0f;
            int16_t expected = (int16_t)lrintf(wf);
            if (i16[j * 32 + k] != expected) mismatches++;
        }
    }

    printf("  mismatches=%d/256  %s\n", mismatches,
        mismatches == 0 ? "PASS" : "FAIL");
    return (mismatches == 0) ? 0 : 1;
}

static int test_q4k_unpack_i16(void) {
    ds4_xeon_block_q4_K x;
    x.d = f32_to_f16(1.0f);
    x.dmin = f32_to_f16(0.0f);
    for (int i = 0; i < 12; i++) x.scales[i] = (uint8_t)(rand() % 256);
    for (int i = 0; i < 128; i++) x.qs[i] = (uint8_t)(rand() % 256);

    int16_t i16[256];
    ds4_xeon_unpack_q4_k_to_i16(i16, &x);

    // Verify raw nibble extraction (no dequant formula applied)
    int mismatches = 0;
    for (int j = 0; j < 8; j++) {
        const uint8_t *q_ptr = x.qs + (j / 2) * 32;
        for (int k = 0; k < 32; k++) {
            int16_t expected = (j % 2 == 0)
                ? (int16_t)(q_ptr[k] & 0x0F)
                : (int16_t)(q_ptr[k] >> 4);
            if (i16[j * 32 + k] != expected) mismatches++;
        }
    }

    printf("  mismatches=%d/256  %s\n", mismatches,
        mismatches == 0 ? "PASS" : "FAIL");
    return (mismatches == 0) ? 0 : 1;
}

// ============================================================================
// Test 5: Q4_K dequant overhead benchmark (on-the-fly vs pre-dequant)
// ============================================================================
static void bench_dequant_overhead(int n_blocks) {
    // Create random Q4_K blocks
    ds4_xeon_block_q4_K *blocks = (ds4_xeon_block_q4_K*)
        aligned_alloc(64, (size_t)n_blocks * sizeof(ds4_xeon_block_q4_K));
    srand(123);
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d = f32_to_f16(0.5f + (float)rand() / RAND_MAX);
        blocks[i].dmin = f32_to_f16(0.1f + (float)rand() / RAND_MAX);
        for (int j = 0; j < 12; j++) blocks[i].scales[j] = (uint8_t)(rand() % 64);
        for (int j = 0; j < 128; j++) blocks[i].qs[j] = (uint8_t)(rand() % 256);
    }

    int16_t *buf = (int16_t*)aligned_alloc(64, (size_t)n_blocks * 256 * sizeof(int16_t));
    int iterations = 100;
    double bytes_per_iter = (double)n_blocks * 256.0 * 2.0; // int16 output

    // Benchmark: full dequant (nibble â†?float â†?scale*q - min â†?clamp â†?int16)
    double t0 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        for (int b = 0; b < n_blocks; b++) {
            ds4_xeon_dequant_q4_k_to_i16(buf + (size_t)b * 256, &blocks[b]);
        }
    }
    double t_dequant = now_sec() - t0;
    double gb_dequant = (bytes_per_iter * iterations) / t_dequant / 1e9;

    // Benchmark: raw unpack only (nibble â†?int16, no arithmetic)
    double t1 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        for (int b = 0; b < n_blocks; b++) {
            ds4_xeon_unpack_q4_k_to_i16(buf + (size_t)b * 256, &blocks[b]);
        }
    }
    double t_unpack = now_sec() - t1;
    double gb_unpack = (bytes_per_iter * iterations) / t_unpack / 1e9;

    double overhead_pct = (t_dequant - t_unpack) / t_dequant * 100.0;
    printf("  n_blocks=%d  iterations=%d\n", n_blocks, iterations);
    printf("    Full dequant : %.2f GB/s  %.3f s\n", gb_dequant, t_dequant);
    printf("    Raw unpack   : %.2f GB/s  %.3f s\n", gb_unpack, t_unpack);
    printf("    Dequant overhead: %.1f%% %s\n", overhead_pct,
        overhead_pct < 50.0 ? "PASS (<50%)" : "HIGH");

    free(blocks); free(buf);
}

// ============================================================================
// Test 6: Quantization throughput benchmark
// ============================================================================
static void bench_quantize_a8(const char *label, int n_tok, int in_dim, int block_size) {
    size_t nf = (size_t)n_tok * in_dim;
    const int n_blocks = in_dim / block_size;
    int iterations = 1000;

    float *src = (float*)aligned_alloc(64, nf * sizeof(float));
    int8_t *dst = (int8_t*)aligned_alloc(64, nf * sizeof(int8_t));
    float *scl = (float*)aligned_alloc(64, (size_t)n_tok * n_blocks * sizeof(float));

    gen_gaussian(src, nf, 42);

    double t0 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        ds4_xeon_quantize_a8_per_block(dst, scl, src, n_tok, in_dim, block_size);
    }
    double t1 = now_sec();
    double dt = t1 - t0;

    double bytes = (double)nf * sizeof(float) * iterations;  // input only (read)
    double gb_s = bytes / dt / 1e9;

    printf("  %-20s: n_tok=%d in_dim=%d bs=%d  %.2f GB/s  %.3f s  %s\n",
        label, n_tok, in_dim, block_size, gb_s, dt,
        gb_s > 100.0 ? "PASS" : "BELOW");

    free(src); free(dst); free(scl);
}

static void bench_quantize_a16(const char *label, int n_tok, int in_dim) {
    size_t nf = (size_t)n_tok * in_dim;
    int iterations = 1000;

    float *src = (float*)aligned_alloc(64, nf * sizeof(float));
    int16_t *dst = (int16_t*)aligned_alloc(64, nf * sizeof(int16_t));
    float *scl = (float*)aligned_alloc(64, n_tok * sizeof(float));

    gen_gaussian(src, nf, 42);

    double t0 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        ds4_xeon_quantize_a16_per_token(dst, scl, src, n_tok, in_dim);
    }
    double t1 = now_sec();
    double dt = t1 - t0;

    double bytes = (double)nf * sizeof(float) * iterations;
    double gb_s = bytes / dt / 1e9;

    printf("  %-20s: n_tok=%d in_dim=%d  %.2f GB/s  %.3f s  %s\n",
        label, n_tok, in_dim, gb_s, dt,
        gb_s > 100.0 ? "PASS" : "BELOW");

    free(src); free(dst); free(scl);
}

// ============================================================================
// Test 7: IQ2XXS vectorized dot product correctness
// ============================================================================
static int test_iq2xxs_correctness(int n_blocks) {
    // Build test blocks with deterministic qs values
    ds4_xeon_block_iq2_xxs *blocks = (ds4_xeon_block_iq2_xxs*)
        aligned_alloc(64, (size_t)n_blocks * sizeof(ds4_xeon_block_iq2_xxs));
    int16_t *activations = (int16_t*)aligned_alloc(64,
        (size_t)n_blocks * DS4_XEON_QK_K * sizeof(int16_t));

    srand(42);
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d = f32_to_f16(0.5f + (float)rand() / RAND_MAX);
        for (int j = 0; j < 32; j++) {
            uint8_t lo = (uint8_t)(rand() % 256);  // grid index
            uint8_t hi = (uint8_t)(rand() % 128);  // sign byte index
            blocks[i].qs[j] = (uint16_t)((uint16_t)hi << 8) | lo;
        }
    }
    for (int k = 0; k < n_blocks * DS4_XEON_QK_K; k++) {
        activations[k] = (int16_t)((rand() % 65536) - 32768);
    }

    // Scalar reference
    float ref = iq2xxs_dot_ref(blocks, activations,
        n_blocks * DS4_XEON_QK_K, 1.0f);

    // Vectorized
    float vec = 0.0f;
    ds4_xeon_vec_dot_iq2_xxs_vnni(n_blocks * DS4_XEON_QK_K, &vec,
        blocks, activations, 1.0f);

    float rel_err = fabsf(ref - vec) / fmaxf(fabsf(ref), 1e-10f);
    printf("  n_blocks=%d  ref=%.6f  vec=%.6f  rel_err=%.2e  %s\n",
        n_blocks, ref, vec, rel_err,
        rel_err < 1e-5f ? "PASS" : "FAIL");

    free(blocks); free(activations);
    return (rel_err < 1e-5f) ? 0 : 1;
}

// ============================================================================
// Test 8: IQ2XXS throughput benchmark (vectorized vs scalar estimate)
// ============================================================================
static void bench_iq2xxs_throughput(int n_blocks) {
    ds4_xeon_block_iq2_xxs *blocks = (ds4_xeon_block_iq2_xxs*)
        aligned_alloc(64, (size_t)n_blocks * sizeof(ds4_xeon_block_iq2_xxs));
    int16_t *activations = (int16_t*)aligned_alloc(64,
        (size_t)n_blocks * DS4_XEON_QK_K * sizeof(int16_t));

    srand(123);
    for (int i = 0; i < n_blocks; i++) {
        blocks[i].d = f32_to_f16(0.5f);
        for (int j = 0; j < 32; j++) {
            blocks[i].qs[j] = (uint16_t)((uint16_t)(rand() % 128) << 8) | (uint8_t)(rand() % 256);
        }
    }
    for (int k = 0; k < n_blocks * DS4_XEON_QK_K; k++) {
        activations[k] = (int16_t)((rand() % 65536) - 32768);
    }

    int iterations = 1000;
    int total_n = n_blocks * DS4_XEON_QK_K;

    // Vectorized throughput
    float sum_v = 0.0f;
    double t0 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        float s = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(total_n, &s, blocks, activations, 1.0f);
        sum_v += s;
    }
    double dt_vec = now_sec() - t0;

    // Scalar reference throughput
    float sum_r = 0.0f;
    double t1 = now_sec();
    for (int iter = 0; iter < iterations; iter++) {
        sum_r += iq2xxs_dot_ref(blocks, activations, total_n, 1.0f);
    }
    double dt_ref = now_sec() - t1;

    double ops_per_iter = (double)n_blocks * 256.0; // 256 MACs per block
    double gops_vec = (ops_per_iter * iterations) / dt_vec / 1e9;
    double gops_ref = (ops_per_iter * iterations) / dt_ref / 1e9;
    double speedup = dt_ref / dt_vec;

    printf("  n_blocks=%d  iterations=%d\n", n_blocks, iterations);
    printf("    Scalar    : %.3f GOPS  %.3f s\n", gops_ref, dt_ref);
    printf("    Vectorized: %.3f GOPS  %.3f s\n", gops_vec, dt_vec);
    printf("    Speedup   : %.1fx  %s\n", speedup,
        speedup > 5.0 ? "PASS (>5x)" : "BELOW");

    // Suppress unused warnings for correctness sums
    volatile float prevent_opt = sum_v + sum_r;
    (void)prevent_opt;

    free(blocks); free(activations);
}

int main(void) {
    printf("=== DS4 Xeon Quantization Math Test ===\n\n");

    int failures = 0;
    const int in_dim = 4096;
    const int block_size = 64;

    // Test 1: Roundtrip fidelity (INT8)
    printf("--- Roundtrip INT8 (per-block, bs=%d) ---\n", block_size);
    failures += test_roundtrip_a8("Gaussian", gen_gaussian, 16, in_dim, block_size);
    failures += test_roundtrip_a8("Uniform", gen_uniform, 16, in_dim, block_size);
    failures += test_roundtrip_a8("Heavy-tailed", gen_heavy_tailed, 16, in_dim, block_size);

    // Test 2: Roundtrip fidelity (INT16)
    printf("\n--- Roundtrip INT16 (per-token) ---\n");
    failures += test_roundtrip_a16("Gaussian", gen_gaussian, 16, in_dim);
    failures += test_roundtrip_a16("Uniform", gen_uniform, 16, in_dim);
    failures += test_roundtrip_a16("Heavy-tailed", gen_heavy_tailed, 16, in_dim);

    // Test 3: SwiGLU mid INT8 vs INT16
    printf("\n--- SwiGLU mid: INT8 vs INT16 ---\n");
    failures += test_swiglu_mid_comparison(16, in_dim);

    // Test 4: Q4_K unpack / dequant correctness
    printf("\n--- Q4_K Unpack to uint8 ---\n");
    failures += test_q4k_unpack_u8();
    printf("--- Q4_K Unpack to int16 ---\n");
    failures += test_q4k_unpack_i16();
    printf("--- Q4_K Dequant to int16 ---\n");
    failures += test_q4k_dequant_i16();

    // Test 5: Q4_K dequant overhead benchmark
    printf("\n--- Q4_K Dequant Overhead ---\n");
    bench_dequant_overhead(4096);

    // Test 6: Quantization throughput benchmarks
    printf("\n--- Quantization Throughput ---\n");
    bench_quantize_a8("INT8 small batch", 8, in_dim, block_size);
    bench_quantize_a8("INT8 large batch", 256, in_dim, block_size);
    bench_quantize_a16("INT16 small batch", 8, in_dim);
    bench_quantize_a16("INT16 large batch", 256, in_dim);

    // Test 7: IQ2XXS vectorized dot product correctness
    printf("\n--- IQ2XXS Correctness ---\n");
    failures += test_iq2xxs_correctness(256);

    // Test 8: IQ2XXS throughput benchmark
    printf("\n--- IQ2XXS Throughput ---\n");
    bench_iq2xxs_throughput(4096);

    printf("\n=== %s ===\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return failures > 0 ? 1 : 0;
}

#else
int main(void) {
    printf("Test skipped: AVX-512 VNNI not available.\n");
    return 0;
}
#endif
