#include "../ds4_xeon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

static inline void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// Convert fp32 to fp16 (rough approximation for test)
static uint16_t f32_to_f16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t val = (x & 0x7fffffff) + 0x1000; // rounding bias
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

// Convert fp16 to fp32
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

void scalar_q4_K_dot(int n, float *s, const ds4_xeon_block_q4_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y) {
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * scale_y;
        const float dmin = f16_to_f32(x[i].dmin) * scale_y;

        const uint8_t *q4 = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int16_t *a16 = y_i16 + i * DS4_XEON_QK_K;
        const int32_t *a32_sum = y_sum_32 + i * (DS4_XEON_QK_K / 32);

        for (int j = 0; j < 8; j++) {
            uint8_t sc_val, m_val;
            ds4q_get_scale_min_k4(j, sc, &sc_val, &m_val);

            const uint8_t *q_ptr = q4 + (j / 2) * 32;
            int32_t dot = 0;
            
            for (int k = 0; k < 32; k++) {
                uint8_t q_val = q_ptr[k];
                uint8_t l;
                if (j % 2 == 0) l = q_val & 0x0F;
                else l = q_val >> 4;
                dot += (int32_t)l * (int32_t)a16[j * 32 + k];
            }
            sumf += (d * sc_val) * (float)dot - (dmin * m_val) * (float)a32_sum[j];
        }
    }
    *s += sumf;
}

int main() {
    int N = DS4_XEON_QK_K;
    
    ds4_xeon_block_q4_K *x = aligned_alloc(64, sizeof(ds4_xeon_block_q4_K));
    int16_t *y = aligned_alloc(64, N * sizeof(int16_t));
    int32_t *y_sum = aligned_alloc(64, (N / 32) * sizeof(int32_t));
    
    // Init random data
    x->d = f32_to_f16(1.5f);
    x->dmin = f32_to_f16(0.5f);
    for (int i=0; i<12; i++) x->scales[i] = rand() % 256;
    for (int i=0; i<128; i++) x->qs[i] = rand() % 256;
    
    for (int i=0; i<N; i++) y[i] = (rand() % 100) - 50;
    
    for (int j=0; j<8; j++) {
        y_sum[j] = 0;
        for (int k=0; k<32; k++) y_sum[j] += y[j*32 + k];
    }
    
    float s_scalar = 0.0f;
    scalar_q4_K_dot(N, &s_scalar, x, y, y_sum, 2.0f);
    
    float s_vnni = 0.0f;
    ds4_xeon_vec_dot_q4_K_vnni(N, &s_vnni, x, y, y_sum, 2.0f);
    
    printf("Scalar Result: %f\n", s_scalar);
    printf("VNNI Result:   %f\n", s_vnni);
    
    if (fabsf(s_scalar - s_vnni) > 1e-3) {
        printf("FAIL: Results do not match!\n");
        return 1;
    }
    
    printf("PASS: Math logic is correct.\n");
    return 0;
}

#else
int main() {
    printf("Test skipped: AVX512 VNNI not available.\n");
    return 0;
}
#endif