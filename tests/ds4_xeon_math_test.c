#include "../ds4_xeon.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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
// Test 1: Roundtrip fidelity — FP32 → INT8 → FP32
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
// Test 2: Roundtrip fidelity — FP32 → INT16 → FP32
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
// Test 4: Quantization throughput benchmark
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

    // Test 4: Throughput benchmarks
    printf("\n--- Quantization Throughput ---\n");
    bench_quantize_a8("INT8 small batch", 8, in_dim, block_size);
    bench_quantize_a8("INT8 large batch", 256, in_dim, block_size);
    bench_quantize_a16("INT16 small batch", 8, in_dim);
    bench_quantize_a16("INT16 large batch", 256, in_dim);

    printf("\n=== %s ===\n", failures == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return failures > 0 ? 1 : 0;
}

#else
int main(void) {
    printf("Test skipped: AVX-512 VNNI not available.\n");
    return 0;
}
#endif
