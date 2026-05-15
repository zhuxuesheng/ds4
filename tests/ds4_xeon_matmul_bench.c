#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

// ============================================================================
// Benchmark 1: VPDPBUSD (INT8 VNNI) — 16-row manually unrolled
// Each dpbusd: 64 int8 × 64 uint8 → 4 sums added to int32 accumulator
// Target: >9 TOPS (70% of 12.9 TOPS theoretical)
// ============================================================================
static void bench_vpdpbusd_int8(void) {
    int N = 4096;
    int K = 4096;
    int iterations = 10000;

    int8_t  *A8 = (int8_t*) aligned_alloc(64, (size_t)K * sizeof(int8_t));
    uint8_t *W8 = (uint8_t*)aligned_alloc(64, (size_t)N * K * sizeof(uint8_t));

    if (!A8 || !W8) {
        printf("VPDPBUSD: memory allocation failed\n");
        return;
    }

    for (int i = 0; i < K; i++) A8[i] = 1;
    for (int i = 0; i < N * K; i++) W8[i] = 1;

    int nth = omp_get_max_threads();
    printf("=== VPDPBUSD (INT8 VNNI) 16-row manual unroll ===\n");
    printf("N=%d K=%d iterations=%d threads=%d\n", N, K, iterations, nth);

    volatile int32_t sink = 0;

    double t0 = now_sec();

    #pragma omp parallel for schedule(static) reduction(+:sink)
    for (int i = 0; i < N; i += 16) {
        int tail = N - i;
        if (tail < 16) {
            // Handle tail with generic code
            uint8_t *wp[16];
            __m512i a[16];
            for (int r = 0; r < tail; r++) {
                wp[r] = W8 + (uint64_t)(i + r) * K;
                a[r] = _mm512_setzero_si512();
            }
            for (int iter = 0; iter < iterations; iter++) {
                for (int k = 0; k < K; k += 64) {
                    __m512i av = _mm512_loadu_si512((const __m512i*)(A8 + k));
                    for (int r = 0; r < tail; r++) {
                        __m512i wv = _mm512_loadu_si512((const __m512i*)(wp[r] + k));
                        a[r] = _mm512_dpbusd_epi32(a[r], wv, av);
                    }
                }
            }
            for (int r = 0; r < tail; r++)
                sink += _mm512_reduce_add_epi32(a[r]);
        } else {
            // Fast path: exactly 16 rows, fully unrolled
            uint8_t *w0 = W8 + (uint64_t)(i + 0) * K;
            uint8_t *w1 = W8 + (uint64_t)(i + 1) * K;
            uint8_t *w2 = W8 + (uint64_t)(i + 2) * K;
            uint8_t *w3 = W8 + (uint64_t)(i + 3) * K;
            uint8_t *w4 = W8 + (uint64_t)(i + 4) * K;
            uint8_t *w5 = W8 + (uint64_t)(i + 5) * K;
            uint8_t *w6 = W8 + (uint64_t)(i + 6) * K;
            uint8_t *w7 = W8 + (uint64_t)(i + 7) * K;
            uint8_t *w8 = W8 + (uint64_t)(i + 8) * K;
            uint8_t *w9 = W8 + (uint64_t)(i + 9) * K;
            uint8_t *wa = W8 + (uint64_t)(i + 10) * K;
            uint8_t *wb = W8 + (uint64_t)(i + 11) * K;
            uint8_t *wc = W8 + (uint64_t)(i + 12) * K;
            uint8_t *wd = W8 + (uint64_t)(i + 13) * K;
            uint8_t *we = W8 + (uint64_t)(i + 14) * K;
            uint8_t *wf = W8 + (uint64_t)(i + 15) * K;

            for (int iter = 0; iter < iterations; iter++) {
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i acc4 = _mm512_setzero_si512();
                __m512i acc5 = _mm512_setzero_si512();
                __m512i acc6 = _mm512_setzero_si512();
                __m512i acc7 = _mm512_setzero_si512();
                __m512i acc8 = _mm512_setzero_si512();
                __m512i acc9 = _mm512_setzero_si512();
                __m512i acca = _mm512_setzero_si512();
                __m512i accb = _mm512_setzero_si512();
                __m512i accc = _mm512_setzero_si512();
                __m512i accd = _mm512_setzero_si512();
                __m512i acce = _mm512_setzero_si512();
                __m512i accf = _mm512_setzero_si512();

                for (int k = 0; k < K; k += 64) {
                    __m512i a8 = _mm512_loadu_si512((const __m512i*)(A8 + k));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + k)), a8);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + k)), a8);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + k)), a8);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + k)), a8);
                    acc4 = _mm512_dpbusd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + k)), a8);
                    acc5 = _mm512_dpbusd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + k)), a8);
                    acc6 = _mm512_dpbusd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + k)), a8);
                    acc7 = _mm512_dpbusd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + k)), a8);
                    acc8 = _mm512_dpbusd_epi32(acc8, _mm512_loadu_si512((const __m512i*)(w8 + k)), a8);
                    acc9 = _mm512_dpbusd_epi32(acc9, _mm512_loadu_si512((const __m512i*)(w9 + k)), a8);
                    acca = _mm512_dpbusd_epi32(acca, _mm512_loadu_si512((const __m512i*)(wa + k)), a8);
                    accb = _mm512_dpbusd_epi32(accb, _mm512_loadu_si512((const __m512i*)(wb + k)), a8);
                    accc = _mm512_dpbusd_epi32(accc, _mm512_loadu_si512((const __m512i*)(wc + k)), a8);
                    accd = _mm512_dpbusd_epi32(accd, _mm512_loadu_si512((const __m512i*)(wd + k)), a8);
                    acce = _mm512_dpbusd_epi32(acce, _mm512_loadu_si512((const __m512i*)(we + k)), a8);
                    accf = _mm512_dpbusd_epi32(accf, _mm512_loadu_si512((const __m512i*)(wf + k)), a8);
                }

                sink += _mm512_reduce_add_epi32(acc0);
                sink += _mm512_reduce_add_epi32(acc1);
                sink += _mm512_reduce_add_epi32(acc2);
                sink += _mm512_reduce_add_epi32(acc3);
                sink += _mm512_reduce_add_epi32(acc4);
                sink += _mm512_reduce_add_epi32(acc5);
                sink += _mm512_reduce_add_epi32(acc6);
                sink += _mm512_reduce_add_epi32(acc7);
                sink += _mm512_reduce_add_epi32(acc8);
                sink += _mm512_reduce_add_epi32(acc9);
                sink += _mm512_reduce_add_epi32(acca);
                sink += _mm512_reduce_add_epi32(accb);
                sink += _mm512_reduce_add_epi32(accc);
                sink += _mm512_reduce_add_epi32(accd);
                sink += _mm512_reduce_add_epi32(acce);
                sink += _mm512_reduce_add_epi32(accf);
            }
        }
    }

    double t1 = now_sec();
    double dt = t1 - t0;

    double ops = (double)N * K * 2.0 * iterations;
    double tops = ops / dt / 1e12;
    printf("Throughput: %.2f TOPS\n", tops);
    printf("Time:       %.3f s\n", dt);
    printf("Target:     >9.0 TOPS (70%% of 12.9 TOPS peak)\n");
    printf("Status:     %s\n\n", tops >= 9.0 ? "PASS" : "BELOW TARGET");

    free(A8); free(W8);
}

// ============================================================================
// Benchmark 2: VPDPBUSD (INT8 VNNI) — 8-row manually unrolled for comparison
// ============================================================================
static void bench_vpdpbusd_int8_8row(void) {
    int N = 4096;
    int K = 4096;
    int iterations = 10000;

    int8_t  *A8 = (int8_t*) aligned_alloc(64, (size_t)K * sizeof(int8_t));
    uint8_t *W8 = (uint8_t*)aligned_alloc(64, (size_t)N * K * sizeof(uint8_t));

    if (!A8 || !W8) {
        printf("VPDPBUSD-8row: memory allocation failed\n");
        return;
    }

    for (int i = 0; i < K; i++) A8[i] = 1;
    for (int i = 0; i < N * K; i++) W8[i] = 1;

    int nth = omp_get_max_threads();
    printf("=== VPDPBUSD (INT8 VNNI) 8-row manual unroll ===\n");
    printf("N=%d K=%d iterations=%d threads=%d\n", N, K, iterations, nth);

    volatile int32_t sink = 0;

    double t0 = now_sec();

    #pragma omp parallel for schedule(static) reduction(+:sink)
    for (int i = 0; i < N; i += 8) {
        uint8_t *w0 = W8 + (uint64_t)(i + 0) * K;
        uint8_t *w1 = W8 + (uint64_t)(i + 1) * K;
        uint8_t *w2 = W8 + (uint64_t)(i + 2) * K;
        uint8_t *w3 = W8 + (uint64_t)(i + 3) * K;
        uint8_t *w4 = W8 + (uint64_t)(i + 4) * K;
        uint8_t *w5 = W8 + (uint64_t)(i + 5) * K;
        uint8_t *w6 = W8 + (uint64_t)(i + 6) * K;
        uint8_t *w7 = W8 + (uint64_t)(i + 7) * K;

        for (int iter = 0; iter < iterations; iter++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();

            for (int k = 0; k < K; k += 64) {
                __m512i a8 = _mm512_loadu_si512((const __m512i*)(A8 + k));
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + k)), a8);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + k)), a8);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + k)), a8);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + k)), a8);
                acc4 = _mm512_dpbusd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + k)), a8);
                acc5 = _mm512_dpbusd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + k)), a8);
                acc6 = _mm512_dpbusd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + k)), a8);
                acc7 = _mm512_dpbusd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + k)), a8);
            }

            sink += _mm512_reduce_add_epi32(acc0);
            sink += _mm512_reduce_add_epi32(acc1);
            sink += _mm512_reduce_add_epi32(acc2);
            sink += _mm512_reduce_add_epi32(acc3);
            sink += _mm512_reduce_add_epi32(acc4);
            sink += _mm512_reduce_add_epi32(acc5);
            sink += _mm512_reduce_add_epi32(acc6);
            sink += _mm512_reduce_add_epi32(acc7);
        }
    }

    double t1 = now_sec();
    double dt = t1 - t0;

    double ops = (double)N * K * 2.0 * iterations;
    double tops = ops / dt / 1e12;
    printf("Throughput: %.2f TOPS\n", tops);
    printf("Time:       %.3f s\n", dt);
    printf("Target:     >9.0 TOPS\n");
    printf("Status:     %s\n\n", tops >= 9.0 ? "PASS" : "BELOW TARGET");

    free(A8); free(W8);
}

// ============================================================================
// Benchmark 3: VPDPWSSD (INT16 VNNI) — 8-row manually unrolled
// Each dpwssd: 32 int16 × 32 int16 → 4 sums added to int32 accumulator
// Target: >4.5 TOPS (70% of 6.4 TOPS theoretical)
// ============================================================================
static void bench_vpdpwssd_int16(void) {
    int N = 4096;
    int K = 4096;
    int iterations = 10000;

    int16_t *A16 = (int16_t*)aligned_alloc(64, (size_t)K * sizeof(int16_t));
    int16_t *W16 = (int16_t*)aligned_alloc(64, (size_t)N * K * sizeof(int16_t));

    if (!A16 || !W16) {
        printf("VPDPWSSD: memory allocation failed\n");
        return;
    }

    for (int i = 0; i < K; i++) A16[i] = 1;
    for (int i = 0; i < N * K; i++) W16[i] = 1;

    int nth = omp_get_max_threads();
    printf("=== VPDPWSSD (INT16 VNNI) 8-row manual unroll ===\n");
    printf("N=%d K=%d iterations=%d threads=%d\n", N, K, iterations, nth);

    volatile int32_t sink = 0;

    double t0 = now_sec();

    #pragma omp parallel for schedule(static) reduction(+:sink)
    for (int i = 0; i < N; i += 8) {
        int16_t *w0 = W16 + (uint64_t)(i + 0) * K;
        int16_t *w1 = W16 + (uint64_t)(i + 1) * K;
        int16_t *w2 = W16 + (uint64_t)(i + 2) * K;
        int16_t *w3 = W16 + (uint64_t)(i + 3) * K;
        int16_t *w4 = W16 + (uint64_t)(i + 4) * K;
        int16_t *w5 = W16 + (uint64_t)(i + 5) * K;
        int16_t *w6 = W16 + (uint64_t)(i + 6) * K;
        int16_t *w7 = W16 + (uint64_t)(i + 7) * K;

        for (int iter = 0; iter < iterations; iter++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();

            for (int k = 0; k < K; k += 32) {
                __m512i a16 = _mm512_loadu_si512((const __m512i*)(A16 + k));
                acc0 = _mm512_dpwssd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + k)), a16);
                acc1 = _mm512_dpwssd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + k)), a16);
                acc2 = _mm512_dpwssd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + k)), a16);
                acc3 = _mm512_dpwssd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + k)), a16);
                acc4 = _mm512_dpwssd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + k)), a16);
                acc5 = _mm512_dpwssd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + k)), a16);
                acc6 = _mm512_dpwssd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + k)), a16);
                acc7 = _mm512_dpwssd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + k)), a16);
            }

            sink += _mm512_reduce_add_epi32(acc0);
            sink += _mm512_reduce_add_epi32(acc1);
            sink += _mm512_reduce_add_epi32(acc2);
            sink += _mm512_reduce_add_epi32(acc3);
            sink += _mm512_reduce_add_epi32(acc4);
            sink += _mm512_reduce_add_epi32(acc5);
            sink += _mm512_reduce_add_epi32(acc6);
            sink += _mm512_reduce_add_epi32(acc7);
        }
    }

    double t1 = now_sec();
    double dt = t1 - t0;

    double ops = (double)N * K * 2.0 * iterations;
    double tops = ops / dt / 1e12;
    printf("Throughput: %.2f TOPS\n", tops);
    printf("Time:       %.3f s\n", dt);
    printf("Target:     >4.5 TOPS (70%% of 6.4 TOPS peak)\n");
    printf("Status:     %s\n\n", tops >= 4.5 ? "PASS" : "BELOW TARGET");

    free(A16); free(W16);
}

// ============================================================================
// Benchmark 4: VPDPWSSD (INT16 VNNI) — 16-row manually unrolled
// ============================================================================
static void bench_vpdpwssd_int16_16row(void) {
    int N = 4096;
    int K = 4096;
    int iterations = 10000;

    int16_t *A16 = (int16_t*)aligned_alloc(64, (size_t)K * sizeof(int16_t));
    int16_t *W16 = (int16_t*)aligned_alloc(64, (size_t)N * K * sizeof(int16_t));

    if (!A16 || !W16) {
        printf("VPDPWSSD-16row: memory allocation failed\n");
        return;
    }

    for (int i = 0; i < K; i++) A16[i] = 1;
    for (int i = 0; i < N * K; i++) W16[i] = 1;

    int nth = omp_get_max_threads();
    printf("=== VPDPWSSD (INT16 VNNI) 16-row manual unroll ===\n");
    printf("N=%d K=%d iterations=%d threads=%d\n", N, K, iterations, nth);

    volatile int32_t sink = 0;

    double t0 = now_sec();

    #pragma omp parallel for schedule(static) reduction(+:sink)
    for (int i = 0; i < N; i += 16) {
        int tail = N - i;
        if (tail < 16) {
            int16_t *wp[16];
            __m512i a[16];
            for (int r = 0; r < tail; r++) {
                wp[r] = W16 + (uint64_t)(i + r) * K;
                a[r] = _mm512_setzero_si512();
            }
            for (int iter = 0; iter < iterations; iter++) {
                for (int k = 0; k < K; k += 32) {
                    __m512i av = _mm512_loadu_si512((const __m512i*)(A16 + k));
                    for (int r = 0; r < tail; r++) {
                        __m512i wv = _mm512_loadu_si512((const __m512i*)(wp[r] + k));
                        a[r] = _mm512_dpwssd_epi32(a[r], wv, av);
                    }
                }
            }
            for (int r = 0; r < tail; r++)
                sink += _mm512_reduce_add_epi32(a[r]);
        } else {
            int16_t *w0 = W16 + (uint64_t)(i + 0) * K;
            int16_t *w1 = W16 + (uint64_t)(i + 1) * K;
            int16_t *w2 = W16 + (uint64_t)(i + 2) * K;
            int16_t *w3 = W16 + (uint64_t)(i + 3) * K;
            int16_t *w4 = W16 + (uint64_t)(i + 4) * K;
            int16_t *w5 = W16 + (uint64_t)(i + 5) * K;
            int16_t *w6 = W16 + (uint64_t)(i + 6) * K;
            int16_t *w7 = W16 + (uint64_t)(i + 7) * K;
            int16_t *w8 = W16 + (uint64_t)(i + 8) * K;
            int16_t *w9 = W16 + (uint64_t)(i + 9) * K;
            int16_t *wa = W16 + (uint64_t)(i + 10) * K;
            int16_t *wb = W16 + (uint64_t)(i + 11) * K;
            int16_t *wc = W16 + (uint64_t)(i + 12) * K;
            int16_t *wd = W16 + (uint64_t)(i + 13) * K;
            int16_t *we = W16 + (uint64_t)(i + 14) * K;
            int16_t *wf = W16 + (uint64_t)(i + 15) * K;

            for (int iter = 0; iter < iterations; iter++) {
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i acc4 = _mm512_setzero_si512();
                __m512i acc5 = _mm512_setzero_si512();
                __m512i acc6 = _mm512_setzero_si512();
                __m512i acc7 = _mm512_setzero_si512();
                __m512i acc8 = _mm512_setzero_si512();
                __m512i acc9 = _mm512_setzero_si512();
                __m512i acca = _mm512_setzero_si512();
                __m512i accb = _mm512_setzero_si512();
                __m512i accc = _mm512_setzero_si512();
                __m512i accd = _mm512_setzero_si512();
                __m512i acce = _mm512_setzero_si512();
                __m512i accf = _mm512_setzero_si512();

                for (int k = 0; k < K; k += 32) {
                    __m512i a16 = _mm512_loadu_si512((const __m512i*)(A16 + k));
                    acc0 = _mm512_dpwssd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + k)), a16);
                    acc1 = _mm512_dpwssd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + k)), a16);
                    acc2 = _mm512_dpwssd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + k)), a16);
                    acc3 = _mm512_dpwssd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + k)), a16);
                    acc4 = _mm512_dpwssd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + k)), a16);
                    acc5 = _mm512_dpwssd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + k)), a16);
                    acc6 = _mm512_dpwssd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + k)), a16);
                    acc7 = _mm512_dpwssd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + k)), a16);
                    acc8 = _mm512_dpwssd_epi32(acc8, _mm512_loadu_si512((const __m512i*)(w8 + k)), a16);
                    acc9 = _mm512_dpwssd_epi32(acc9, _mm512_loadu_si512((const __m512i*)(w9 + k)), a16);
                    acca = _mm512_dpwssd_epi32(acca, _mm512_loadu_si512((const __m512i*)(wa + k)), a16);
                    accb = _mm512_dpwssd_epi32(accb, _mm512_loadu_si512((const __m512i*)(wb + k)), a16);
                    accc = _mm512_dpwssd_epi32(accc, _mm512_loadu_si512((const __m512i*)(wc + k)), a16);
                    accd = _mm512_dpwssd_epi32(accd, _mm512_loadu_si512((const __m512i*)(wd + k)), a16);
                    acce = _mm512_dpwssd_epi32(acce, _mm512_loadu_si512((const __m512i*)(we + k)), a16);
                    accf = _mm512_dpwssd_epi32(accf, _mm512_loadu_si512((const __m512i*)(wf + k)), a16);
                }

                sink += _mm512_reduce_add_epi32(acc0);
                sink += _mm512_reduce_add_epi32(acc1);
                sink += _mm512_reduce_add_epi32(acc2);
                sink += _mm512_reduce_add_epi32(acc3);
                sink += _mm512_reduce_add_epi32(acc4);
                sink += _mm512_reduce_add_epi32(acc5);
                sink += _mm512_reduce_add_epi32(acc6);
                sink += _mm512_reduce_add_epi32(acc7);
                sink += _mm512_reduce_add_epi32(acc8);
                sink += _mm512_reduce_add_epi32(acc9);
                sink += _mm512_reduce_add_epi32(acca);
                sink += _mm512_reduce_add_epi32(accb);
                sink += _mm512_reduce_add_epi32(accc);
                sink += _mm512_reduce_add_epi32(accd);
                sink += _mm512_reduce_add_epi32(acce);
                sink += _mm512_reduce_add_epi32(accf);
            }
        }
    }

    double t1 = now_sec();
    double dt = t1 - t0;

    double ops = (double)N * K * 2.0 * iterations;
    double tops = ops / dt / 1e12;
    printf("Throughput: %.2f TOPS\n", tops);
    printf("Time:       %.3f s\n", dt);
    printf("Target:     >4.5 TOPS\n");
    printf("Status:     %s\n\n", tops >= 4.5 ? "PASS" : "BELOW TARGET");

    free(A16); free(W16);
}

#else
static void bench_vpdpbusd_int8(void) {
    printf("AVX-512 VNNI not available. Compile with -march=native on Ice Lake or newer.\n");
}
static void bench_vpdpbusd_int8_8row(void) {
    printf("AVX-512 VNNI not available.\n");
}
static void bench_vpdpwssd_int16(void) {
    printf("AVX-512 VNNI not available.\n");
}
static void bench_vpdpwssd_int16_16row(void) {
    printf("AVX-512 VNNI not available.\n");
}
#endif

int main(void) {
    printf("DS4 Xeon Matmul Micro-Benchmark\n");
    printf("================================\n\n");

    bench_vpdpbusd_int8();        // 16-row VPDPBUSD
    bench_vpdpbusd_int8_8row();   // 8-row VPDPBUSD for comparison
    bench_vpdpwssd_int16();       // 8-row VPDPWSSD
    bench_vpdpwssd_int16_16row(); // 16-row VPDPWSSD for comparison

    printf("================================\n");
    printf("Benchmark complete.\n");
    return 0;
}
