#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

// A simple timer
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

static void bench_vnni_multi_thread(void) {
    int N = 4096; // Output features
    int K = 4096; // Input features
    int iterations = 10000;
    
    // Allocate memory for simulated weights and activations
    int16_t *W16 = aligned_alloc(64, (size_t)N * K * sizeof(int16_t));
    int16_t *A16 = aligned_alloc(64, (size_t)K * sizeof(int16_t));
    int32_t *C32 = aligned_alloc(64, (size_t)N * sizeof(int32_t));
    
    if (!W16 || !A16 || !C32) {
        printf("Memory allocation failed.\n");
        return;
    }
    
    for (int i = 0; i < N * K; i++) W16[i] = 1;
    for (int i = 0; i < K; i++) A16[i] = 1;
    for (int i = 0; i < N; i++) C32[i] = 0;

    int num_threads = omp_get_max_threads();
    printf("Starting multi-thread VNNI W4A16 micro-benchmark with %d threads...\n", num_threads);
    double t0 = now_sec();
    
    // Multi-threaded across output dimension N, unrolled by 8
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += 8) {
        for (int iter = 0; iter < iterations; iter++) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();

            int16_t *w_ptr0 = &W16[(i + 0) * K];
            int16_t *w_ptr1 = &W16[(i + 1) * K];
            int16_t *w_ptr2 = &W16[(i + 2) * K];
            int16_t *w_ptr3 = &W16[(i + 3) * K];
            int16_t *w_ptr4 = &W16[(i + 4) * K];
            int16_t *w_ptr5 = &W16[(i + 5) * K];
            int16_t *w_ptr6 = &W16[(i + 6) * K];
            int16_t *w_ptr7 = &W16[(i + 7) * K];

            for (int k = 0; k < K; k += 32) {
                __m512i a = _mm512_load_si512(&A16[k]);

                acc0 = _mm512_dpwssd_epi32(acc0, _mm512_load_si512(&w_ptr0[k]), a);
                acc1 = _mm512_dpwssd_epi32(acc1, _mm512_load_si512(&w_ptr1[k]), a);
                acc2 = _mm512_dpwssd_epi32(acc2, _mm512_load_si512(&w_ptr2[k]), a);
                acc3 = _mm512_dpwssd_epi32(acc3, _mm512_load_si512(&w_ptr3[k]), a);
                acc4 = _mm512_dpwssd_epi32(acc4, _mm512_load_si512(&w_ptr4[k]), a);
                acc5 = _mm512_dpwssd_epi32(acc5, _mm512_load_si512(&w_ptr5[k]), a);
                acc6 = _mm512_dpwssd_epi32(acc6, _mm512_load_si512(&w_ptr6[k]), a);
                acc7 = _mm512_dpwssd_epi32(acc7, _mm512_load_si512(&w_ptr7[k]), a);
            }
            
            C32[i + 0] += _mm512_reduce_add_epi32(acc0);
            C32[i + 1] += _mm512_reduce_add_epi32(acc1);
            C32[i + 2] += _mm512_reduce_add_epi32(acc2);
            C32[i + 3] += _mm512_reduce_add_epi32(acc3);
            C32[i + 4] += _mm512_reduce_add_epi32(acc4);
            C32[i + 5] += _mm512_reduce_add_epi32(acc5);
            C32[i + 6] += _mm512_reduce_add_epi32(acc6);
            C32[i + 7] += _mm512_reduce_add_epi32(acc7);
        }
    }
    
    double t1 = now_sec();
    double dt = t1 - t0;
    
    // Each MAC is 2 ops (multiply and add). 
    double tops = (double)N * K * 2 * iterations / dt / 1e12;
    printf("Multi-Thread Throughput: %.2f TOPS\n", tops);
    printf("Time: %.3f s\n\n", dt);
    
    free(W16);
    free(A16);
    free(C32);
}

#else
static void bench_vnni_multi_thread(void) {
    printf("AVX512-VNNI not available at compile time. Ensure you compile with -march=native on an Ice Lake or newer CPU.\n");
}
#endif

int main(void) {
    bench_vnni_multi_thread();
    return 0;
}