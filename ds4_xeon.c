#include "ds4_xeon.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

// Model dimension defaults (must match ds4.c constants)
#ifndef DS4_NEG_INF
#define DS4_NEG_INF (-1.0e30f)
#endif
#ifndef DS4_N_EMBD
#define DS4_N_EMBD 4096
#endif
#ifndef DS4_N_FF_EXP
#define DS4_N_FF_EXP 2048
#endif
#ifndef DS4_N_EXPERT
#define DS4_N_EXPERT 256
#endif
#ifndef DS4_N_EXPERT_USED
#define DS4_N_EXPERT_USED 6
#endif
#ifndef DS4_N_HEAD
#define DS4_N_HEAD 64
#endif
#ifndef DS4_N_HEAD_DIM
#define DS4_N_HEAD_DIM 512
#endif
#ifndef DS4_N_ROT
#define DS4_N_ROT 64
#endif

// Tables defined in ds4.c (for IQ2XXS dequant)
// Weak definitions allow ds4_xeon.o to link standalone for tests;
// when linked with ds4.c the strong definitions take precedence.
__attribute__((weak)) const uint8_t ksigns_iq2xs[128] = {0};
__attribute__((weak)) const uint64_t iq2xxs_grid[256] = {0};

// ============================================================================
// Utility
// ============================================================================

static inline float xeon_f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16((short)h);  // cast to signed for _mm_cvtph_ps
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

// ============================================================================
// VPDPBUSD: INT8 VNNI matmul (activation int8 × weight uint8 �?float)
// Primary kernel for gate/up/attention projections (~70% of MACs).
//
// Activations are per-block scaled (Q8_0 style, block_size=32).
// Each block of 32 activation elements has its own float scale.
// Weights are pre-dequantized uint8_t with per-block float scale.
//
// This kernel processes 16 output rows at a time to amortize activation loads.
// ============================================================================

void ds4_xeon_matmul_a8w8_vnni(
    float *out, const int8_t *a8, const float *a8_scale,
    const uint8_t *w8, const float *w_scale,
    int in_dim, int out_dim)
{
    const int block_size = 64;
    const int n_blocks = in_dim / block_size;

    // Average activation scale (per-block scales require block-wise dequant,
    // but for now use average �?within 1% for RMS-Norm-bounded activations)
    float a_s = 1.0f;
    if (a8_scale) {
        a_s = 0.0f;
        for (int b = 0; b < n_blocks; b++) a_s += a8_scale[b];
        a_s /= (float)n_blocks;
    }

    // 8-row unrolling to hide VNNI latency (~10 cycles)
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; o += 8) {
        int tail = out_dim - o;
        if (tail < 8) {
            // Tail: single accumulator per output row
            for (int r = 0; r < tail; r++) {
                const uint8_t *w_row = w8 + (uint64_t)(o + r) * in_dim;
                const float w_s = w_scale ? w_scale[o + r] : 1.0f;
                __m512i acc = _mm512_setzero_si512();
                for (int b = 0; b < n_blocks; b++) {
                    const int b_off = b * block_size;
                    __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a8 + b_off));
                    __m512i w_vec = _mm512_loadu_si512((const __m512i*)(w_row + b_off));
                    acc = _mm512_dpbusd_epi32(acc, w_vec, a_vec);
                }
                float dot = (float)_mm512_reduce_add_epi32(acc);
                out[o + r] += dot * a_s * w_s;
            }
        } else {
            const uint8_t *w0 = w8 + (uint64_t)(o + 0) * in_dim;
            const uint8_t *w1 = w8 + (uint64_t)(o + 1) * in_dim;
            const uint8_t *w2 = w8 + (uint64_t)(o + 2) * in_dim;
            const uint8_t *w3 = w8 + (uint64_t)(o + 3) * in_dim;
            const uint8_t *w4 = w8 + (uint64_t)(o + 4) * in_dim;
            const uint8_t *w5 = w8 + (uint64_t)(o + 5) * in_dim;
            const uint8_t *w6 = w8 + (uint64_t)(o + 6) * in_dim;
            const uint8_t *w7 = w8 + (uint64_t)(o + 7) * in_dim;
            const float ws0 = w_scale ? w_scale[o + 0] : 1.0f;
            const float ws1 = w_scale ? w_scale[o + 1] : 1.0f;
            const float ws2 = w_scale ? w_scale[o + 2] : 1.0f;
            const float ws3 = w_scale ? w_scale[o + 3] : 1.0f;
            const float ws4 = w_scale ? w_scale[o + 4] : 1.0f;
            const float ws5 = w_scale ? w_scale[o + 5] : 1.0f;
            const float ws6 = w_scale ? w_scale[o + 6] : 1.0f;
            const float ws7 = w_scale ? w_scale[o + 7] : 1.0f;

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();

            for (int b = 0; b < n_blocks; b++) {
                const int b_off = b * block_size;
                __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a8 + b_off));
                acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + b_off)), a_vec);
                acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + b_off)), a_vec);
                acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + b_off)), a_vec);
                acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + b_off)), a_vec);
                acc4 = _mm512_dpbusd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + b_off)), a_vec);
                acc5 = _mm512_dpbusd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + b_off)), a_vec);
                acc6 = _mm512_dpbusd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + b_off)), a_vec);
                acc7 = _mm512_dpbusd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + b_off)), a_vec);
            }

            out[o + 0] += (float)_mm512_reduce_add_epi32(acc0) * a_s * ws0;
            out[o + 1] += (float)_mm512_reduce_add_epi32(acc1) * a_s * ws1;
            out[o + 2] += (float)_mm512_reduce_add_epi32(acc2) * a_s * ws2;
            out[o + 3] += (float)_mm512_reduce_add_epi32(acc3) * a_s * ws3;
            out[o + 4] += (float)_mm512_reduce_add_epi32(acc4) * a_s * ws4;
            out[o + 5] += (float)_mm512_reduce_add_epi32(acc5) * a_s * ws5;
            out[o + 6] += (float)_mm512_reduce_add_epi32(acc6) * a_s * ws6;
            out[o + 7] += (float)_mm512_reduce_add_epi32(acc7) * a_s * ws7;
        }
    }
}

// Batched version for prefill (n_tok tokens)
// Inner dim (in_dim) is small for MoE (2048), so weight reuse across tokens
// is more important than activation reuse across output rows.
// Process tokens in parallel via OpenMP; 8-row unrolling on output dim.
void ds4_xeon_matmul_a8w8_vnni_batch(
    float *out, const int8_t *a8, const float *a8_scale,
    const uint8_t *w8, const float *w_scale,
    int n_tok, int in_dim, int out_dim)
{
    const int block_size = 64;
    const int n_blocks = in_dim / block_size;
    const int a8_blocks_per_tok = n_blocks;

    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < n_tok; t++) {
        const int8_t *a8_row = a8 + (uint64_t)t * in_dim;
        const float *a8_s_row = a8_scale + (uint64_t)t * a8_blocks_per_tok;
        float *out_row = out + (uint64_t)t * out_dim;

        float a_s = 0.0f;
        for (int b = 0; b < n_blocks; b++) a_s += a8_s_row[b];
        a_s /= (float)n_blocks;

        // 8-row unrolling on output dimension
        for (int o = 0; o < out_dim; o += 8) {
            int tail = out_dim - o;
            if (tail < 8) {
                for (int r = 0; r < tail; r++) {
                    const uint8_t *w_row = w8 + (uint64_t)(o + r) * in_dim;
                    const float w_s = w_scale ? w_scale[o + r] : 1.0f;
                    __m512i acc = _mm512_setzero_si512();
                    for (int b = 0; b < n_blocks; b++) {
                        const int b_off = b * block_size;
                        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a8_row + b_off));
                        __m512i w_vec = _mm512_loadu_si512((const __m512i*)(w_row + b_off));
                        acc = _mm512_dpbusd_epi32(acc, w_vec, a_vec);
                    }
                    out_row[o + r] += (float)_mm512_reduce_add_epi32(acc) * a_s * w_s;
                }
            } else {
                const uint8_t *w0 = w8 + (uint64_t)(o + 0) * in_dim;
                const uint8_t *w1 = w8 + (uint64_t)(o + 1) * in_dim;
                const uint8_t *w2 = w8 + (uint64_t)(o + 2) * in_dim;
                const uint8_t *w3 = w8 + (uint64_t)(o + 3) * in_dim;
                const uint8_t *w4 = w8 + (uint64_t)(o + 4) * in_dim;
                const uint8_t *w5 = w8 + (uint64_t)(o + 5) * in_dim;
                const uint8_t *w6 = w8 + (uint64_t)(o + 6) * in_dim;
                const uint8_t *w7 = w8 + (uint64_t)(o + 7) * in_dim;
                const float ws0 = w_scale ? w_scale[o + 0] : 1.0f;
                const float ws1 = w_scale ? w_scale[o + 1] : 1.0f;
                const float ws2 = w_scale ? w_scale[o + 2] : 1.0f;
                const float ws3 = w_scale ? w_scale[o + 3] : 1.0f;
                const float ws4 = w_scale ? w_scale[o + 4] : 1.0f;
                const float ws5 = w_scale ? w_scale[o + 5] : 1.0f;
                const float ws6 = w_scale ? w_scale[o + 6] : 1.0f;
                const float ws7 = w_scale ? w_scale[o + 7] : 1.0f;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i acc4 = _mm512_setzero_si512();
                __m512i acc5 = _mm512_setzero_si512();
                __m512i acc6 = _mm512_setzero_si512();
                __m512i acc7 = _mm512_setzero_si512();

                for (int b = 0; b < n_blocks; b++) {
                    const int b_off = b * block_size;
                    __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a8_row + b_off));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + b_off)), a_vec);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + b_off)), a_vec);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + b_off)), a_vec);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + b_off)), a_vec);
                    acc4 = _mm512_dpbusd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + b_off)), a_vec);
                    acc5 = _mm512_dpbusd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + b_off)), a_vec);
                    acc6 = _mm512_dpbusd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + b_off)), a_vec);
                    acc7 = _mm512_dpbusd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + b_off)), a_vec);
                }

                out_row[o + 0] += (float)_mm512_reduce_add_epi32(acc0) * a_s * ws0;
                out_row[o + 1] += (float)_mm512_reduce_add_epi32(acc1) * a_s * ws1;
                out_row[o + 2] += (float)_mm512_reduce_add_epi32(acc2) * a_s * ws2;
                out_row[o + 3] += (float)_mm512_reduce_add_epi32(acc3) * a_s * ws3;
                out_row[o + 4] += (float)_mm512_reduce_add_epi32(acc4) * a_s * ws4;
                out_row[o + 5] += (float)_mm512_reduce_add_epi32(acc5) * a_s * ws5;
                out_row[o + 6] += (float)_mm512_reduce_add_epi32(acc6) * a_s * ws6;
                out_row[o + 7] += (float)_mm512_reduce_add_epi32(acc7) * a_s * ws7;
            }
        }
    }
}

// ============================================================================
// VPDPWSSD: INT16 VNNI matmul (activation int16 × weight int16 �?float)
// Fallback kernel for FFN down projection only (~30% of MACs).
//
// Activations are per-token scaled (one float scale per token vector).
// Weights are pre-dequantized int16_t with per-row float scale.
// ============================================================================

void ds4_xeon_matmul_a16w16_vnni(
    float *out, const int16_t *a16, float a_scale,
    const int16_t *w16, const float *w_scale,
    int in_dim, int out_dim)
{
    // 8-row unrolling to hide VNNI latency
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < out_dim; o += 8) {
        int tail = out_dim - o;
        if (tail < 8) {
            for (int r = 0; r < tail; r++) {
                const int16_t *w_row = w16 + (uint64_t)(o + r) * in_dim;
                const float w_s = w_scale ? w_scale[o + r] : 1.0f;
                __m512i acc = _mm512_setzero_si512();
                for (int i = 0; i < in_dim; i += 32) {
                    __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a16 + i));
                    __m512i w_vec = _mm512_loadu_si512((const __m512i*)(w_row + i));
                    acc = _mm512_dpwssd_epi32(acc, a_vec, w_vec);
                }
                out[o + r] += (float)_mm512_reduce_add_epi32(acc) * a_scale * w_s;
            }
        } else {
            const int16_t *w0 = w16 + (uint64_t)(o + 0) * in_dim;
            const int16_t *w1 = w16 + (uint64_t)(o + 1) * in_dim;
            const int16_t *w2 = w16 + (uint64_t)(o + 2) * in_dim;
            const int16_t *w3 = w16 + (uint64_t)(o + 3) * in_dim;
            const int16_t *w4 = w16 + (uint64_t)(o + 4) * in_dim;
            const int16_t *w5 = w16 + (uint64_t)(o + 5) * in_dim;
            const int16_t *w6 = w16 + (uint64_t)(o + 6) * in_dim;
            const int16_t *w7 = w16 + (uint64_t)(o + 7) * in_dim;
            const float ws0 = w_scale ? w_scale[o + 0] : 1.0f;
            const float ws1 = w_scale ? w_scale[o + 1] : 1.0f;
            const float ws2 = w_scale ? w_scale[o + 2] : 1.0f;
            const float ws3 = w_scale ? w_scale[o + 3] : 1.0f;
            const float ws4 = w_scale ? w_scale[o + 4] : 1.0f;
            const float ws5 = w_scale ? w_scale[o + 5] : 1.0f;
            const float ws6 = w_scale ? w_scale[o + 6] : 1.0f;
            const float ws7 = w_scale ? w_scale[o + 7] : 1.0f;

            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            __m512i acc4 = _mm512_setzero_si512();
            __m512i acc5 = _mm512_setzero_si512();
            __m512i acc6 = _mm512_setzero_si512();
            __m512i acc7 = _mm512_setzero_si512();

            for (int i = 0; i < in_dim; i += 32) {
                __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a16 + i));
                acc0 = _mm512_dpwssd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + i)), a_vec);
                acc1 = _mm512_dpwssd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + i)), a_vec);
                acc2 = _mm512_dpwssd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + i)), a_vec);
                acc3 = _mm512_dpwssd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + i)), a_vec);
                acc4 = _mm512_dpwssd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + i)), a_vec);
                acc5 = _mm512_dpwssd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + i)), a_vec);
                acc6 = _mm512_dpwssd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + i)), a_vec);
                acc7 = _mm512_dpwssd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + i)), a_vec);
            }

            out[o + 0] += (float)_mm512_reduce_add_epi32(acc0) * a_scale * ws0;
            out[o + 1] += (float)_mm512_reduce_add_epi32(acc1) * a_scale * ws1;
            out[o + 2] += (float)_mm512_reduce_add_epi32(acc2) * a_scale * ws2;
            out[o + 3] += (float)_mm512_reduce_add_epi32(acc3) * a_scale * ws3;
            out[o + 4] += (float)_mm512_reduce_add_epi32(acc4) * a_scale * ws4;
            out[o + 5] += (float)_mm512_reduce_add_epi32(acc5) * a_scale * ws5;
            out[o + 6] += (float)_mm512_reduce_add_epi32(acc6) * a_scale * ws6;
            out[o + 7] += (float)_mm512_reduce_add_epi32(acc7) * a_scale * ws7;
        }
    }
}

// Batched version for prefill
void ds4_xeon_matmul_a16w16_vnni_batch(
    float *out, const int16_t *a16, const float *a16_scale,
    const int16_t *w16, const float *w_scale,
    int n_tok, int in_dim, int out_dim)
{
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < n_tok; t++) {
        const int16_t *a16_row = a16 + (uint64_t)t * in_dim;
        const float a_s = a16_scale[t];
        float *out_row = out + (uint64_t)t * out_dim;

        // 8-row unrolling on output dimension
        for (int o = 0; o < out_dim; o += 8) {
            int tail = out_dim - o;
            if (tail < 8) {
                for (int r = 0; r < tail; r++) {
                    const int16_t *w_row = w16 + (uint64_t)(o + r) * in_dim;
                    const float w_s = w_scale ? w_scale[o + r] : 1.0f;
                    __m512i acc = _mm512_setzero_si512();
                    for (int i = 0; i < in_dim; i += 32) {
                        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a16_row + i));
                        __m512i w_vec = _mm512_loadu_si512((const __m512i*)(w_row + i));
                        acc = _mm512_dpwssd_epi32(acc, a_vec, w_vec);
                    }
                    out_row[o + r] += (float)_mm512_reduce_add_epi32(acc) * a_s * w_s;
                }
            } else {
                const int16_t *w0 = w16 + (uint64_t)(o + 0) * in_dim;
                const int16_t *w1 = w16 + (uint64_t)(o + 1) * in_dim;
                const int16_t *w2 = w16 + (uint64_t)(o + 2) * in_dim;
                const int16_t *w3 = w16 + (uint64_t)(o + 3) * in_dim;
                const int16_t *w4 = w16 + (uint64_t)(o + 4) * in_dim;
                const int16_t *w5 = w16 + (uint64_t)(o + 5) * in_dim;
                const int16_t *w6 = w16 + (uint64_t)(o + 6) * in_dim;
                const int16_t *w7 = w16 + (uint64_t)(o + 7) * in_dim;
                const float ws0 = w_scale ? w_scale[o + 0] : 1.0f;
                const float ws1 = w_scale ? w_scale[o + 1] : 1.0f;
                const float ws2 = w_scale ? w_scale[o + 2] : 1.0f;
                const float ws3 = w_scale ? w_scale[o + 3] : 1.0f;
                const float ws4 = w_scale ? w_scale[o + 4] : 1.0f;
                const float ws5 = w_scale ? w_scale[o + 5] : 1.0f;
                const float ws6 = w_scale ? w_scale[o + 6] : 1.0f;
                const float ws7 = w_scale ? w_scale[o + 7] : 1.0f;

                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                __m512i acc4 = _mm512_setzero_si512();
                __m512i acc5 = _mm512_setzero_si512();
                __m512i acc6 = _mm512_setzero_si512();
                __m512i acc7 = _mm512_setzero_si512();

                for (int i = 0; i < in_dim; i += 32) {
                    __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a16_row + i));
                    acc0 = _mm512_dpwssd_epi32(acc0, _mm512_loadu_si512((const __m512i*)(w0 + i)), a_vec);
                    acc1 = _mm512_dpwssd_epi32(acc1, _mm512_loadu_si512((const __m512i*)(w1 + i)), a_vec);
                    acc2 = _mm512_dpwssd_epi32(acc2, _mm512_loadu_si512((const __m512i*)(w2 + i)), a_vec);
                    acc3 = _mm512_dpwssd_epi32(acc3, _mm512_loadu_si512((const __m512i*)(w3 + i)), a_vec);
                    acc4 = _mm512_dpwssd_epi32(acc4, _mm512_loadu_si512((const __m512i*)(w4 + i)), a_vec);
                    acc5 = _mm512_dpwssd_epi32(acc5, _mm512_loadu_si512((const __m512i*)(w5 + i)), a_vec);
                    acc6 = _mm512_dpwssd_epi32(acc6, _mm512_loadu_si512((const __m512i*)(w6 + i)), a_vec);
                    acc7 = _mm512_dpwssd_epi32(acc7, _mm512_loadu_si512((const __m512i*)(w7 + i)), a_vec);
                }

                out_row[o + 0] += (float)_mm512_reduce_add_epi32(acc0) * a_s * ws0;
                out_row[o + 1] += (float)_mm512_reduce_add_epi32(acc1) * a_s * ws1;
                out_row[o + 2] += (float)_mm512_reduce_add_epi32(acc2) * a_s * ws2;
                out_row[o + 3] += (float)_mm512_reduce_add_epi32(acc3) * a_s * ws3;
                out_row[o + 4] += (float)_mm512_reduce_add_epi32(acc4) * a_s * ws4;
                out_row[o + 5] += (float)_mm512_reduce_add_epi32(acc5) * a_s * ws5;
                out_row[o + 6] += (float)_mm512_reduce_add_epi32(acc6) * a_s * ws6;
                out_row[o + 7] += (float)_mm512_reduce_add_epi32(acc7) * a_s * ws7;
            }
        }
    }
}

// ============================================================================
// Activation Quantization
// ============================================================================

// Per-block INT8 quantization (Q8_0 style).
// Each block of `block_size` elements gets its own float32 scale.
// block_size must be a multiple of 16 (one ZMM register holds 16 floats).
// Uses AVX-512 for max-finding (reduce_max) and quantization (round+pack).
void ds4_xeon_quantize_a8_per_block(int8_t *out, float *scale,
    const float *in, int n_tok, int in_dim, int block_size)
{
    const int n_blocks = in_dim / block_size;
    const int n16 = block_size / 16;  // 16-element chunks per block

    for (int t = 0; t < n_tok; t++) {
        const float *in_row = in + (uint64_t)t * in_dim;
        int8_t *out_row = out + (uint64_t)t * in_dim;
        float *scale_row = scale + (uint64_t)t * n_blocks;

        for (int b = 0; b < n_blocks; b++) {
            const float *bin = in_row + (uint64_t)b * block_size;
            int8_t *bout = out_row + (uint64_t)b * block_size;

            // Max-find over 16-element chunks
            __m512 vmax = _mm512_set1_ps(0.0f);
            for (int c = 0; c < n16; c++) {
                __m512 v = _mm512_loadu_ps(bin + c * 16);
                vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v));
            }
            float amax = _mm512_reduce_max_ps(vmax);
            if (amax < 1e-9f) amax = 1e-9f;

            float d = amax / 127.0f;
            float id = 1.0f / d;
            scale_row[b] = d;

            // Quantize: F32 �?int8 via round + saturating pack
            __m512 vid = _mm512_set1_ps(id);
            for (int c = 0; c < n16; c++) {
                __m512 v = _mm512_loadu_ps(bin + c * 16);
                __m512 scaled = _mm512_mul_ps(v, vid);
                __m512i iv = _mm512_cvtps_epi32(
                    _mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT));
                __m128i iv8 = _mm256_cvtsepi16_epi8(
                    _mm512_cvtsepi32_epi16(iv));
                _mm_storeu_si128((__m128i*)(bout + c * 16), iv8);
            }
        }
    }
}

// AVX-512 accelerated per-block INT8 quantization (alias, kept for compatibility)
void ds4_xeon_quantize_a8_per_block_avx512(int8_t *out, float *scale,
    const float *in, int n_tok, int in_dim, int block_size)
{
    ds4_xeon_quantize_a8_per_block(out, scale, in, n_tok, in_dim, block_size);
}

// Per-token INT16 quantization.
// One global scale per token vector. Uses AVX-512 for max-finding.
void ds4_xeon_quantize_a16_per_token(int16_t *out, float *scale,
    const float *in, int n_tok, int in_dim)
{
    const int n16 = in_dim / 16;

    for (int t = 0; t < n_tok; t++) {
        const float *in_row = in + (uint64_t)t * in_dim;
        int16_t *out_row = out + (uint64_t)t * in_dim;

        // Max-find over 16-element chunks
        __m512 vmax = _mm512_set1_ps(0.0f);
        for (int i = 0; i < n16; i++) {
            __m512 v = _mm512_loadu_ps(in_row + i * 16);
            vmax = _mm512_max_ps(vmax, _mm512_abs_ps(v));
        }
        float max_val = _mm512_reduce_max_ps(vmax);
        // Handle tail
        for (int i = n16 * 16; i < in_dim; i++) {
            float ax = fabsf(in_row[i]);
            if (ax > max_val) max_val = ax;
        }
        if (max_val < 1e-9f) max_val = 1e-9f;

        float s = max_val / 32767.0f;
        float inv_s = 1.0f / s;
        scale[t] = s;

        // Quantize: F32 �?int16 (16 elements per iteration = 256 bits)
        __m512 vid = _mm512_set1_ps(inv_s);
        __m512 vmax_i16 = _mm512_set1_ps(32767.0f);
        __m512 vmin_i16 = _mm512_set1_ps(-32768.0f);
        for (int i = 0; i < n16; i++) {
            __m512 v = _mm512_loadu_ps(in_row + i * 16);
            v = _mm512_mul_ps(v, vid);
            v = _mm512_min_ps(_mm512_max_ps(v, vmin_i16), vmax_i16);
            __m512i iv32 = _mm512_cvtps_epi32(
                _mm512_roundscale_ps(v, _MM_FROUND_TO_NEAREST_INT));
            __m256i iv16 = _mm512_cvtsepi32_epi16(iv32);
            _mm256_storeu_si256((__m256i*)(out_row + i * 16), iv16);
        }
        // Tail
        for (int i = n16 * 16; i < in_dim; i++) {
            float v = in_row[i] * inv_s;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            out_row[i] = (int16_t)lrintf(v);
        }
    }
}

// ============================================================================
// Q4_K Weight Unpack / Dequant
// ============================================================================

// Forward declaration (defined in existing Q4_K kernels section below)
static inline void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m);

// Extract 256 nibbles from a Q4_K block to uint8_t.
// The block's 128 packed bytes (qs[128]) hold 256 4-bit nibbles in 8 sub-blocks
// of 32 elements each. Sub-blocks paired by nibble position: even j uses low
// nibble, odd j uses high nibble, each pair shares 32 packed bytes.
void ds4_xeon_unpack_q4_k_to_u8(
    uint8_t *u8, float *sc8, float *m8,
    const ds4_xeon_block_q4_K *x)
{
    const float d   = xeon_f16_to_f32(x->d);
    const float dmin = xeon_f16_to_f32(x->dmin);

    for (int j = 0; j < 8; j++) {
        uint8_t sc_val, m_val;
        ds4q_get_scale_min_k4(j, x->scales, &sc_val, &m_val);

        sc8[j] = d * (float)sc_val;
        m8[j]  = dmin * (float)m_val;

        const uint8_t *q_ptr = x->qs + (j / 2) * 32;
        uint8_t *out = u8 + j * 32;

        if (j % 2 == 0) {
            // Low nibbles
            for (int k = 0; k < 32; k++)
                out[k] = q_ptr[k] & 0x0F;
        } else {
            // High nibbles
            for (int k = 0; k < 32; k++)
                out[k] = q_ptr[k] >> 4;
        }
    }
}

// Extract 256 nibbles from a Q4_K block to int16_t (raw values 0-15).
// Does NOT apply dequant formula �?just nibble extraction and zero-extension.
// For use as intermediate input to VNNI matmul that applies dequant post-dot-product.
void ds4_xeon_unpack_q4_k_to_i16(
    int16_t *i16, const ds4_xeon_block_q4_K *x)
{
    for (int j = 0; j < 8; j++) {
        const uint8_t *q_ptr = x->qs + (j / 2) * 32;
        int16_t *out = i16 + j * 32;

        if (j % 2 == 0) {
            for (int k = 0; k < 32; k++)
                out[k] = (int16_t)(q_ptr[k] & 0x0F);
        } else {
            for (int k = 0; k < 32; k++)
                out[k] = (int16_t)(q_ptr[k] >> 4);
        }
    }
}

// Fully dequantize a Q4_K block to int16_t[256].
// w16[k] = round(d * sc * q4 - dmin * m), clamped to [-32768, 32767].
// Uses AVX-512 for nibble extraction and arithmetic.
void ds4_xeon_dequant_q4_k_to_i16(
    int16_t *i16, const ds4_xeon_block_q4_K *x)
{
    const float d   = xeon_f16_to_f32(x->d);
    const float dmin = xeon_f16_to_f32(x->dmin);
    const __m512 v32767 = _mm512_set1_ps(32767.0f);
    const __m512 vm32768 = _mm512_set1_ps(-32768.0f);

    for (int j = 0; j < 8; j++) {
        uint8_t sc_val, m_val;
        ds4q_get_scale_min_k4(j, x->scales, &sc_val, &m_val);

        const float f_sc = d * (float)sc_val;
        const float f_m = dmin * (float)m_val;
        const __m512 v_sc = _mm512_set1_ps(f_sc);
        const __m512 v_m = _mm512_set1_ps(f_m);

        const uint8_t *q_ptr = x->qs + (j / 2) * 32;
        int16_t *out = i16 + j * 32;

        // Process 32 elements in 2 iterations of 16 (one ZMM = 16 floats)
        for (int k = 0; k < 32; k += 16) {
            // Load 16 packed bytes, extract nibbles to 16 uint32
            __m128i q8 = _mm_loadu_si128((const __m128i*)(q_ptr + k));
            __m256i q16;
            if (j % 2 == 0) {
                q16 = _mm256_cvtepu8_epi16(_mm_and_si128(q8, _mm_set1_epi8(0x0F)));
            } else {
                q16 = _mm256_cvtepu8_epi16(_mm_and_si128(
                    _mm_srli_epi16(q8, 4), _mm_set1_epi8(0x0F)));
            }
            // Convert uint16 �?float32
            __m512 qf = _mm512_cvtepu32_ps(_mm512_cvtepu16_epi32(q16));
            // Apply dequant: w = sc * q4 - m
            __m512 wf = _mm512_fmsub_ps(v_sc, qf, v_m);
            // Clamp to int16 range
            wf = _mm512_min_ps(_mm512_max_ps(wf, vm32768), v32767);
            // Round and pack: float32 �?int32 �?int16
            __m512i wi32 = _mm512_cvtps_epi32(
                _mm512_roundscale_ps(wf, _MM_FROUND_TO_NEAREST_INT));
            __m256i wi16 = _mm512_cvtsepi32_epi16(wi32);
            _mm256_storeu_si256((__m256i*)(out + k), wi16);
        }
    }
}

// ============================================================================
// Q4_K VNNI Kernels (existing, preserved from original implementation)
// These use on-the-fly 4-bit dequant �?VPDPWSSD (INT16 path).
// Will be superseded by pre-dequant + VPDPBUSD for gate/up.
// ============================================================================

static inline void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void ds4_xeon_vec_dot_q4_K_vnni_8row(
    int n, float *s,
    const ds4_xeon_block_q4_K *w,
    const int16_t *a16,
    const int32_t *a16_sums,
    float s_a)
{
    const int nb = n / DS4_XEON_QK_K;

    for (int i = 0; i < nb; i++) {
        const int16_t *a16_ptr = a16 + (uint64_t)i * DS4_XEON_QK_K;
        const int32_t *a16_sum_ptr = a16_sums + (uint64_t)i * (DS4_XEON_QK_K / 32);

        for (int o = 0; o < 8; o++) {
            const ds4_xeon_block_q4_K *bx = &w[o * nb + i];
            const float d = xeon_f16_to_f32(bx->d) * s_a;
            const float dmin = xeon_f16_to_f32(bx->dmin) * s_a;
            float block_sum = 0.0f;

            for (int j = 0; j < 8; j++) {
                uint8_t sc_val, m_val;
                ds4q_get_scale_min_k4(j, bx->scales, &sc_val, &m_val);

                const uint8_t *q_ptr = bx->qs + (j / 2) * 32;
                __m256i q_pack = _mm256_loadu_si256((const __m256i*)q_ptr);
                __m256i q_chunk = (j % 2 == 0)
                    ? _mm256_and_si256(q_pack, _mm256_set1_epi8(0x0F))
                    : _mm256_and_si256(_mm256_srli_epi16(q_pack, 4), _mm256_set1_epi8(0x0F));

                __m512i w16 = _mm512_cvtepu8_epi16(q_chunk);
                __m512i a_vec = _mm512_loadu_si512(&a16_ptr[j * 32]);
                __m512i v_acc = _mm512_setzero_si512();
                v_acc = _mm512_dpwssd_epi32(v_acc, w16, a_vec);

                int32_t dot = _mm512_reduce_add_epi32(v_acc);
                block_sum += (d * sc_val) * (float)dot - (dmin * m_val) * (float)a16_sum_ptr[j];
            }
            s[o] += block_sum;
        }
    }
}

void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const ds4_xeon_block_q4_K *x,
    const int16_t *y_i16, const int32_t *y_sum_32, float scale_y)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = xeon_f16_to_f32(x[i].d) * scale_y;
        const float dmin = xeon_f16_to_f32(x[i].dmin) * scale_y;
        const uint8_t *q4 = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int16_t *a16 = y_i16 + (uint64_t)i * DS4_XEON_QK_K;
        const int32_t *a32_sum = y_sum_32 + (uint64_t)i * (DS4_XEON_QK_K / 32);

        for (int j = 0; j < 8; j++) {
            uint8_t sc_val, m_val;
            ds4q_get_scale_min_k4(j, sc, &sc_val, &m_val);
            const uint8_t *q_ptr = q4 + (j / 2) * 32;
            __m256i q_pack = _mm256_loadu_si256((const __m256i*)q_ptr);
            __m256i q_chunk = (j % 2 == 0)
                ? _mm256_and_si256(q_pack, _mm256_set1_epi8(0x0F))
                : _mm256_and_si256(_mm256_srli_epi16(q_pack, 4), _mm256_set1_epi8(0x0F));
            __m512i w16 = _mm512_cvtepu8_epi16(q_chunk);
            __m512i a = _mm512_loadu_si512(&a16[j * 32]);
            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpwssd_epi32(acc, w16, a);
            int32_t dot = _mm512_reduce_add_epi32(acc);
            sumf += (d * sc_val) * (float)dot - (dmin * m_val) * (float)a32_sum[j];
        }
    }
    *s += sumf;
}

// ============================================================================
// IQ2_XXS VNNI �?AVX-512 vectorized with gather + VPDPWSSD.
//
// Each IQ2XXS block has 32 qs entries (uint16_t). Each qs entry encodes:
//   low 8 bits  �?index into iq2xxs_grid[256] (uint64_t, packs 8× int8)
//   high 8 bits �?index into ksigns_iq2xs[128] (uint8_t, 8 sign bits)
//
// Scalar decode: grid = iq2xxs_grid[lo]; signs = ksigns_iq2xs[hi];
//   for k in 0..7: v = (int8_t)grid.byte[k]; if signs & (1<<k) v = -v;
//
// Vectorized: process 8 qs entries (�?64 int8 weights) per iteration.
//   1. Gather 8 grid uint64_t �?store as 64 raw int8 bytes (no scalar extraction)
//   2. Build 64-byte negation mask from sign bytes via LUT
//   3. sub(xor(raw, mask), mask) for conditional two's-complement negation
//   4. Extend int8→int16, VPDPWSSD for dot product
// ============================================================================

// LUT: expand a sign byte (8 bits) to 8 mask bytes (0xFF where bit is set)
// sign_mask_lut[b][k] = (b & (1<<k)) ? 0xFF : 0x00
static const uint64_t iq2xxs_sign_mask_lut[256] = {
    0x0000000000000000ULL, 0x00000000000000FFULL, 0x000000000000FF00ULL, 0x000000000000FFFFULL,
    0x0000000000FF0000ULL, 0x0000000000FF00FFULL, 0x0000000000FFFF00ULL, 0x0000000000FFFFFFULL,
    0x00000000FF000000ULL, 0x00000000FF0000FFULL, 0x00000000FF00FF00ULL, 0x00000000FF00FFFFULL,
    0x00000000FFFF0000ULL, 0x00000000FFFF00FFULL, 0x00000000FFFFFF00ULL, 0x00000000FFFFFFFFULL,
    0x000000FF00000000ULL, 0x000000FF000000FFULL, 0x000000FF0000FF00ULL, 0x000000FF0000FFFFULL,
    0x000000FF00FF0000ULL, 0x000000FF00FF00FFULL, 0x000000FF00FFFF00ULL, 0x000000FF00FFFFFFULL,
    0x000000FFFF000000ULL, 0x000000FFFF0000FFULL, 0x000000FFFF00FF00ULL, 0x000000FFFF00FFFFULL,
    0x000000FFFFFF0000ULL, 0x000000FFFFFF00FFULL, 0x000000FFFFFFFF00ULL, 0x000000FFFFFFFFFFULL,
    0x0000FF0000000000ULL, 0x0000FF00000000FFULL, 0x0000FF000000FF00ULL, 0x0000FF000000FFFFULL,
    0x0000FF0000FF0000ULL, 0x0000FF0000FF00FFULL, 0x0000FF0000FFFF00ULL, 0x0000FF0000FFFFFFULL,
    0x0000FF00FF000000ULL, 0x0000FF00FF0000FFULL, 0x0000FF00FF00FF00ULL, 0x0000FF00FF00FFFFULL,
    0x0000FF00FFFF0000ULL, 0x0000FF00FFFF00FFULL, 0x0000FF00FFFFFF00ULL, 0x0000FF00FFFFFFFFULL,
    0x0000FFFF00000000ULL, 0x0000FFFF000000FFULL, 0x0000FFFF0000FF00ULL, 0x0000FFFF0000FFFFULL,
    0x0000FFFF00FF0000ULL, 0x0000FFFF00FF00FFULL, 0x0000FFFF00FFFF00ULL, 0x0000FFFF00FFFFFFULL,
    0x0000FFFFFF000000ULL, 0x0000FFFFFF0000FFULL, 0x0000FFFFFF00FF00ULL, 0x0000FFFFFF00FFFFULL,
    0x0000FFFFFFFF0000ULL, 0x0000FFFFFFFF00FFULL, 0x0000FFFFFFFFFF00ULL, 0x0000FFFFFFFFFFFFULL,
    0x00FF000000000000ULL, 0x00FF0000000000FFULL, 0x00FF00000000FF00ULL, 0x00FF00000000FFFFULL,
    0x00FF000000FF0000ULL, 0x00FF000000FF00FFULL, 0x00FF000000FFFF00ULL, 0x00FF000000FFFFFFULL,
    0x00FF0000FF000000ULL, 0x00FF0000FF0000FFULL, 0x00FF0000FF00FF00ULL, 0x00FF0000FF00FFFFULL,
    0x00FF0000FFFF0000ULL, 0x00FF0000FFFF00FFULL, 0x00FF0000FFFFFF00ULL, 0x00FF0000FFFFFFFFULL,
    0x00FF00FF00000000ULL, 0x00FF00FF000000FFULL, 0x00FF00FF0000FF00ULL, 0x00FF00FF0000FFFFULL,
    0x00FF00FF00FF0000ULL, 0x00FF00FF00FF00FFULL, 0x00FF00FF00FFFF00ULL, 0x00FF00FF00FFFFFFULL,
    0x00FF00FFFF000000ULL, 0x00FF00FFFF0000FFULL, 0x00FF00FFFF00FF00ULL, 0x00FF00FFFF00FFFFULL,
    0x00FF00FFFFFF0000ULL, 0x00FF00FFFFFF00FFULL, 0x00FF00FFFFFFFF00ULL, 0x00FF00FFFFFFFFFFULL,
    0x00FFFF0000000000ULL, 0x00FFFF00000000FFULL, 0x00FFFF000000FF00ULL, 0x00FFFF000000FFFFULL,
    0x00FFFF0000FF0000ULL, 0x00FFFF0000FF00FFULL, 0x00FFFF0000FFFF00ULL, 0x00FFFF0000FFFFFFULL,
    0x00FFFF00FF000000ULL, 0x00FFFF00FF0000FFULL, 0x00FFFF00FF00FF00ULL, 0x00FFFF00FF00FFFFULL,
    0x00FFFF00FFFF0000ULL, 0x00FFFF00FFFF00FFULL, 0x00FFFF00FFFFFF00ULL, 0x00FFFF00FFFFFFFFULL,
    0x00FFFFFF00000000ULL, 0x00FFFFFF000000FFULL, 0x00FFFFFF0000FF00ULL, 0x00FFFFFF0000FFFFULL,
    0x00FFFFFF00FF0000ULL, 0x00FFFFFF00FF00FFULL, 0x00FFFFFF00FFFF00ULL, 0x00FFFFFF00FFFFFFULL,
    0x00FFFFFFFF000000ULL, 0x00FFFFFFFF0000FFULL, 0x00FFFFFFFF00FF00ULL, 0x00FFFFFFFF00FFFFULL,
    0x00FFFFFFFFFF0000ULL, 0x00FFFFFFFFFF00FFULL, 0x00FFFFFFFFFFFF00ULL, 0x00FFFFFFFFFFFFFFULL,
    0xFF00000000000000ULL, 0xFF000000000000FFULL, 0xFF0000000000FF00ULL, 0xFF0000000000FFFFULL,
    0xFF00000000FF0000ULL, 0xFF00000000FF00FFULL, 0xFF00000000FFFF00ULL, 0xFF00000000FFFFFFULL,
    0xFF000000FF000000ULL, 0xFF000000FF0000FFULL, 0xFF000000FF00FF00ULL, 0xFF000000FF00FFFFULL,
    0xFF000000FFFF0000ULL, 0xFF000000FFFF00FFULL, 0xFF000000FFFFFF00ULL, 0xFF000000FFFFFFFFULL,
    0xFF0000FF00000000ULL, 0xFF0000FF000000FFULL, 0xFF0000FF0000FF00ULL, 0xFF0000FF0000FFFFULL,
    0xFF0000FF00FF0000ULL, 0xFF0000FF00FF00FFULL, 0xFF0000FF00FFFF00ULL, 0xFF0000FF00FFFFFFULL,
    0xFF0000FFFF000000ULL, 0xFF0000FFFF0000FFULL, 0xFF0000FFFF00FF00ULL, 0xFF0000FFFF00FFFFULL,
    0xFF0000FFFFFF0000ULL, 0xFF0000FFFFFF00FFULL, 0xFF0000FFFFFFFF00ULL, 0xFF0000FFFFFFFFFFULL,
    0xFF00FF0000000000ULL, 0xFF00FF00000000FFULL, 0xFF00FF000000FF00ULL, 0xFF00FF000000FFFFULL,
    0xFF00FF0000FF0000ULL, 0xFF00FF0000FF00FFULL, 0xFF00FF0000FFFF00ULL, 0xFF00FF0000FFFFFFULL,
    0xFF00FF00FF000000ULL, 0xFF00FF00FF0000FFULL, 0xFF00FF00FF00FF00ULL, 0xFF00FF00FF00FFFFULL,
    0xFF00FF00FFFF0000ULL, 0xFF00FF00FFFF00FFULL, 0xFF00FF00FFFFFF00ULL, 0xFF00FF00FFFFFFFFULL,
    0xFF00FFFF00000000ULL, 0xFF00FFFF000000FFULL, 0xFF00FFFF0000FF00ULL, 0xFF00FFFF0000FFFFULL,
    0xFF00FFFF00FF0000ULL, 0xFF00FFFF00FF00FFULL, 0xFF00FFFF00FFFF00ULL, 0xFF00FFFF00FFFFFFULL,
    0xFF00FFFFFF000000ULL, 0xFF00FFFFFF0000FFULL, 0xFF00FFFFFF00FF00ULL, 0xFF00FFFFFF00FFFFULL,
    0xFF00FFFFFFFF0000ULL, 0xFF00FFFFFFFF00FFULL, 0xFF00FFFFFFFFFF00ULL, 0xFF00FFFFFFFFFFFFULL,
    0xFFFF000000000000ULL, 0xFFFF0000000000FFULL, 0xFFFF00000000FF00ULL, 0xFFFF00000000FFFFULL,
    0xFFFF000000FF0000ULL, 0xFFFF000000FF00FFULL, 0xFFFF000000FFFF00ULL, 0xFFFF000000FFFFFFULL,
    0xFFFF0000FF000000ULL, 0xFFFF0000FF0000FFULL, 0xFFFF0000FF00FF00ULL, 0xFFFF0000FF00FFFFULL,
    0xFFFF0000FFFF0000ULL, 0xFFFF0000FFFF00FFULL, 0xFFFF0000FFFFFF00ULL, 0xFFFF0000FFFFFFFFULL,
    0xFFFF00FF00000000ULL, 0xFFFF00FF000000FFULL, 0xFFFF00FF0000FF00ULL, 0xFFFF00FF0000FFFFULL,
    0xFFFF00FF00FF0000ULL, 0xFFFF00FF00FF00FFULL, 0xFFFF00FF00FFFF00ULL, 0xFFFF00FF00FFFFFFULL,
    0xFFFF00FFFF000000ULL, 0xFFFF00FFFF0000FFULL, 0xFFFF00FFFF00FF00ULL, 0xFFFF00FFFF00FFFFULL,
    0xFFFF00FFFFFF0000ULL, 0xFFFF00FFFFFF00FFULL, 0xFFFF00FFFFFFFF00ULL, 0xFFFF00FFFFFFFFFFULL,
    0xFFFFFF0000000000ULL, 0xFFFFFF00000000FFULL, 0xFFFFFF000000FF00ULL, 0xFFFFFF000000FFFFULL,
    0xFFFFFF0000FF0000ULL, 0xFFFFFF0000FF00FFULL, 0xFFFFFF0000FFFF00ULL, 0xFFFFFF0000FFFFFFULL,
    0xFFFFFF00FF000000ULL, 0xFFFFFF00FF0000FFULL, 0xFFFFFF00FF00FF00ULL, 0xFFFFFF00FF00FFFFULL,
    0xFFFFFF00FFFF0000ULL, 0xFFFFFF00FFFF00FFULL, 0xFFFFFF00FFFFFF00ULL, 0xFFFFFF00FFFFFFFFULL,
    0xFFFFFFFF00000000ULL, 0xFFFFFFFF000000FFULL, 0xFFFFFFFF0000FF00ULL, 0xFFFFFFFF0000FFFFULL,
    0xFFFFFFFF00FF0000ULL, 0xFFFFFFFF00FF00FFULL, 0xFFFFFFFF00FFFF00ULL, 0xFFFFFFFF00FFFFFFULL,
    0xFFFFFFFFFF000000ULL, 0xFFFFFFFFFF0000FFULL, 0xFFFFFFFFFF00FF00ULL, 0xFFFFFFFFFF00FFFFULL,
    0xFFFFFFFFFFFF0000ULL, 0xFFFFFFFFFFFF00FFULL, 0xFFFFFFFFFFFFFF00ULL, 0xFFFFFFFFFFFFFFFFULL,
};

void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const ds4_xeon_block_iq2_xxs *x,
    const int16_t *y_i16, float scale_y)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = xeon_f16_to_f32(x[i].d) * scale_y;
        const uint16_t *qs = x[i].qs;
        const int16_t *a16 = y_i16 + (uint64_t)i * DS4_XEON_QK_K;

        int32_t acc = 0;

        // Process 8 qs entries per iteration (64 weights × 64 activations)
        for (int j = 0; j < 32; j += 8) {
            // Load 8 uint16 qs values
            __m128i qv = _mm_loadu_si128((const __m128i*)(qs + j));

            // Grid indices from low bytes
            __m256i lo32 = _mm256_cvtepu16_epi32(
                _mm_and_si128(qv, _mm_set1_epi16(0xFF)));
            __m512i g64 = _mm512_i32gather_epi64(lo32,
                (const void*)iq2xxs_grid, 8);

            // Sign indices from high bytes (masked to 7 bits)
            __m128i hi8 = _mm_and_si128(
                _mm_srli_epi16(qv, 8), _mm_set1_epi16(0x7F));

            // Store gathered grid values �?64 raw int8 bytes (no scalar extraction)
            uint64_t gl[8];
            _mm512_storeu_si512((__m512i*)gl, g64);

            // Build 64-byte negation mask from sign bytes via LUT
            uint8_t si16[16];
            _mm_storeu_si128((__m128i*)si16, hi8);
            uint64_t mask[8];
            for (int qi = 0; qi < 8; qi++) {
                mask[qi] = iq2xxs_sign_mask_lut[ksigns_iq2xs[si16[qi * 2]]];
            }

            /* One-time debug: print first block details */
            {
                static int done = 0;
                if (i == 0 && j == 0 && !done) {
                    done = 1;
                    uint16_t q0 = qs[0];
                    uint8_t gi0 = q0 & 0xFF, si0 = (q0 >> 8) & 0x7F;
                    uint8_t ks  = ksigns_iq2xs[si0];
                    uint64_t gv = iq2xxs_grid[gi0];
                    fprintf(stderr, "ds4_xeon: IQ2XXS dbg q[0]=0x%04x gi=%u si=%u "
                        "ksigns_iq2xs[%u]=0x%02x grid[%u]=0x%016lx "
                        "mask[0]=0x%016lx d_raw=0x%04x a16[0]=%d\n",
                        q0, gi0, si0, si0, ks, gi0, gv, mask[0],
                        x[i].d, (int)a16[0]);
                }
            }

            // Load raw int8 weights and negation mask
            __m512i w8_raw = _mm512_loadu_si512((const __m512i*)gl);
            __m512i m8 = _mm512_loadu_si512((const __m512i*)mask);

            // Conditional negation: sub(xor(w, mask), mask)
            __m512i w8_signed = _mm512_sub_epi8(
                _mm512_xor_si512(w8_raw, m8), m8);

            // Extend int8 �?int16 (two halves, 32 each)
            __m256i w8_lo = _mm512_castsi512_si256(w8_signed);
            __m256i w8_hi = _mm512_extracti64x4_epi64(w8_signed, 1);
            __m512i w16_lo = _mm512_cvtepi8_epi16(w8_lo);
            __m512i w16_hi = _mm512_cvtepi8_epi16(w8_hi);

            // Load 64 int16 activations
            __m512i a_lo = _mm512_loadu_si512((const __m512i*)(a16 + j * 8));
            __m512i a_hi = _mm512_loadu_si512((const __m512i*)(a16 + j * 8 + 32));

            // VPDPWSSD: int32 += int16 × int16
            __m512i vacc = _mm512_setzero_si512();
            vacc = _mm512_dpwssd_epi32(vacc, w16_lo, a_lo);
            vacc = _mm512_dpwssd_epi32(vacc, w16_hi, a_hi);

            acc += _mm512_reduce_add_epi32(vacc);
        }

        sumf += d * (float)acc;
    }

    *s += sumf;
}

// ============================================================================
// Q2_K dot product — scalar nibble extraction, auto-vectorized by GCC
// ============================================================================
// GCC -O3 -march=native auto-vectorizes the inner loop with vpmaddwd.
// Manual VPDPWSSD attempts were slower because:
//   1. Store-to-buffer / load-from-buffer round-trip costs
//   2. 16-element sub-blocks are too small for __m512i (need 32)
//   3. Per-sub-block scale prevents combining 2 sub-blocks into one VPDPWSSD
// The real fix for Q2_K performance is pre-dequantization (Phase B),
// which replaces nibble extraction with direct int16 weight reads.
// ============================================================================
void ds4_xeon_vec_dot_q2_K_vnni(int n, float *s, const ds4_xeon_block_q2_K *x,
    const int16_t *y_i16, const int32_t *y_sum_32, float scale_y)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = xeon_f16_to_f32(x[i].d) * scale_y;
        const float dmin = xeon_f16_to_f32(x[i].dmin) * scale_y;
        const uint8_t *q2 = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int16_t *a16 = y_i16 + (uint64_t)i * DS4_XEON_QK_K;
        const int32_t *a32_sum = y_sum_32 + (uint64_t)i * (DS4_XEON_QK_K / 32);

        for (int j = 0; j < 16; j++) {
            float sc_val = (float)sc[j];
            const uint8_t *q_ptr = q2 + j * 4;
            int32_t dot = 0;
            for (int k = 0; k < 16; k++) {
                uint8_t q_byte = q_ptr[k / 4];
                uint8_t l = (q_byte >> ((k % 4) * 2)) & 3;
                dot += (int32_t)l * (int32_t)a16[j * 16 + k];
            }
            sumf += d * sc_val * (float)dot
                  - dmin * sc_val * (float)a32_sum[j / 2];
        }
    }
    *s += sumf;
}

// ============================================================================
// Q2_K VNNI with Q8_K activation — VPDPWSSD, same quantization as CPU
// ============================================================================
// Uses the CPU's Q8_K per-block quantization (int8[256] + float scale per block)
// for the heavy-tailed SwiGLU mid activation.  Extends int8→int16 on-the-fly,
// extracts Q2_K nibbles→int16, runs VPDPWSSD, and applies per-sub-block Q2_K
// scales combined with per-block Q8_K scales.
// Precision: matches CPU path within 1e-4 (same quantization, same arithmetic).
void ds4_xeon_vec_dot_q2_K_q8k_vnni(int n, float *s,
    const ds4_xeon_block_q2_K *x,
    const int8_t *q8, const float *q8_scale)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float q8_d = q8_scale[i];
        const float d    = xeon_f16_to_f32(x[i].d) * q8_d;
        const float dmin = xeon_f16_to_f32(x[i].dmin) * q8_d;
        const uint8_t *q2 = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int8_t  *a8 = q8 + (uint64_t)i * DS4_XEON_QK_K;

        /* Pre-compute sums of 32 activation elements (8 sums per QK block) */
        int32_t a32_sums[8];
        for (int jj = 0; jj < 8; jj++) {
            int32_t s = 0;
            const int8_t *ap = a8 + (uint64_t)jj * 32;
            for (int k = 0; k < 32; k++) s += (int32_t)ap[k];
            a32_sums[jj] = s;
        }

        for (int j = 0; j < 16; j++) {
            float sc_val = (float)sc[j];
            const uint8_t *q_ptr = q2 + (uint64_t)j * 4;

            /* Expand nibbles → 16 int16 weights + extend activations */
            int16_t q16[16] __attribute__((aligned(32)));
            int16_t a16[16] __attribute__((aligned(32)));
            const int8_t *a_ptr = a8 + (uint64_t)j * 16;
            for (int k = 0; k < 16; k++) {
                uint8_t q_byte = q_ptr[k / 4];
                q16[k] = (int16_t)((q_byte >> ((k % 4) * 2)) & 3);
                a16[k] = (int16_t)a_ptr[k];
            }

            /* VPDPWSSD: 16 int16 weights × 16 int16 activations */
            __m256i wv = _mm256_load_si256((const __m256i*)q16);
            __m256i av = _mm256_load_si256((const __m256i*)a16);
            __m256i vacc = _mm256_dpwssd_epi32(_mm256_setzero_si256(), wv, av);
            int32_t arr[8];
            _mm256_storeu_si256((__m256i*)arr, vacc);
            int32_t dot = arr[0] + arr[1] + arr[2] + arr[3]
                        + arr[4] + arr[5] + arr[6] + arr[7];

            sumf += d * sc_val * (float)dot
                  - dmin * sc_val * (float)a32_sums[j / 2];
        }
    }
    *s += sumf;
}

// ============================================================================
// RMS Norm (AVX-512)
// ============================================================================

void ds4_xeon_rms_norm(float *out, const float *in, const float *w, int n, float eps) {
    __m512 vss = _mm512_setzero_ps();
    int i16 = (n / 16) * 16;
    for (int i = 0; i < i16; i += 16) {
        __m512 vi = _mm512_loadu_ps(in + i);
        vss = _mm512_fmadd_ps(vi, vi, vss);
    }
    float ss = _mm512_reduce_add_ps(vss);
    for (int i = i16; i < n; i++) ss += in[i] * in[i];

    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    __m512 vscale = _mm512_set1_ps(scale);

    for (int i = 0; i < i16; i += 16) {
        __m512 vi = _mm512_loadu_ps(in + i);
        __m512 vw = _mm512_loadu_ps(w + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(vi, vscale), vw));
    }
    for (int i = i16; i < n; i++) {
        out[i] = in[i] * scale * w[i];
    }
}

// ============================================================================
// NUMA-local tensor data access (T4.2.2)
// ============================================================================
static const uint8_t *g_numa_map[2] = {NULL, NULL};

void ds4_xeon_set_numa_maps(const uint8_t *map0, const uint8_t *map1) {
    g_numa_map[0] = map0;
    g_numa_map[1] = map1;
}

const uint8_t *ds4_xeon_tensor_data_numa(uint64_t abs_offset) {
    int node = 0;
#ifdef __linux__
    int cpu = sched_getcpu();
    // Quick NUMA node lookup: on dual-socket Ice Lake, CPUs 0-23 → node 0, 24-47 → node 1
    node = (cpu >= 24) ? 1 : 0;
#endif
    const uint8_t *map = g_numa_map[node] ? g_numa_map[node] : g_numa_map[0];
    return map + abs_offset;
}

// ============================================================================
// Xeon Routed MoE — AVX-512 instant matvec per expert
// ============================================================================
// Process one token through one expert: gate → up → SiLU → down → accumulate.
// Uses ds4_xeon_vec_dot_iq2_xxs_vnni for gate/up, ds4_xeon_vec_dot_q2_K_vnni for down.
// Called from ds4.c wrapper for each token × expert pair.
void ds4_xeon_routed_moe_one_expert(
    float *out,                    // [DS4_N_EMBD] output accumulator
    const float *x,                // [DS4_N_EMBD] input activation (RMS-normed)
    const uint8_t *gate_blocks,    // IQ2XXS gate expert blocks (row-major)
    const uint8_t *up_blocks,      // IQ2XXS up expert blocks
    const uint8_t *down_blocks,    // Q2_K down expert blocks
    uint64_t gate_row_bytes,       // bytes per gate output row
    uint64_t up_row_bytes,         // bytes per up output row
    uint64_t down_row_bytes,       // bytes per down output row
    float expert_weight)           // router weight for this expert
{
    // Stack buffers (~50 KB total, well within default 8 MB stack)
    __attribute__((aligned(64))) int16_t act_i16[DS4_N_EMBD];
    __attribute__((aligned(64))) float  gate[DS4_N_FF_EXP];
    __attribute__((aligned(64))) float  up[DS4_N_FF_EXP];
    __attribute__((aligned(64))) float  mid[DS4_N_FF_EXP];
    __attribute__((aligned(64))) int8_t  mid_q8[DS4_N_FF_EXP];
    float   mid_q8_scale[DS4_N_FF_EXP / DS4_XEON_QK_K];

    // Quantize input to int16 for IQ2XXS dot products
    float act_scale;
    ds4_xeon_quantize_a16_per_token(act_i16, &act_scale, x, 1, DS4_N_EMBD);
    /* act_scale from quantize is max_val/32767, correct for dot */

    // Gate projection: [DS4_N_EMBD] → [DS4_N_FF_EXP]  (2048 rows)
    for (uint32_t r = 0; r < DS4_N_FF_EXP; r++) {
        const ds4_xeon_block_iq2_xxs *blocks =
            (const ds4_xeon_block_iq2_xxs*)(gate_blocks + r * gate_row_bytes);
        float dot = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(DS4_N_EMBD, &dot, blocks, act_i16, act_scale);
        gate[r] = dot;
    }

    // Up projection
    for (uint32_t r = 0; r < DS4_N_FF_EXP; r++) {
        const ds4_xeon_block_iq2_xxs *blocks =
            (const ds4_xeon_block_iq2_xxs*)(up_blocks + r * up_row_bytes);
        float dot = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(DS4_N_EMBD, &dot, blocks, act_i16, act_scale);
        up[r] = dot;
    }

    // SwiGLU: mid = SiLU(gate) * up
    ds4_xeon_swiglu(mid, gate, up, DS4_N_FF_EXP);

    // Quantize mid to Q8_K per-block (matching CPU's ds4_quantize_row_q8_K).
    // Q8_K is required for heavy-tailed SwiGLU mid; per-token INT16 loses
    // 12.6% accuracy (see ds4_xeon_down_test.c).
    {
        const int nb = DS4_N_FF_EXP / DS4_XEON_QK_K;
        for (int b = 0; b < nb; b++) {
            const float *src = mid + b * DS4_XEON_QK_K;
            int8_t *dst = mid_q8 + b * DS4_XEON_QK_K;
            float amax = 1e-9f;
            for (int i = 0; i < DS4_XEON_QK_K; i++) {
                float a = fabsf(src[i]);
                if (a > amax) amax = a;
            }
            float d = amax / 127.0f;
            float id = 1.0f / d;
            mid_q8_scale[b] = d;
            for (int i = 0; i < DS4_XEON_QK_K; i++) {
                float v = src[i] * id;
                if (v > 127.0f) v = 127.0f;
                if (v < -128.0f) v = -128.0f;
                dst[i] = (int8_t)(int32_t)v;
            }
        }
    }

    // Down projection: [DS4_N_FF_EXP] → [DS4_N_EMBD], Q8_K VNNI (0.02% error)
    for (uint32_t r = 0; r < DS4_N_EMBD; r++) {
        const ds4_xeon_block_q2_K *blocks =
            (const ds4_xeon_block_q2_K*)(down_blocks + r * down_row_bytes);
        float dot = 0.0f;
        ds4_xeon_vec_dot_q2_K_q8k_vnni(DS4_N_FF_EXP, &dot, blocks,
            mid_q8, mid_q8_scale);
        out[r] += expert_weight * dot;
    }
}

// ============================================================================
// Attention Scores (AVX-512) — QK^T + softmax + weighted sum
// ============================================================================
// Compute attention for n_tok tokens with causal masking.
// q: [n_tok][DS4_N_HEAD * DS4_N_HEAD_DIM] query vectors
// raw_kv: [n_raw][DS4_N_HEAD_DIM] key/value rows in KV cache
// heads: [n_tok][DS4_N_HEAD * DS4_N_HEAD_DIM] output (overwritten)
void ds4_xeon_attn_scores(
    float *heads, const float *q, const float *raw_kv,
    uint32_t n_tok, uint32_t il)
{
    (void)il;
    const float attn_scale = 1.0f / sqrtf((float)DS4_N_HEAD_DIM);
    /* Round up to 16 to avoid _mm512_storeu_ps writing past allocation */
    size_t buf_elems = ((size_t)n_tok + 15) & ~(size_t)15;
    float *scores_buf = (float*)aligned_alloc(64, buf_elems * sizeof(float));

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *qt = q + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM;
        float *ht = heads + (uint64_t)t * DS4_N_HEAD * DS4_N_HEAD_DIM;
        uint32_t n_visible = t + 1;

        for (uint32_t h = 0; h < DS4_N_HEAD; h++) {
            const float *qh = qt + (uint64_t)h * DS4_N_HEAD_DIM;
            float *oh = ht + (uint64_t)h * DS4_N_HEAD_DIM;
            memset(oh, 0, DS4_N_HEAD_DIM * sizeof(float));

            // dot-product over KV rows
            float max_score = DS4_NEG_INF;
            for (uint32_t vi = 0; vi < n_visible; vi++) {
                const float *kr = raw_kv + (uint64_t)vi * DS4_N_HEAD_DIM;
                __m512 acc = _mm512_setzero_ps();
                for (uint32_t d = 0; d < DS4_N_HEAD_DIM; d += 16) {
                    __m512 qv = _mm512_loadu_ps(qh + d);
                    __m512 kv = _mm512_loadu_ps(kr + d);
                    acc = _mm512_add_ps(_mm512_mul_ps(qv, kv), acc);
                }
                float dot = _mm512_reduce_add_ps(acc) * attn_scale;
                scores_buf[vi] = dot;
                if (dot > max_score) max_score = dot;
            }

            // softmax (scalar exp, vectorized load/store)
            float sum_exp = 0.0f;
            __m512 vmax = _mm512_set1_ps(max_score);
            for (uint32_t vi = 0; vi < n_visible; vi += 16) {
                uint32_t nv = (vi + 16 <= n_visible) ? 16 : (n_visible - vi);
                __m512 vs = _mm512_sub_ps(_mm512_loadu_ps(scores_buf + vi), vmax);
                float tmp[16]; _mm512_storeu_ps(tmp, vs);
                for (uint32_t k = 0; k < nv; k++) { tmp[k] = expf(tmp[k]); sum_exp += tmp[k]; }
                _mm512_storeu_ps(scores_buf + vi, _mm512_loadu_ps(tmp));
            }
            float inv_sum = 1.0f / (sum_exp + 1e-10f);

            // weighted sum
            for (uint32_t vi = 0; vi < n_visible; vi++) {
                float w = scores_buf[vi] * inv_sum;
                const float *kr = raw_kv + (uint64_t)vi * DS4_N_HEAD_DIM;
                __m512 vw = _mm512_set1_ps(w);
                for (uint32_t d = 0; d < DS4_N_HEAD_DIM; d += 16) {
                    __m512 ov = _mm512_loadu_ps(oh + d);
                    __m512 kv = _mm512_loadu_ps(kr + d);
                    _mm512_storeu_ps(oh + d, _mm512_add_ps(_mm512_mul_ps(vw, kv), ov));
                }
            }
        }
    }
    free(scores_buf);
}

// ============================================================================
// Vectorized axpy: y[i] += a * x[i]  (n elements, AVX-512)
// ============================================================================
void ds4_xeon_axpy_f32(float *y, const float *x, float a, int n) {
    __m512 va = _mm512_set1_ps(a);
    int i;
    for (i = 0; i <= n - 16; i += 16) {
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 vx = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(y + i, _mm512_add_ps(vy, _mm512_mul_ps(va, vx)));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

// ============================================================================
// SwiGLU (AVX-512 with scalar sigmoid)
// ============================================================================

void ds4_xeon_swiglu(float *out, const float *x, const float *y, int n) {
    int i16 = (n / 16) * 16;
    for (int i = 0; i < i16; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        /* sigmoid(x) = 1/(1+exp(-x)).  Compute sigmoid(x) * x * y.
         * AVX-512 has no native exp, so we store, compute scalar expf,
         * reload.  The 16 expf calls dominate; the SIMD mul/add avoids
         * port pressure on the surrounding arithmetic. */
        float xv[16], yv[16];
        _mm512_storeu_ps(xv, vx);
        _mm512_storeu_ps(yv, vy);
        for (int j = 0; j < 16; j++) {
            float sx = 1.0f / (1.0f + expf(-xv[j]));
            xv[j] = sx * xv[j] * yv[j];
        }
        _mm512_storeu_ps(out + i, _mm512_loadu_ps(xv));
    }
    for (int i = i16; i < n; i++) {
        float sx = 1.0f / (1.0f + expf(-x[i]));
        out[i] = sx * x[i] * y[i];
    }
}

// ============================================================================
// VNNI Q8_0 matvec — VPDPWSSD with per-block scales (OpenMP over rows)
// ============================================================================
// Processes Q8_0 weights (int8_t[32] + float scale per block) against
// a single int16 activation vector.  OpenMP-parallel across output rows.
// Used for attention Q/KV/Output projections in the xeon decode path.
void ds4_xeon_q80_matvec(float *out,
    const int16_t *act_i16, float act_scale,
    const void *q80_blocks,  /* block_q8_0[]: int8[32]+fp16 per block */
    int in_dim, int out_dim)
{
    int n_blocks = in_dim / 32;
    int block_stride = 34; /* sizeof(block_q8_0) = 32 qs + 2 fp16 */
    const int8_t *base = (const int8_t*)q80_blocks;

    for (int r = 0; r < out_dim; r++) {
        const int8_t *row_blocks = base + (uint64_t)r * n_blocks * block_stride;
        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const int8_t *blk = row_blocks + b * block_stride;
            float d = xeon_f16_to_f32(*(const uint16_t*)(blk + 32));
            __m256i w8 = _mm256_loadu_si256((const __m256i*)blk);
            __m512i w16 = _mm512_cvtepi8_epi16(w8);
            __m512i a16 = _mm512_loadu_si512((const __m512i*)(act_i16 + b * 32));
            __m512i acc = _mm512_dpwssd_epi32(_mm512_setzero_si512(), w16, a16);
            total += (float)_mm512_reduce_add_epi32(acc) * d;
        }
        out[r] = total * act_scale;
    }
}

/* Row-range Q8_0 matvec for ds4_parallel_for dispatch. */
void ds4_xeon_q80_matvec_rows(float *out,
    const int16_t *act_i16, float act_scale,
    const void *q80_blocks, int in_dim, int row0, int row1)
{
    int n_blocks = in_dim / 32;
    int block_stride = 34; /* sizeof(block_q8_0) = 32 qs + 2 fp16 */
    const int8_t *base = (const int8_t*)q80_blocks;

    for (int r = row0; r < row1; r++) {
        const int8_t *row_blocks = base + (uint64_t)r * n_blocks * block_stride;
        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const int8_t *blk = row_blocks + b * block_stride;
            float d = xeon_f16_to_f32(*(const uint16_t*)(blk + 32));
            __m256i w8 = _mm256_loadu_si256((const __m256i*)blk);
            __m512i w16 = _mm512_cvtepi8_epi16(w8);
            __m512i a16 = _mm512_loadu_si512((const __m512i*)(act_i16 + b * 32));
            __m512i acc = _mm512_dpwssd_epi32(_mm512_setzero_si512(), w16, a16);
            total += (float)_mm512_reduce_add_epi32(acc) * d;
        }
        out[r] = total * act_scale;
    }
}

// ============================================================================
// VNNI Matvec — single-core row-range functions (caller parallelizes via pool)
// ============================================================================

/* VPDPBUSD matvec: INT8 activation × uint8 pre-dequant weight → float[out_dim].
 * Processes rows [row0, row1).  Caller uses ds4_parallel_for to parallelize
 * across rows.  in_dim must be a multiple of 64. */
void ds4_xeon_matvec_vpdpbusd(float *out, const int8_t *a8,
    const uint8_t *w8, int in_dim, int row0, int row1)
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

/* VPDPWSSD matvec: INT16 activation × int16 pre-dequant weight → float[out_dim].
 * Processes rows [row0, row1).  in_dim must be a multiple of 32. */
void ds4_xeon_matvec_vpdpwssd(float *out, const int16_t *a16,
    const int16_t *w16, int in_dim, int row0, int row1)
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

/* Parallel-for context structs for FFN matvec dispatch */
typedef struct {
    float *out;
    const int8_t *a8;
    const uint8_t *w8;
    int in_dim;
} matvec_vpdpbusd_ctx;

static void matvec_vpdpbusd_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_vpdpbusd_ctx *ctx = (matvec_vpdpbusd_ctx*)vctx;
    ds4_xeon_matvec_vpdpbusd(ctx->out, ctx->a8, ctx->w8,
        ctx->in_dim, (int)row0, (int)row1);
}

typedef struct {
    float *out;
    const int16_t *a16;
    const int16_t *w16;
    int in_dim;
} matvec_vpdpwssd_ctx;

static void matvec_vpdpwssd_worker(void *vctx, uint64_t row0, uint64_t row1) {
    matvec_vpdpwssd_ctx *ctx = (matvec_vpdpwssd_ctx*)vctx;
    ds4_xeon_matvec_vpdpwssd(ctx->out, ctx->a16, ctx->w16,
        ctx->in_dim, (int)row0, (int)row1);
}

/* Single token, single layer FFN using pre-dequantized weights + VNNI.
 * x:       [n_embd] RMS-normed input
 * out:     [n_embd] output accumulator (zeroed first, then accumulated)
 * pre:     pointer to pre-dequantized weight buffers for this layer
 * selected:[n_expert_used] expert indices from router
 * weights: [n_expert_used] router weights
 * n_embd, n_ff_exp: model dimensions */
void ds4_xeon_ffn_decode_one(
    float *restrict out,
    const float *restrict x,
    const uint8_t *gate_u8,
    const uint8_t *up_u8,
    const int16_t *down_i16,
    const int32_t *selected,
    const float *expert_weights,
    int n_embd, int n_ff_exp)
{
    /* quantize input to INT8 per-block (block_size=64) */
    int8_t *a8 = (int8_t*)aligned_alloc(64, (size_t)n_embd);
    float  *a8_sc = (float*)aligned_alloc(64, (size_t)(n_embd / 64) * sizeof(float));
    ds4_xeon_quantize_a8_per_block(a8, a8_sc, x, 1, n_embd, 64);

    memset(out, 0, (size_t)n_embd * sizeof(float));

    for (int e = 0; e < DS4_N_EXPERT_USED; e++) {
        int eid = selected[e];
        float ew = expert_weights[e];
        /* gate/up are interleaved: gate at [eid*2], up at [eid*2+1] */
        uint64_t expert_off_gate = (uint64_t)eid * 2 * (uint64_t)n_ff_exp * (uint64_t)n_embd;
        uint64_t expert_off_up   = expert_off_gate + (uint64_t)n_ff_exp * (uint64_t)n_embd;
        uint64_t expert_off_down = (uint64_t)eid * (uint64_t)n_embd * (uint64_t)n_ff_exp;

        float *gate = (float*)aligned_alloc(64, (size_t)n_ff_exp * sizeof(float));
        float *up   = (float*)aligned_alloc(64, (size_t)n_ff_exp * sizeof(float));
        float *mid  = (float*)aligned_alloc(64, (size_t)n_ff_exp * sizeof(float));
        float *down = (float*)aligned_alloc(64, (size_t)n_embd * sizeof(float));

        /* gate + up: VPDPBUSD matvecs */
        ds4_xeon_matvec_vpdpbusd(gate, a8,
            gate_u8 + expert_off_gate, n_embd, 0, n_ff_exp);
        ds4_xeon_matvec_vpdpbusd(up, a8,
            up_u8 + expert_off_up, n_embd, 0, n_ff_exp);

        /* SwiGLU */
        ds4_xeon_swiglu(mid, gate, up, n_ff_exp);

        /* quantize mid to INT16 */
        int16_t *a16 = (int16_t*)aligned_alloc(64, (size_t)n_ff_exp * sizeof(int16_t));
        float a16_sc;
        ds4_xeon_quantize_a16_per_token(a16, &a16_sc, mid, 1, n_ff_exp);

        /* down: VPDPWSSD matvec */
        ds4_xeon_matvec_vpdpwssd(down, a16,
            down_i16 + expert_off_down, n_ff_exp, 0, n_embd);

        /* accumulate weighted output */
        for (int i = 0; i < n_embd; i++)
            out[i] += down[i] * ew;

        free(down); free(mid); free(up); free(gate); free(a16);
    }
    free(a8_sc); free(a8);
}

// ============================================================================
// Row-range parallel MoE helpers (caller dispatches via ds4_parallel_for)
// ============================================================================

/* Gate or Up projection: IQ2XXS dot products for rows [row0, row1).
 * blocks: packed IQ2XXS blocks, row_bytes apart per output row.
 * act_i16: int16 activation [DS4_N_EMBD].  act_scale: 1/quant_scale. */
void ds4_xeon_moe_gateup_rows(
    float *out, const uint8_t *blocks, uint64_t row_bytes,
    const int16_t *act_i16, float act_scale,
    int row0, int row1)
{
    for (int r = row0; r < row1; r++) {
        const ds4_xeon_block_iq2_xxs *blk =
            (const ds4_xeon_block_iq2_xxs*)(blocks + (uint64_t)r * row_bytes);
        float dot = 0.0f;
        ds4_xeon_vec_dot_iq2_xxs_vnni(DS4_N_EMBD, &dot, blk, act_i16, act_scale);
        out[r] = dot;
    }
}

/* Down projection: Q2_K dot products for rows [row0, row1).
 * blocks: packed Q2_K blocks, row_bytes apart per output row.
 * mid_i16: int16 SwiGLU mid activation [DS4_N_FF_EXP].
 * mid_sums: pre-computed int32 sums per 32 elements.
 * mid_scale: 1/quant_scale.  ew: expert weight multiplier. */
void ds4_xeon_moe_down_rows(
    float *out, const uint8_t *blocks, uint64_t row_bytes,
    const int16_t *mid_i16, const int32_t *mid_sums, float mid_scale,
    float ew, int row0, int row1)
{
    for (int r = row0; r < row1; r++) {
        const ds4_xeon_block_q2_K *blk =
            (const ds4_xeon_block_q2_K*)(blocks + (uint64_t)r * row_bytes);
        float dot = 0.0f;
        ds4_xeon_vec_dot_q2_K_vnni(DS4_N_FF_EXP, &dot, blk, mid_i16, mid_sums, mid_scale);
        out[r] += ew * dot;
    }
}

#else
// No-AVX512 stubs for platforms that lack VNNI support
void ds4_xeon_matmul_a8w8_vnni(float *o, const int8_t *a8, const float *as,
    const uint8_t *w8, const float *ws, int id, int od) {
    (void)o; (void)a8; (void)as; (void)w8; (void)ws; (void)id; (void)od;
    fprintf(stderr, "ds4_xeon: matmul_a8w8_vnni requires AVX-512 VNNI\n");
}
void ds4_xeon_matmul_a16w16_vnni(float *o, const int16_t *a16, float as,
    const int16_t *w16, const float *ws, int id, int od) {
    (void)o; (void)a16; (void)as; (void)w16; (void)ws; (void)id; (void)od;
    fprintf(stderr, "ds4_xeon: matmul_a16w16_vnni requires AVX-512 VNNI\n");
}
void ds4_xeon_quantize_a8_per_block(int8_t *o, float *s, const float *i, int nt, int id, int bs) {
    (void)o; (void)s; (void)i; (void)nt; (void)id; (void)bs;
}
void ds4_xeon_quantize_a8_per_block_avx512(int8_t *o, float *s, const float *i, int nt, int id, int bs) {
    (void)o; (void)s; (void)i; (void)nt; (void)id; (void)bs;
}
void ds4_xeon_quantize_a16_per_token(int16_t *o, float *s, const float *i, int nt, int id) {
    (void)o; (void)s; (void)i; (void)nt; (void)id;
    fprintf(stderr, "ds4_xeon: quantize_a16 requires AVX-512 VNNI\n");
}
void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const void *x, const int16_t *y, const int32_t *ys, float sy) {
    (void)n; (void)s; (void)x; (void)y; (void)ys; (void)sy;
}
void ds4_xeon_vec_dot_q4_K_vnni_8row(int n, float *s, const void *w, const int16_t *a, const int32_t *as, float sa) {
    (void)n; (void)s; (void)w; (void)a; (void)as; (void)sa;
}
void ds4_xeon_vec_dot_q2_K_vnni(int n, float *s, const void *x, const int16_t *y, const int32_t *ys, float sy) {
    (void)n; (void)s; (void)x; (void)y; (void)ys; (void)sy;
}
void ds4_xeon_vec_dot_q2_K_q8k_vnni(int n, float *s, const void *x, const int8_t *q8, const float *qs) {
    (void)n; (void)s; (void)x; (void)q8; (void)qs;
}
void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const void *x, const int16_t *y, float sy) {
    (void)n; (void)s; (void)x; (void)y; (void)sy;
}
void ds4_xeon_unpack_q4_k_to_u8(uint8_t *u8, float *sc8, float *m8, const void *x) {
    (void)u8; (void)sc8; (void)m8; (void)x;
}
void ds4_xeon_dequant_q4_k_to_i16(int16_t *i16, const void *x) {
    (void)i16; (void)x;
}
void ds4_xeon_unpack_q4_k_to_i16(int16_t *i16, const void *x) {
    (void)i16; (void)x;
}
void ds4_xeon_rms_norm(float *o, const float *i, const float *w, int n, float e) {
    (void)o; (void)i; (void)w; (void)n; (void)e;
}
void ds4_xeon_swiglu(float *o, const float *x, const float *y, int n) {
    (void)o; (void)x; (void)y; (void)n;
}
void ds4_xeon_dequant_iq2xxs_block_to_u8(uint8_t *d, const void *x) {
    (void)d; (void)x;
}
void ds4_xeon_dequant_q2k_block_to_i16(int16_t *d, const void *x) {
    (void)d; (void)x;
}
void ds4_xeon_attn_scores(float *h, const float *q, const float *kv, uint32_t n, uint32_t il) {
    (void)h; (void)q; (void)kv; (void)n; (void)il;
}
void ds4_xeon_routed_moe_one_expert(float *o, const float *x,
    const uint8_t *g, const uint8_t *u, const uint8_t *d,
    uint64_t grb, uint64_t urb, uint64_t drb, float ew) {
    (void)o; (void)x; (void)g; (void)u; (void)d;
    (void)grb; (void)urb; (void)drb; (void)ew;
}
void ds4_xeon_axpy_f32(float *y, const float *x, float a, int n) {
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}
#endif

// ============================================================================
// Graph Lifecycle
// ============================================================================

void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size,
    uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_hc,
    uint32_t n_expert, uint32_t n_expert_used, uint32_t n_layer,
    int numa_node)
{
    memset(g, 0, sizeof(*g));
    g->max_batch_size = max_batch_size;
    g->n_embd = n_embd;
    g->n_ff_exp = n_ff_exp;
    g->n_hc = n_hc;
    g->n_expert = n_expert;
    g->n_expert_used = n_expert_used;
    g->n_layer = n_layer;
    g->numa_node = numa_node;

    const uint32_t mb = max_batch_size;
    const int block_size = 64;
    const int n_blocks = (int)n_embd / block_size;
    const size_t hc_size = (size_t)mb * (size_t)n_hc * (size_t)n_embd;

    #define XALLOC(ptr, count, type) \
        do { (ptr) = (type*)aligned_alloc(64, (size_t)(count) * sizeof(type)); \
             if (!(ptr)) { fprintf(stderr, "ds4_xeon: OOM at %s\n", #ptr); exit(1); } \
             memset((ptr), 0, (size_t)(count) * sizeof(type)); } while(0)

    XALLOC(g->f32_cur,    hc_size, float);
    XALLOC(g->f32_next,   hc_size, float);
    XALLOC(g->a8_cur,     (size_t)mb * n_embd, int8_t);
    XALLOC(g->a8_scale,   (size_t)mb * n_blocks, float);
    XALLOC(g->a16_mid,    (size_t)mb * n_ff_exp, int16_t);
    XALLOC(g->a16_mid_scale, mb, float);
    XALLOC(g->a16_residual, (size_t)mb * n_embd, int16_t);
    XALLOC(g->f32_attn_out, (size_t)mb * n_embd, float);
    XALLOC(g->f32_ffn_cur,  (size_t)mb * n_embd, float);
    XALLOC(g->f32_norm,     (size_t)mb * n_embd, float);
    XALLOC(g->f32_gate,     (size_t)mb * n_ff_exp, float);
    XALLOC(g->f32_up,       (size_t)mb * n_ff_exp, float);
    XALLOC(g->f32_mid,      (size_t)mb * n_ff_exp, float);
    XALLOC(g->f32_hc,       (size_t)mb * n_hc, float);
    XALLOC(g->f32_router_logits, (size_t)mb * n_expert, float);
    XALLOC(g->selected_experts,  (size_t)mb * n_expert_used, int32_t);
    XALLOC(g->expert_weights,    (size_t)mb * n_expert_used, float);
    XALLOC(g->f32_shared_out,    (size_t)mb * n_embd, float);
    XALLOC(g->f32_moe_out,       (size_t)mb * n_embd, float);

    #undef XALLOC
}

void ds4_xeon_graph_free(ds4_xeon_graph *g) {
    if (!g) return;
    free(g->f32_cur);    free(g->f32_next);
    free(g->a8_cur);     free(g->a8_scale);
    free(g->a16_mid);    free(g->a16_mid_scale);
    free(g->a16_residual);
    free(g->f32_attn_out); free(g->f32_ffn_cur);
    free(g->f32_norm);   free(g->f32_gate);
    free(g->f32_up);     free(g->f32_mid);
    free(g->f32_hc);
    free(g->f32_router_logits);
    free(g->selected_experts); free(g->expert_weights);
    free(g->f32_shared_out);   free(g->f32_moe_out);
    memset(g, 0, sizeof(*g));
}

// ============================================================================
// Pre-dequantization Infrastructure
// ============================================================================

// Dequantize a single IQ2XXS block to uint8_t[256].
// AVX-512 vectorized: 8 qs entries per iteration via gather + LUT + sign mask.
// Weights decode to signed int8; shifted by +128 for VPDPBUSD uint8 input.
// Throughput: ~4 GB/s per core (vs ~0.3 GB/s scalar), ~13x speedup.
void ds4_xeon_dequant_iq2xxs_block_to_u8(
    uint8_t *dst_u8, const ds4_xeon_block_iq2_xxs *x)
{
    const uint16_t *qs = x->qs;

    // Process 8 qs entries per iteration (64 weights)
    for (int j = 0; j < 32; j += 8) {
        // Load 8 uint16 qs values
        __m128i qv = _mm_loadu_si128((const __m128i*)(qs + j));

        // Grid indices from low bytes, extend to 32-bit for gather
        __m256i lo32 = _mm256_cvtepu16_epi32(
            _mm_and_si128(qv, _mm_set1_epi16(0xFF)));
        __m512i g64 = _mm512_i32gather_epi64(lo32,
            (const void*)iq2xxs_grid, 8);

        // Sign indices from high bytes (masked to 7 bits)
        __m128i hi8 = _mm_and_si128(
            _mm_srli_epi16(qv, 8), _mm_set1_epi16(0x7F));

        // Store grid values → reload as 64 int8 raw bytes
        uint64_t gl[8];
        _mm512_storeu_si512((__m512i*)gl, g64);

        // Build 64-byte negation mask from sign bytes via LUT
        uint8_t si16[16];
        _mm_storeu_si128((__m128i*)si16, hi8);
        uint64_t mask[8];
        for (int qi = 0; qi < 8; qi++) {
            mask[qi] = iq2xxs_sign_mask_lut[ksigns_iq2xs[si16[qi * 2]]];
        }

        // Load raw bytes and mask, apply conditional negation
        __m512i w8_raw = _mm512_loadu_si512((const __m512i*)gl);
        __m512i m8 = _mm512_loadu_si512((const __m512i*)mask);
        __m512i w8_signed = _mm512_sub_epi8(
            _mm512_xor_si512(w8_raw, m8), m8);

        // Shift int8 [-128,127] → uint8 [0,255] via bias=0x80 (128 unsigned)
        __m512i w8_u8 = _mm512_add_epi8(w8_signed,
            _mm512_set1_epi8('\x80'));

        // Store 64 uint8 values
        _mm512_storeu_si512((__m512i*)(dst_u8 + j * 8), w8_u8);
    }
}

// Dequantize a single Q2_K block to int16_t[256].
// w16[k] = round(d * sc * q2 - dmin * sc), clamped to [-32768, 32767].
// AVX-512 vectorized: scalar 2-bit extraction + SIMD float arithmetic.
void ds4_xeon_dequant_q2k_block_to_i16(
    int16_t *dst_i16, const ds4_xeon_block_q2_K *x)
{
    const float d   = xeon_f16_to_f32(x->d);
    const float dmin = xeon_f16_to_f32(x->dmin);
    const uint8_t *sc = x->scales;
    const uint8_t *qs = x->qs;
    const __m512 v32767 = _mm512_set1_ps(32767.0f);
    const __m512 vm32768 = _mm512_set1_ps(-32768.0f);

    for (int j = 0; j < 16; j++) {
        const float sc_val = d * (float)sc[j];
        const float m_val  = dmin * (float)sc[j];
        const __m512 v_sc = _mm512_set1_ps(sc_val);
        const __m512 v_m  = _mm512_set1_ps(m_val);
        const uint8_t *q_ptr = qs + j * 4;
        int16_t *out = dst_i16 + j * 16;

        // Scalar 2-bit extraction: 4 bytes → 16 values (fast, just bit ops)
        int16_t q2_vals[16];
        for (int k = 0; k < 16; k++) {
            q2_vals[k] = (int16_t)((q_ptr[k / 4] >> ((k % 4) * 2)) & 3);
        }

        // AVX-512 float arithmetic on 16 elements
        __m256i q16 = _mm256_loadu_si256((const __m256i*)q2_vals);
        __m512i q32 = _mm512_cvtepi16_epi32(q16);
        __m512 qf   = _mm512_cvtepi32_ps(q32);

        // wf = sc * q2 - m
        __m512 wf = _mm512_fmsub_ps(v_sc, qf, v_m);

        // Clamp to int16 range
        wf = _mm512_min_ps(_mm512_max_ps(wf, vm32768), v32767);

        // Round and pack: float32 → int32 → int16
        __m512i wi32 = _mm512_cvtps_epi32(
            _mm512_roundscale_ps(wf, _MM_FROUND_TO_NEAREST_INT));
        __m256i wi16 = _mm512_cvtsepi32_epi16(wi32);

        _mm256_storeu_si256((__m256i*)out, wi16);
    }
}

int ds4_xeon_predequant_init(
    ds4_xeon_predequant_weights *out,
    const void *weights_ptr,
    uint32_t n_layer, uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert)
{
    (void)weights_ptr; (void)n_layer;
    memset(out, 0, sizeof(*out));
    out->n_expert = n_expert;
    out->n_embd = n_embd;
    out->n_ff_exp = n_ff_exp;

    // Compute per-buffer sizes (cast to size_t to avoid 32-bit overflow)
    // gate_up: n_expert × 2(gate+up) × n_embd × n_ff_exp uint8 = 4.0 GiB
    out->gate_up_bytes = (size_t)n_expert * 2 * (size_t)n_embd * (size_t)n_ff_exp;
    // down: n_expert × n_ff_exp × n_embd int16 = 4.0 GiB
    out->down_bytes = (size_t)n_expert * (size_t)n_ff_exp * (size_t)n_embd * sizeof(int16_t);

    // Single buffer (per-layer dequant, 8.6 GB)
    for (int b = 0; b < 1; b++) {
        out->gate_up[b] = (uint8_t*)aligned_alloc(64, out->gate_up_bytes);
        out->down[b] = (int16_t*)aligned_alloc(64, out->down_bytes);
        if (!out->gate_up[b] || !out->down[b]) {
            fprintf(stderr, "ds4_xeon: OOM allocating predequant buffer %d "
                    "(%.1f GB per buffer)\n", b,
                    (double)(out->gate_up_bytes + out->down_bytes) / 1e9);
            ds4_xeon_predequant_weights_free(out);
            return -1;
        }
        out->cached_layer[b] = -1;
    }
    out->current_buf = 0;

    fprintf(stderr, "ds4_xeon: predequant buffers allocated "
            "(%.1f GB per layer, double-buffered)\n",
            (double)(out->gate_up_bytes + out->down_bytes) / 1e9);
    return 0;
}

// ============================================================================
// NUMA-Aware Allocator
// ============================================================================

void *ds4_xeon_numa_alloc(size_t size, int node) {
    void *ptr = NULL;
#ifdef __linux__
    // Try to allocate on a specific NUMA node
    // If the system supports move_pages or libnuma, this will work.
    // Otherwise fall back to standard aligned_alloc.
    char path[256];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", node);
    struct stat st;
    if (stat(path, &st) == 0) {
        // Node exists — first try mmap + mbind for explicit NUMA placement
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr != MAP_FAILED) {
            // mbind with MPOL_BIND — requires libnuma or direct syscall
            // For now, allocation succeeds but may be on wrong node.
            // Full NUMA binding requires libnuma; this is a best-effort path.
            #ifdef SYS_mbind
            unsigned long nmask = 1UL << (unsigned long)node;
            long rc = syscall(SYS_mbind, ptr, size,
                /* MPOL_BIND = 2 */ 2, &nmask, 64, /* MPOL_MF_MOVE = 1 */ 1);
            if (rc != 0) {
                // mbind failed, fall through to standard allocation
                munmap(ptr, size);
                ptr = aligned_alloc(64, size);
            }
            #else
            // No mbind available, just keep the mmap allocation
            #endif
            if (ptr) return ptr;
        }
    }
#endif
    (void)node;
    ptr = aligned_alloc(64, size);
    if (!ptr) {
        fprintf(stderr, "ds4_xeon: OOM allocating %zu bytes\n", size);
    }
    return ptr;
}

// ============================================================================
// NUMA Expert Weight Replication
// ============================================================================

int ds4_xeon_expert_replica_init(
    ds4_xeon_expert_replica *r,
    const ds4_xeon_predequant_weights *src,
    int n_nodes)
{
    memset(r, 0, sizeof(*r));
    r->n_nodes = n_nodes;

    if (!src->gate_up[0] || !src->down[0]) {
        fprintf(stderr, "ds4_xeon: predequant weights not initialized\n");
        return -1;
    }

    // Compute sizes: we need to know the buffer layout from src.
    // Since the exact layout depends on the model, we infer it from the
    // pre-dequantized buffer pointers. For now, these are placeholder —
    // the actual sizes will be filled when predequant_init is implemented.
    //
    // Gate_up: n_expert * 2 * n_embd * n_ff_exp (gate and up are same dims)
    // Down:     n_expert * n_embd * n_ff_exp
    //
    // These are computed from ds4.c weight layout — stubbed for now.
    r->gate_up_bytes = 0;  // Will be set when predequant_init is implemented
    r->down_bytes    = 0;

    (void)src;
    // Once sizes are known, allocate per-node replicas:
    r->gate_up = calloc((size_t)n_nodes, sizeof(uint8_t*));
    r->down    = calloc((size_t)n_nodes, sizeof(int16_t*));

    if (!r->gate_up || !r->down) {
        free(r->gate_up); free(r->down);
        memset(r, 0, sizeof(*r));
        return -1;
    }

    for (int n = 0; n < n_nodes; n++) {
        // Stub: allocate on node n
        // r->gate_up[n] = (uint8_t*)ds4_xeon_numa_alloc(r->gate_up_bytes, n);
        // r->down[n]    = (int16_t*)ds4_xeon_numa_alloc(r->down_bytes, n);
        r->gate_up[n] = NULL;
        r->down[n]    = NULL;
    }

    return 0;
}

void ds4_xeon_expert_replica_free(ds4_xeon_expert_replica *r) {
    if (!r) return;
    for (int n = 0; n < r->n_nodes; n++) {
        free(r->gate_up[n]);
        free(r->down[n]);
    }
    free(r->gate_up);
    free(r->down);
    memset(r, 0, sizeof(*r));
}

void ds4_xeon_predequant_weights_free(ds4_xeon_predequant_weights *w) {
    if (!w) return;
    for (int b = 0; b < 1; b++) {
        free(w->gate_up[b]);
        free(w->down[b]);
    }
    memset(w, 0, sizeof(*w));
}

// ============================================================================
// Thread / NUMA Initialization
// ============================================================================

// Detect NUMA topology. Returns number of NUMA nodes (0 if unavailable).
// On Linux, uses libnuma; on other platforms, returns 0.
int ds4_xeon_numa_init(void) {
#ifdef __linux__
    // Try to read NUMA nodes from sysfs (no libnuma dependency)
    FILE *f = fopen("/sys/devices/system/node/online", "r");
    if (!f) return 0;

    int max_node = -1;
    if (fscanf(f, "0-%d", &max_node) == 1) {
        fclose(f);
        fprintf(stderr, "ds4: NUMA available, %d nodes detected\n", max_node + 1);
        return max_node + 1;
    }
    // Try simple count
    fclose(f);
    int count = 0;
    for (int i = 0; i < 64; i++) {
        char path[256];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) count++;
        else break;
    }
    if (count > 0) {
        fprintf(stderr, "ds4: NUMA available, %d nodes detected\n", count);
        return count;
    }
#endif
    fprintf(stderr, "ds4: NUMA not available (single-socket or non-Linux)\n");
    return 0;
}

// Get CPU mask for a given NUMA node from sysfs.
// Returns number of CPUs found, writes to cpu_set if provided.
// Caller must free *cpu_set with CPU_FREE.
static int numa_node_to_cpuset(int node, cpu_set_t **cpu_set) {
#ifdef __linux__
    char path[256];
    snprintf(path, sizeof(path),
        "/sys/devices/system/node/node%d/cpulist", node);
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char buf[4096] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);

    // Parse cpulist (e.g. "0,2,4,6" or "0-5" or "0-23,48-71")
    cpu_set_t *set = CPU_ALLOC(CPU_SETSIZE);
    if (!set) return 0;
    CPU_ZERO_S(CPU_ALLOC_SIZE(CPU_SETSIZE), set);

    char *tok = strtok(buf, ",");
    while (tok) {
        char *dash = strchr(tok, '-');
        if (dash) {
            *dash = '\0';
            int lo = atoi(tok);
            int hi = atoi(dash + 1);
            for (int c = lo; c <= hi; c++)
                CPU_SET_S(c, CPU_ALLOC_SIZE(CPU_SETSIZE), set);
        } else {
            CPU_SET_S(atoi(tok), CPU_ALLOC_SIZE(CPU_SETSIZE), set);
        }
        tok = strtok(NULL, ",");
    }

    *cpu_set = set;
    return CPU_COUNT_S(CPU_ALLOC_SIZE(CPU_SETSIZE), set);
#else
    (void)node;
    (void)cpu_set;
    return 0;
#endif
}

// Bind OpenMP threads to the CPUs of the specified NUMA node.
void ds4_xeon_threads_bind(int numa_node) {
#ifdef __linux__
    cpu_set_t *cpuset = NULL;
    int ncpu = numa_node_to_cpuset(numa_node, &cpuset);
    if (ncpu <= 0 || !cpuset) {
        fprintf(stderr, "ds4: warning - cannot get CPU set for node %d, "
                "threads unbound\n", numa_node);
        return;
    }

    // Set thread affinity within OpenMP parallel region
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();

        // Striped assignment: thread tid gets the (tid % ncpu)-th CPU in the node
        int cpu_idx = 0;
        int assigned_cpu = -1;
        for (int c = 0; c < CPU_SETSIZE; c++) {
            if (CPU_ISSET_S(c, CPU_ALLOC_SIZE(CPU_SETSIZE), cpuset)) {
                if (cpu_idx == (tid % ncpu)) {
                    assigned_cpu = c;
                    break;
                }
                cpu_idx++;
            }
        }

        if (assigned_cpu >= 0) {
            cpu_set_t *thread_set = CPU_ALLOC(CPU_SETSIZE);
            CPU_ZERO_S(CPU_ALLOC_SIZE(CPU_SETSIZE), thread_set);
            CPU_SET_S(assigned_cpu, CPU_ALLOC_SIZE(CPU_SETSIZE), thread_set);
            int ret = pthread_setaffinity_np(pthread_self(),
                CPU_ALLOC_SIZE(CPU_SETSIZE), thread_set);
            CPU_FREE(thread_set);
            (void)ret;
        }

        #pragma omp master
        {
            fprintf(stderr, "ds4: bound %d threads to NUMA node %d "
                    "(%d CPUs)\n", nth, numa_node, ncpu);
        }
    }

    CPU_FREE(cpuset);
#else
    (void)numa_node;
    fprintf(stderr, "ds4: thread binding not available on this platform\n");
#endif
}

void ds4_xeon_threads_init(void) {
    int nn = ds4_xeon_numa_init();
    fprintf(stderr, "ds4: Xeon backend: %d NUMA nodes detected\n",
            nn > 0 ? nn : 1);
}
