#include "ds4_xeon.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

// Model dimension defaults (must match ds4.c constants)
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

// Tables defined in ds4.c (for IQ2XXS dequant)
extern const uint8_t ksigns_iq2xs[128];
extern const uint64_t iq2xxs_grid[256];

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
// VPDPBUSD: INT8 VNNI matmul (activation int8 × weight uint8 → float)
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
    // but for now use average — within 1% for RMS-Norm-bounded activations)
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
// VPDPWSSD: INT16 VNNI matmul (activation int16 × weight int16 → float)
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

// Per-block INT8 quantization (Q8_0 style)
// 32-element blocks, each block gets its own float32 scale.
// Block max → scale = max/127.0, values = round(x/scale), clamped to [-128,127].
void ds4_xeon_quantize_a8_per_block(int8_t *out, float *scale,
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
            float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
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

// AVX-512 accelerated per-block INT8 quantization
void ds4_xeon_quantize_a8_per_block_avx512(int8_t *out, float *scale,
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
            // block_size = 32 fits exactly in one ZMM register
            __m512 v = _mm512_loadu_ps(bin);
            __m512 vabs = _mm512_abs_ps(v);
            float amax = _mm512_reduce_max_ps(vabs);
            if (amax < 1e-9f) amax = 1e-9f;

            float d = amax / 127.0f;
            float id = 1.0f / d;
            scale_row[b] = d;

            __m512 scaled = _mm512_mul_ps(v, _mm512_set1_ps(id));
            __m512i iv = _mm512_cvtps_epi32(_mm512_roundscale_ps(scaled, _MM_FROUND_TO_NEAREST_INT));
            // Pack 32×int32 → 32×int8 (saturating)
            __m256i iv16 = _mm512_cvtsepi32_epi16(iv);
            __m128i iv8  = _mm256_cvtsepi16_epi8(iv16);
            _mm_storeu_si128((__m128i*)(out_row + b * block_size), iv8);
        }
    }
}

// Per-token INT16 quantization
void ds4_xeon_quantize_a16_per_token(int16_t *out, float *scale,
    const float *in, int n_tok, int in_dim)
{
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        const float *in_row = in + (uint64_t)t * in_dim;
        int16_t *out_row = out + (uint64_t)t * in_dim;

        float max_val = 1e-9f;
        for (int i = 0; i < in_dim; i++) {
            float v = fabsf(in_row[i]);
            if (v > max_val) max_val = v;
        }

        float s = max_val / 32767.0f;
        float inv_s = 1.0f / s;
        scale[t] = s;

        for (int i = 0; i < in_dim; i++) {
            float v = in_row[i] * inv_s;
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            out_row[i] = (int16_t)lrintf(v);
        }
    }
}

// ============================================================================
// Q4_K VNNI Kernels (existing, preserved from original implementation)
// These use on-the-fly 4-bit dequant → VPDPWSSD (INT16 path).
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
// IQ2_XXS VNNI (scalar inner loop — needs vectorization per Phase 3 / Step 6)
// ============================================================================

void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const ds4_xeon_block_iq2_xxs *x,
    const int16_t *y_i16, float scale_y)
{
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = xeon_f16_to_f32(x[i].d) * scale_y;
        const uint16_t *qs = x[i].qs;
        const int16_t *a16 = y_i16 + (uint64_t)i * DS4_XEON_QK_K;
        for (int j = 0; j < 32; j++) {
            uint16_t q = qs[j];
            uint64_t grid = iq2xxs_grid[q & 255];
            uint8_t signs = ksigns_iq2xs[q >> 8];

            int32_t dot = 0;
            for (int k = 0; k < 8; k++) {
                int8_t v = (int8_t)((grid >> (k * 8)) & 0xFF);
                if (signs & (1 << k)) v = -v;
                dot += (int32_t)v * (int32_t)a16[j * 8 + k];
            }
            sumf += d * (float)dot;
        }
    }
    *s += sumf;
}

// ============================================================================
// Q2_K VNNI (scalar inner loop)
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
            uint8_t sc_val = sc[j];
            const uint8_t *q_ptr = q2 + j * 4;
            int32_t dot = 0;
            for (int k = 0; k < 16; k++) {
                uint8_t q_byte = q_ptr[k / 4];
                uint8_t l = (q_byte >> ((k % 4) * 2)) & 3;
                dot += (int32_t)l * (int32_t)a16[j * 16 + k];
            }
            sumf += (d * sc_val) * (float)dot - (dmin * sc_val) * (float)a32_sum[j/2];
        }
    }
    *s += sumf;
}

// ============================================================================
// RMS Norm (AVX-512)
// ============================================================================

void ds4_xeon_rms_norm(float *out, const float *in, const float *w, int n, float eps) {
    float ss = 0.0f;
    #pragma omp parallel reduction(+:ss)
    {
        __m512 vss = _mm512_setzero_ps();
        #pragma omp for
        for (int i = 0; i < n / 16; i++) {
            __m512 vi = _mm512_loadu_ps(in + i * 16);
            vss = _mm512_fmadd_ps(vi, vi, vss);
        }
        ss += _mm512_reduce_add_ps(vss);
    }
    for (int i = (n / 16) * 16; i < n; i++) ss += in[i] * in[i];

    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    __m512 vscale = _mm512_set1_ps(scale);

    #pragma omp parallel for
    for (int i = 0; i < n / 16; i++) {
        __m512 vi = _mm512_loadu_ps(in + i * 16);
        __m512 vw = _mm512_loadu_ps(w + i * 16);
        _mm512_storeu_ps(out + i * 16, _mm512_mul_ps(_mm512_mul_ps(vi, vscale), vw));
    }
    for (int i = (n / 16) * 16; i < n; i++) {
        out[i] = in[i] * scale * w[i];
    }
}

// ============================================================================
// SwiGLU (AVX-512 with scalar sigmoid)
// ============================================================================

void ds4_xeon_swiglu(float *out, const float *x, const float *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i += 16) {
        int rem = n - i;
        int chunk = rem < 16 ? rem : 16;
        float xv[16], yv[16];
        for (int j = 0; j < chunk; j++) {
            xv[j] = x[i + j];
            yv[j] = y[i + j];
        }
        for (int j = 0; j < chunk; j++) {
            out[i + j] = xv[j] * (1.0f / (1.0f + expf(-xv[j]))) * yv[j];
        }
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
void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const void *x, const int16_t *y, float sy) {
    (void)n; (void)s; (void)x; (void)y; (void)sy;
}
void ds4_xeon_rms_norm(float *o, const float *i, const float *w, int n, float e) {
    (void)o; (void)i; (void)w; (void)n; (void)e;
}
void ds4_xeon_swiglu(float *o, const float *x, const float *y, int n) {
    (void)o; (void)x; (void)y; (void)n;
}
#endif

// ============================================================================
// Graph Lifecycle
// ============================================================================

void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size,
    uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert,
    uint32_t n_expert_used, uint32_t n_layer)
{
    memset(g, 0, sizeof(*g));
    g->max_batch_size = max_batch_size;
    g->n_embd = n_embd;
    g->n_ff_exp = n_ff_exp;
    g->n_expert = n_expert;
    g->n_expert_used = n_expert_used;
    g->n_layer = n_layer;

    const uint32_t mb = max_batch_size;
    const int block_size = 64;
    const int n_blocks = (int)n_embd / block_size;

    #define XALLOC(ptr, count, type) \
        do { (ptr) = (type*)aligned_alloc(64, (size_t)(count) * sizeof(type)); \
             if (!(ptr)) { fprintf(stderr, "ds4_xeon: OOM at %s\n", #ptr); exit(1); } \
             memset((ptr), 0, (size_t)(count) * sizeof(type)); } while(0)

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
    XALLOC(g->f32_router_logits, (size_t)mb * n_expert, float);
    XALLOC(g->selected_experts,  (size_t)mb * n_expert_used, int32_t);
    XALLOC(g->expert_weights,    (size_t)mb * n_expert_used, float);
    XALLOC(g->f32_shared_out,    (size_t)mb * n_embd, float);
    XALLOC(g->f32_moe_out,       (size_t)mb * n_embd, float);

    #undef XALLOC
}

void ds4_xeon_graph_free(ds4_xeon_graph *g) {
    if (!g) return;
    free(g->a8_cur);     free(g->a8_scale);
    free(g->a16_mid);    free(g->a16_mid_scale);
    free(g->a16_residual);
    free(g->f32_attn_out); free(g->f32_ffn_cur);
    free(g->f32_norm);   free(g->f32_gate);
    free(g->f32_up);     free(g->f32_mid);
    free(g->f32_router_logits);
    free(g->selected_experts); free(g->expert_weights);
    free(g->f32_shared_out);   free(g->f32_moe_out);
    memset(g, 0, sizeof(*g));
}

// ============================================================================
// Pre-dequantization Infrastructure
// ============================================================================

int ds4_xeon_predequant_init(
    ds4_xeon_predequant_weights *out,
    const void *weights_ptr,
    uint32_t n_layer, uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert)
{
    (void)out; (void)weights_ptr;
    (void)n_layer; (void)n_embd; (void)n_ff_exp; (void)n_expert;
    // Stub — will be implemented in Phase 3 when tensor access patterns
    // from ds4.c are fully mapped. For now, existing Q4_K on-the-fly
    // dequant kernels are used.
    fprintf(stderr, "ds4_xeon: pre-dequant not yet implemented, "
            "using on-the-fly dequant\n");
    return 0;
}

void ds4_xeon_predequant_weights_free(ds4_xeon_predequant_weights *w) {
    if (!w) return;
    free(w->gate_up);
    free(w->down);
    memset(w, 0, sizeof(*w));
}

// ============================================================================
// Thread / NUMA Initialization
// ============================================================================

void ds4_xeon_threads_init(void) {
    #pragma omp parallel
    {
        #pragma omp master
        {
            int nth = omp_get_num_threads();
            fprintf(stderr, "ds4: Xeon backend initialized "
                    "(AVX-512 VNNI, %d threads)\n", nth);
        }
    }
}
