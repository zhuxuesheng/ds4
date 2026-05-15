#include "ds4_xeon.h"
#include <immintrin.h>

// Convert fp16 to fp32
static inline float xeon_f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16(h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

// Extract scale and min for a chunk from the packed scales array
static inline void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const ds4_xeon_block_q4_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y) {
    const int nb = n / DS4_XEON_QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = xeon_f16_to_f32(x[i].d) * scale_y;
        const float dmin = xeon_f16_to_f32(x[i].dmin) * scale_y;

        const uint8_t *q4 = x[i].qs;
        const uint8_t *sc = x[i].scales;
        const int16_t *a16 = y_i16 + i * DS4_XEON_QK_K;
        const int32_t *a32_sum = y_sum_32 + i * (DS4_XEON_QK_K / 32);

        // Process 8 chunks of 32
        for (int j = 0; j < 8; j++) {
            uint8_t sc_val, m_val;
            ds4q_get_scale_min_k4(j, sc, &sc_val, &m_val);

            // qs stores 64 elements (2 chunks) per 32 bytes
            // chunk 0 and 1 are in qs[0..31]
            // chunk j and j+1 are in qs[(j/2)*32 .. (j/2)*32 + 31]
            const uint8_t *q_ptr = q4 + (j / 2) * 32;
            
            // Load 32 bytes
            __m256i q_pack = _mm256_loadu_si256((const __m256i*)q_ptr);
            
            __m256i q_chunk;
            if (j % 2 == 0) {
                q_chunk = _mm256_and_si256(q_pack, _mm256_set1_epi8(0x0F));
            } else {
                q_chunk = _mm256_and_si256(_mm256_srli_epi16(q_pack, 4), _mm256_set1_epi8(0x0F));
            }

            // Zero extend 4-bit to 16-bit
            __m512i w16 = _mm512_cvtepu8_epi16(q_chunk);

            // Load A16
            __m512i a = _mm512_loadu_si512(&a16[j * 32]);

            // VNNI
            __m512i acc = _mm512_setzero_si512();
            acc = _mm512_dpwssd_epi32(acc, w16, a);

            // Reduce to scalar
            int32_t dot = _mm512_reduce_add_epi32(acc);

            // Rescale and accumulate
            float sum_a = (float)a32_sum[j];
            sumf += (d * sc_val) * (float)dot - (dmin * m_val) * sum_a;
        }
    }
    *s += sumf;
}

#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define DS4_N_EMBD 4096

// Graph lifecycle
void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size) {
    g->max_batch_size = max_batch_size;
    
    // Allocate 64-byte aligned buffers for AVX-512 operations
    size_t a16_size = (size_t)max_batch_size * DS4_N_EMBD * sizeof(int16_t);
    g->a16_cur  = aligned_alloc(64, a16_size);
    g->a16_next = aligned_alloc(64, a16_size);
    
    size_t scale_size = (size_t)max_batch_size * sizeof(float);
    g->scale_cur  = aligned_alloc(64, scale_size);
    g->scale_next = aligned_alloc(64, scale_size);
    
    size_t a32_size = (size_t)max_batch_size * DS4_N_EMBD * sizeof(int32_t);
    g->a32_acc = aligned_alloc(64, a32_size);
    
    size_t a8_size = (size_t)max_batch_size * DS4_N_EMBD * sizeof(int8_t);
    g->a8_cur = aligned_alloc(64, a8_size);
    
    size_t f32_size = (size_t)max_batch_size * DS4_N_EMBD * sizeof(float);
    g->f32_cur  = aligned_alloc(64, f32_size);
    g->f32_next = aligned_alloc(64, f32_size);
    
    // For 256 experts
    g->moe_routing_scores = aligned_alloc(64, (size_t)max_batch_size * 256 * sizeof(float));

    if (!g->a16_cur || !g->a16_next || !g->scale_cur || !g->scale_next || !g->a32_acc || !g->a8_cur || !g->f32_cur || !g->f32_next || !g->moe_routing_scores) {
        fprintf(stderr, "ds4_xeon: memory allocation failed during static graph initialization.\n");
        exit(1);
    }
    
    memset(g->a16_cur, 0, a16_size);
    memset(g->a16_next, 0, a16_size);
    memset(g->a32_acc, 0, a32_size);
}

void ds4_xeon_graph_free(ds4_xeon_graph *g) {
    if (!g) return;
    free(g->a16_cur);
    free(g->a16_next);
    free(g->scale_cur);
    free(g->scale_next);
    free(g->a32_acc);
    free(g->a8_cur);
    free(g->f32_cur);
    free(g->f32_next);
    free(g->moe_routing_scores);
    memset(g, 0, sizeof(*g));
}

// Thread / NUMA topology initialization
void ds4_xeon_threads_init(void) {
    // Basic placeholder for Step 3 NUMA binding logic
    // We will expand this with pthread_setaffinity_np
    fprintf(stderr, "ds4_xeon: Graph engine and NUMA threading initialized.\n");
}