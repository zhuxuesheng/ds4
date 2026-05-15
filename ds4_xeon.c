#include "ds4_xeon.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#define DS4_N_EMBD 4096
#define DS4_N_FF_EXP 2048
#define DS4_N_EXPERT 256
#define DS4_N_EXPERT_USED 6

// Tables defined in ds4.c
extern const uint8_t ksigns_iq2xs[128];
extern const uint64_t iq2xxs_grid[256];

// Convert fp16 to fp32
static inline float xeon_f16_to_f32(uint16_t h) {
    __m128i v = _mm_set1_epi16(h);
    __m128 f = _mm_cvtph_ps(v);
    return _mm_cvtss_f32(f);
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512BW__)

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

// Highly optimized GEMV for Q4_K.
// Processes 8 output rows at a time to maximize register reuse of the A16 vector.
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
                __m256i q_chunk = (j % 2 == 0) ? _mm256_and_si256(q_pack, _mm256_set1_epi8(0x0F)) : _mm256_and_si256(_mm256_srli_epi16(q_pack, 4), _mm256_set1_epi8(0x0F));

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

// Single row Q4_K dot product (fallback)
void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const ds4_xeon_block_q4_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y) {
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
            __m256i q_chunk = (j % 2 == 0) ? _mm256_and_si256(q_pack, _mm256_set1_epi8(0x0F)) : _mm256_and_si256(_mm256_srli_epi16(q_pack, 4), _mm256_set1_epi8(0x0F));
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

// IQ2_XXS VNNI implementation (Partially vectorized)
void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const ds4_xeon_block_iq2_xxs *x, const int16_t *y_i16, float scale_y) {
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

// Q2_K VNNI implementation
void ds4_xeon_vec_dot_q2_K_vnni(int n, float *s, const ds4_xeon_block_q2_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y) {
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

// Activation quantization to INT16
void ds4_xeon_quantize_a16(int16_t *out, float *scale, const float *in, int n_tok, int n_embd) {
    #pragma omp parallel for
    for (int t = 0; t < n_tok; t++) {
        float max_val = 1e-9f;
        const float *in_row = in + (uint64_t)t * n_embd;
        for (int i = 0; i < n_embd; i++) {
            float v = fabsf(in_row[i]);
            if (v > max_val) max_val = v;
        }
        float s = max_val / 32767.0f;
        float inv_s = 1.0f / s;
        scale[t] = s;
        int16_t *out_row = out + (uint64_t)t * n_embd;
        for (int i = 0; i < n_embd; i++) {
            out_row[i] = (int16_t)roundf(in_row[i] * inv_s);
        }
    }
}

void ds4_xeon_matmul_q4_K_batch_vnni(float *out, const ds4_xeon_block_q4_K *w, const int16_t *a16, const float *a16_scale, int n_tok, int in_dim, int out_dim) {
    const int n_blocks = in_dim / DS4_XEON_QK_K;
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < n_tok; t++) {
        const int16_t *a16_row = a16 + (uint64_t)t * in_dim;
        float *out_row = out + (uint64_t)t * out_dim;
        float s_a = a16_scale[t];
        int32_t a16_sums[in_dim / 32];
        for (int j = 0; j < in_dim / 32; j++) {
            int32_t sum = 0;
            for (int k = 0; k < 32; k++) sum += (int32_t)a16_row[j * 32 + k];
            a16_sums[j] = sum;
        }
        int o = 0;
        for (; o <= out_dim - 8; o += 8) {
            ds4_xeon_vec_dot_q4_K_vnni_8row(in_dim, &out_row[o], w + (uint64_t)o * n_blocks, a16_row, a16_sums, s_a);
        }
        for (; o < out_dim; o++) {
            float sumf = 0.0f;
            ds4_xeon_vec_dot_q4_K_vnni(in_dim, &sumf, w + (uint64_t)o * n_blocks, a16_row, a16_sums, s_a);
            out_row[o] = sumf;
        }
    }
}

// Forward declarations
typedef struct ds4_tensor ds4_tensor;
typedef struct ds4_model ds4_model;
typedef struct ds4_layer_weights ds4_layer_weights;
extern const void * tensor_data(const ds4_model *m, const ds4_tensor *t);
extern void layer_hash_selected_experts(int *selected, const ds4_model *m, const ds4_layer_weights *l, int token);

void ds4_xeon_ffn_shared_batch(
    float *out,
    const void *e_ptr,
    const void *lw_ptr,
    const float *inp,
    const int *token_ids,
    int n_tok,
    int il) 
{
    (void)out; (void)e_ptr; (void)lw_ptr; (void)inp; (void)token_ids; (void)n_tok; (void)il;
    // Implementation will follow. For now, empty to fix compile errors.
}

#endif

// Graph lifecycle
void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size) {
    g->max_batch_size = max_batch_size;
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
    g->moe_routing_scores = aligned_alloc(64, (size_t)max_batch_size * 256 * sizeof(float));

    if (!g->a16_cur || !g->a16_next || !g->scale_cur || !g->scale_next || !g->a32_acc || !g->a8_cur || !g->f32_cur || !g->f32_next || !g->moe_routing_scores) {
        fprintf(stderr, "ds4_xeon: memory allocation failed.\n");
        exit(1);
    }
    memset(g->a16_cur, 0, a16_size);
    memset(g->a16_next, 0, a16_size);
}

void ds4_xeon_graph_free(ds4_xeon_graph *g) {
    if (!g) return;
    free(g->a16_cur); free(g->a16_next);
    free(g->scale_cur); free(g->scale_next);
    free(g->a32_acc); free(g->a8_cur);
    free(g->f32_cur); free(g->f32_next);
    free(g->moe_routing_scores);
    memset(g, 0, sizeof(*g));
}

void ds4_xeon_threads_init(void) {
    fprintf(stderr, "ds4: Xeon backend initialized (AVX-512 VNNI enabled)\n");
}
