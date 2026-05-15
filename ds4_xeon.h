#ifndef DS4_XEON_H
#define DS4_XEON_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define DS4_XEON_QK_K 256

// DeepSeek V4 Q4_K block layout
typedef struct {
    uint16_t d;           // super-scale (fp16)
    uint16_t dmin;        // super-min (fp16)
    uint8_t  scales[12];  // 8x 6-bit scales and mins
    uint8_t  qs[DS4_XEON_QK_K / 2]; // 256 4-bit weights packed
} ds4_xeon_block_q4_K;

// DeepSeek V4 Q2_K block layout
typedef struct {
    uint8_t  scales[DS4_XEON_QK_K / 16];
    uint8_t  qs[DS4_XEON_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} ds4_xeon_block_q2_K;

// DeepSeek V4 IQ2_XXS block layout
typedef struct {
    uint16_t d;
    uint16_t qs[DS4_XEON_QK_K / 8];
} ds4_xeon_block_iq2_xxs;

// Static execution graph for Xeon Backend
// Pre-allocates all necessary intermediate buffers to guarantee zero mallocs during execution.
typedef struct {
    uint32_t max_batch_size;
    
    // Primary activation buffers (ping-pong between layers)
    // A16 layout: [max_batch_size][DS4_N_EMBD]
    int16_t *a16_cur;
    int16_t *a16_next;
    
    // Scale vectors (FP32) to track the true numerical scale of the A16 integers.
    // True value = a16_cur[i] * scale_cur[token_index]
    float *scale_cur;
    float *scale_next;
    
    // Temporary 32-bit accumulators for VNNI dot products
    // Layout: [max_batch_size][DS4_N_EMBD]
    int32_t *a32_acc;
    
    // Temporary INT8 activations for attention projections (which are Q8_0)
    int8_t *a8_cur;
    
    // High-capacity FP32 buffers for precise control layers (e.g., MLA, Router)
    float *f32_cur;
    float *f32_next;
    
    // MoE Routing
    float *moe_routing_scores;
} ds4_xeon_graph;

// Computes dot product: s += x * y
void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const ds4_xeon_block_q4_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y);
void ds4_xeon_vec_dot_q2_K_vnni(int n, float *s, const ds4_xeon_block_q2_K *x, const int16_t *y_i16, const int32_t *y_sum_32, float scale_y);
void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const ds4_xeon_block_iq2_xxs *x, const int16_t *y_i16, float scale_y);

// High-performance primitives for the prefill loop
void ds4_xeon_quantize_a16(int16_t *out, float *scale, const float *in, int n_tok, int n_embd);
void ds4_xeon_matmul_q4_K_batch_vnni(float *out, const ds4_xeon_block_q4_K *w, const int16_t *a16, const float *a16_scale, int n_tok, int in_dim, int out_dim);

// MoE and Shared FFN high-level dispatchers
void ds4_xeon_ffn_shared_batch(
    float *out,
    const void *e_ptr, // ds4_engine
    const void *lw_ptr, // ds4_layer_weights
    const float *inp,
    const int *token_ids,
    int n_tok,
    int il);

// Graph lifecycle
void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size);
void ds4_xeon_graph_free(ds4_xeon_graph *g);

// Thread / NUMA topology initialization
void ds4_xeon_threads_init(void);

#endif // DS4_XEON_H