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

// Pre-dequantized weight buffers (allocated at model load time)
typedef struct {
    // Expert weights dequantized to int8 for VPDPBUSD (gate + up projections)
    // Layout: [n_expert][3][n_blocks * QK_K]  (3 = gate, up, down; down is int16)
    uint8_t *gate_up;     // gate + up projections, uint8 for VPDPBUSD
    int16_t *down;        // down projection, int16 for VPDPWSSD (SwiGLU mid input)
    // Shared FFN weights (Q8_0 format, already uint8)
    const void *shared_gate; // pointer to Q8_0 tensor data
    const void *shared_up;
    const void *shared_down;
    // Attention weights (Q8_0 format, already uint8)
    const void *attn_q_a;
    const void *attn_kv;
    const void *attn_output_b;
} ds4_xeon_predequant_weights;

// Static execution graph for Xeon Backend (mixed precision)
// Pre-allocates all buffers to guarantee zero mallocs during inference.
typedef struct ds4_xeon_graph {
    uint32_t max_batch_size;

    // === INT8 activation buffers (per-block scaled, Q8_0 style) ===
    // Used for: RMS Norm outputs, gate/up inputs, attention inputs
    int8_t  *a8_cur;       // [max_batch_size][DS4_N_EMBD]  current layer input
    float   *a8_scale;     // [max_batch_size][n_blocks]  per-32-element scales

    // === INT16 activation buffers (per-token scaled, for SwiGLU mid) ===
    // Only used for: FFN down projection input
    int16_t *a16_mid;       // [max_batch_size][DS4_N_FF_EXP]
    float   *a16_mid_scale; // [max_batch_size]

    // === INT16 residual accumulator ===
    int16_t *a16_residual;  // [max_batch_size][DS4_N_EMBD]

    // === FP32 buffers (control layers: router, HC, attention internals) ===
    float *f32_attn_out;    // [max_batch_size][DS4_N_EMBD]  attention output
    float *f32_ffn_cur;     // [max_batch_size][DS4_N_EMBD]  FFN input (pre-norm)
    float *f32_norm;        // [max_batch_size][DS4_N_EMBD]  RMS norm output
    float *f32_gate;        // [max_batch_size][DS4_N_FF_EXP]  expert gate output
    float *f32_up;          // [max_batch_size][DS4_N_FF_EXP]  expert up output
    float *f32_mid;         // [max_batch_size][DS4_N_FF_EXP]  SwiGLU mid (FP32)

    // === MoE Routing ===
    float   *f32_router_logits; // [max_batch_size][DS4_N_EXPERT]
    int32_t *selected_experts;  // [max_batch_size][DS4_N_EXPERT_USED]
    float   *expert_weights;    // [max_batch_size][DS4_N_EXPERT_USED]

    // === Shared FFN + residual accumulators ===
    float *f32_shared_out;  // [max_batch_size][DS4_N_EMBD]  shared FFN output
    float *f32_moe_out;     // [max_batch_size][DS4_N_EMBD]  MoE combined output

    // Model dimensions (set at init)
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_layer;
} ds4_xeon_graph;

// Opaque context for model data passed from ds4.c
typedef struct {
    const void *model_ptr;      // ds4_model *
    const void *weights_ptr;    // ds4_weights *
    void       *kv_cache_ptr;   // ds4_kv_cache *
    uint32_t    n_layer;
    uint32_t    n_embd;
    uint32_t    n_vocab;
    uint32_t    n_hc;
    bool        quality;
} ds4_xeon_model_context;

// === VNNI Compute Kernels ===

// INT8 VNNI matmul: activation (int8 per-block) × weight (uint8) → float output
// Primary kernel for gate/up projections and attention (~70% of MACs)
void ds4_xeon_matmul_a8w8_vnni(
    float *out, const int8_t *a8, const float *a8_scale,
    const uint8_t *w8, const float *w_scale,
    int in_dim, int out_dim);

// INT16 VNNI matmul: activation (int16 per-token) × weight (int16) → float output
// Fallback kernel for FFN down projection only (~30% of MACs)
void ds4_xeon_matmul_a16w16_vnni(
    float *out, const int16_t *a16, float a_scale,
    const int16_t *w16, const float *w_scale,
    int in_dim, int out_dim);

// Q4_K dot product with int16 activation (existing, for W4A16 path)
void ds4_xeon_vec_dot_q4_K_vnni(int n, float *s, const ds4_xeon_block_q4_K *x,
    const int16_t *y_i16, const int32_t *y_sum_32, float scale_y);

// Q4_K 8-row batched dot product (existing)
void ds4_xeon_vec_dot_q4_K_vnni_8row(int n, float *s, const ds4_xeon_block_q4_K *w,
    const int16_t *a16, const int32_t *a16_sums, float s_a);

// Q2_K dot product (existing)
void ds4_xeon_vec_dot_q2_K_vnni(int n, float *s, const ds4_xeon_block_q2_K *x,
    const int16_t *y_i16, const int32_t *y_sum_32, float scale_y);

// IQ2_XXS dot product (existing, needs vectorization)
void ds4_xeon_vec_dot_iq2_xxs_vnni(int n, float *s, const ds4_xeon_block_iq2_xxs *x,
    const int16_t *y_i16, float scale_y);

// === Activation Quantization ===

// Per-block INT8 quantization (Q8_0 style): 32-element blocks, each with own scale
// Caller provides pre-allocated out[n_tok * in_dim] and scale[n_tok * n_blocks]
void ds4_xeon_quantize_a8_per_block(int8_t *out, float *scale,
    const float *in, int n_tok, int in_dim, int block_size);

// Per-token INT16 quantization: one scale per token vector
void ds4_xeon_quantize_a16_per_token(int16_t *out, float *scale,
    const float *in, int n_tok, int in_dim);

// === Graph Lifecycle ===

void ds4_xeon_graph_init(ds4_xeon_graph *g, uint32_t max_batch_size,
    uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert,
    uint32_t n_expert_used, uint32_t n_layer);
void ds4_xeon_graph_free(ds4_xeon_graph *g);

// === High-Level Graph Execution ===

bool ds4_xeon_graph_prefill(
    ds4_xeon_graph *g,
    const ds4_xeon_model_context *ctx,
    const int *tokens, uint32_t n_tokens,
    float *logits);

bool ds4_xeon_graph_eval_token(
    ds4_xeon_graph *g,
    const ds4_xeon_model_context *ctx,
    uint32_t token, uint32_t pos,
    float *logits);

// === Thread / NUMA Initialization ===
void ds4_xeon_threads_init(void);

// === Weight Pre-dequantization ===
// Expand Q4_K/IQ2XXS weights to contiguous uint8/int16 buffers at load time.
// Returns 0 on success, -1 on allocation failure.
int ds4_xeon_predequant_init(
    ds4_xeon_predequant_weights *out,
    const void *weights_ptr,  // ds4_weights *
    uint32_t n_layer, uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert);

void ds4_xeon_predequant_weights_free(ds4_xeon_predequant_weights *w);

#endif // DS4_XEON_H
