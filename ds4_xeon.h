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

// Pre-dequantized weight buffers (per-layer caching, double-buffered)
typedef struct {
    // Expert weights dequantized for one layer at a time
    // gate_up: concatenated gate + up projection experts → uint8 for VPDPBUSD
    // Layout: [n_expert][2][n_embd][n_ff_exp] (gate then up, interleaved)
    uint8_t *gate_up[2]; // double-buffered: [0]=current, [1]=next
    int16_t *down[2];    // down projection, int16 for VPDPWSSD
    int      current_buf; // which buffer is "current" (0 or 1)
    int      cached_layer[2]; // which layer is cached in each buffer (-1 = none)
    size_t   gate_up_bytes; // per-buffer gate+up size
    size_t   down_bytes;    // per-buffer down size
    uint32_t n_expert, n_embd, n_ff_exp;
} ds4_xeon_predequant_weights;

// Static execution graph for Xeon Backend (mixed precision)
// Pre-allocates all buffers to guarantee zero mallocs during inference.
typedef struct ds4_xeon_graph {
    uint32_t max_batch_size;

    // === HC state buffers (FP32, used by prefill/decode layer loop) ===
    float  *f32_cur;       // [max_batch_size][DS4_N_HC * DS4_N_EMBD]  current HC state
    float  *f32_next;      // [max_batch_size][DS4_N_HC * DS4_N_EMBD]  next HC state

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
    float *f32_hc;          // [max_batch_size][DS4_N_HC]  host-context routing buffer

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
    uint32_t n_hc;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_layer;

    // NUMA configuration
    int  numa_nodes;         // number of NUMA nodes (0 if NUMA unavailable)
    int  numa_node;          // NUMA node this graph is bound to (-1 = interleaved)
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

// Batched INT8 VNNI matmul for prefill (n_tok tokens)
void ds4_xeon_matmul_a8w8_vnni_batch(
    float *out, const int8_t *a8, const float *a8_scale,
    const uint8_t *w8, const float *w_scale,
    int n_tok, int in_dim, int out_dim);

// INT16 VNNI matmul: activation (int16 per-token) × weight (int16) → float output
void ds4_xeon_matmul_a16w16_vnni(
    float *out, const int16_t *a16, float a_scale,
    const int16_t *w16, const float *w_scale,
    int in_dim, int out_dim);

// Batched INT16 VNNI matmul for prefill
void ds4_xeon_matmul_a16w16_vnni_batch(
    float *out, const int16_t *a16, const float *a16_scale,
    const int16_t *w16, const float *w_scale,
    int n_tok, int in_dim, int out_dim);

// Q4_K dot product with int16 activation (existing, for W4A16 path)
// === Q4_K Weight Unpack / Dequant ===

// Extract 256 nibbles from a Q4_K block to uint8_t[256] (values 0-15).
// The per-sub-block dequant parameters (d, dmin, sc[], m[]) are stored
// in the output arrays sc8 and m8 (8 entries each) for later use.
void ds4_xeon_unpack_q4_k_to_u8(
    uint8_t *u8, float *sc8, float *m8,
    const ds4_xeon_block_q4_K *x);

// Extract 256 nibbles from a Q4_K block to int16_t (raw values 0-15).
void ds4_xeon_unpack_q4_k_to_i16(
    int16_t *i16, const ds4_xeon_block_q4_K *x);

// Fully dequantize a Q4_K block to int16_t[256].
// w16[k] = round(d * sc * q4 - dmin * m), clamped to [-32768, 32767].
void ds4_xeon_dequant_q4_k_to_i16(
    int16_t *i16, const ds4_xeon_block_q4_K *x);

// === Q4_K VNNI Dot Products (existing) ===

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

// Xeon AVX-512 routed MoE: one expert's gate→up→SiLU→down
// out: [DS4_N_EMBD] accumulator, x: [DS4_N_EMBD] RMS-normed input
// gate/up_blocks: IQ2XXS row-major, down_blocks: Q2_K row-major
void ds4_xeon_routed_moe_one_expert(
    float *out, const float *x,
    const uint8_t *gate_blocks, const uint8_t *up_blocks,
    const uint8_t *down_blocks,
    uint64_t gate_row_bytes, uint64_t up_row_bytes,
    uint64_t down_row_bytes, float expert_weight);

// === Attention ===

// AVX-512 attention scores: QK^T + softmax + weighted sum (causal masking)
// q: [n_tok][DS4_N_HEAD * DS4_N_HEAD_DIM], raw_kv: [n_raw][DS4_N_HEAD_DIM]
// heads: [n_tok][DS4_N_HEAD * DS4_N_HEAD_DIM] output
void ds4_xeon_attn_scores(float *heads, const float *q, const float *raw_kv,
    uint32_t n_tok, uint32_t il);

// RMS Norm: out[i] = in[i] * (1/sqrt(mean(in^2)+eps)) * w[i]
void ds4_xeon_rms_norm(float *out, const float *in, const float *w, int n, float eps);

// SwiGLU activation: out[i] = sigmoid(x[i]) * x[i] * y[i]
void ds4_xeon_swiglu(float *out, const float *x, const float *y, int n);

// Vector axpy: y[i] += a * x[i]
void ds4_xeon_axpy_f32(float *y, const float *x, float a, int n);

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
    uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_hc,
    uint32_t n_expert, uint32_t n_expert_used, uint32_t n_layer,
    int numa_node);
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
// Detect NUMA topology and return number of NUMA nodes (0 if unavailable).
int  ds4_xeon_numa_init(void);

// Bind OpenMP threads to CPUs on the specified NUMA node.
void ds4_xeon_threads_bind(int numa_node);

// Initialize threads (legacy, delegates to bind with numa_node=0).
void ds4_xeon_threads_init(void);

// NUMA-local tensor data: returns model->map_alt + offset for node 1,
// model->map + offset for node 0. Set map pointers via ds4_xeon_set_numa_maps.
void ds4_xeon_set_numa_maps(const uint8_t *map0, const uint8_t *map1);
const uint8_t *ds4_xeon_tensor_data_numa(uint64_t abs_offset);

// === Weight Pre-dequantization ===

// Dequantize a single IQ2XXS block (66 bytes → 256 uint8).
// Output range [0,255] (shifted from signed int8 for VPDPBUSD).
void ds4_xeon_dequant_iq2xxs_block_to_u8(
    uint8_t *dst_u8, const ds4_xeon_block_iq2_xxs *x);

// Dequantize a single Q2_K block (84 bytes → 256 int16).
// Output clamped to [-32768, 32767] for VPDPWSSD.
void ds4_xeon_dequant_q2k_block_to_i16(
    int16_t *dst_i16, const ds4_xeon_block_q2_K *x);

// Expand Q4_K/IQ2XXS weights to contiguous uint8/int16 buffers at load time.
// Returns 0 on success, -1 on allocation failure.
int ds4_xeon_predequant_init(
    ds4_xeon_predequant_weights *out,
    const void *weights_ptr,  // ds4_weights *
    uint32_t n_layer, uint32_t n_embd, uint32_t n_ff_exp, uint32_t n_expert);

void ds4_xeon_predequant_weights_free(ds4_xeon_predequant_weights *w);

// === NUMA Expert Weight Replication ===
// Per-socket replica of pre-dequantized expert weights.
// On dual-socket systems, the same expert weights are replicated on each
// socket's local memory to avoid cross-socket traffic during inference.
typedef struct {
    // Gate + up projections (int8 for VPDPBUSD), replicated per socket
    // Layout: gate_up[node][n_expert][2 * n_blocks_per_expert * QK_K]
    uint8_t **gate_up;      // [numa_nodes][total_gate_up_bytes]
    // Down projections (int16 for VPDPWSSD), replicated per socket
    int16_t **down;         // [numa_nodes][total_down_bytes]
    // Sizes
    size_t   gate_up_bytes; // per-socket gate+up buffer size in bytes
    size_t   down_bytes;    // per-socket down buffer size in bytes
    int      n_nodes;       // number of NUMA nodes (replicas)
} ds4_xeon_expert_replica;

// Allocate per-socket expert weight replicas.
// Call after predequant_init to replicate the pre-dequantized weights.
int ds4_xeon_expert_replica_init(
    ds4_xeon_expert_replica *r,
    const ds4_xeon_predequant_weights *src,
    int n_nodes);

void ds4_xeon_expert_replica_free(ds4_xeon_expert_replica *r);

// === NUMA-Aware Allocator ===
// Allocate memory on a specific NUMA node (Linux only, falls back to aligned_alloc).
void *ds4_xeon_numa_alloc(size_t size, int node);

#endif // DS4_XEON_H
