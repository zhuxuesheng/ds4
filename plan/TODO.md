# Xeon Backend — Detailed Implementation TODO

> 目标: decode 5-10 tok/s (IQ2XXS 模型), interactive prefill 30-80 tok/s, large-batch prefill 70-140 tok/s
> 方法: 先 benchmark 每个算子 → 推算每层时间 → 对比预算 → 组装完整流程
> 关键约束: 单 token decode 受 DRAM 带宽限制 (~7.6 GB 权重读取), 上限由带宽决定非算力

---

## 时间预算 (per token, per layer)

decode 目标 5-10 tok/s → 100-200ms/token → **2.33-4.65ms/layer**

| 子层 | 操作 | 预算 (μs) | 占比 |
|------|------|-----------|------|
| Attention | Q/KV/Output 投影 + QK^T + softmax + weighted sum | ~400 | ~15% |
| MoE FFN | 6 experts × (gate + up + SwiGLU + down) | ~1600 | ~60% |
| Shared FFN | gate + up + SwiGLU + down | ~300 | ~10% |
| HC + Norm + 其他 | HC pre/post, RMS norm, router, residual | ~200 | ~8% |
| Barrier × 3 | 线程同步 | ~50 | ~2% |
| **合计** | | **~2550** | |

> 如果单层超过 4.6ms, decode 无法达到 5 tok/s。每个算子的 benchmark 必须在这个预算内验证。

prefill interactive (batch=32) 目标 30 tok/s → 32/30 = 1.07s per prefill → 1.07/43 = **~25ms/layer**

---

## Phase A: 算子级 Benchmark — 用真实维度测量每个算子的吞吐

所有 benchmark 放在 `tests/ds4_xeon_op_bench.c`。用真实的模型维度参数：
- n_embd = 4096, n_ff_exp = 2048, n_head = 64, n_head_dim = 512
- n_expert = 256, n_expert_used = 6, n_lora_q = 1024, n_layer = 43

### A.1 IQ2XXS 即时反量化点积 (gate/up 投影)

**CPU 参考**: `ds4_vec_dot_iq2_xxs_q8_K` (ds4.c) — 标量 grid 查表 + 点积
**Xeon 已有**: `ds4_xeon_vec_dot_iq2_xxs_vnni` (ds4_xeon.c:808) — AVX-512 gather + LUT + VPDPWSSD, 2.6x vs scalar

Benchmark:
- 输入: INT16 activation [256], IQ2XXS block (66 bytes), scale_y
- 调用次数: gate 投影需要 2048 次 (每个输出行一次), 6 experts × 2 (gate+up) = 24576 次/layer
- 验证: 计算 1 次 dot 的延迟 (ns), 以及 24576 次的总时间 (μs)
- **预算**: gate+up 共 24576 dots/layer, 每个 dot 目标 < 16ns → 总时间 < 400μs

### A.2 Q2_K 即时反量化点积 (down 投影)

**CPU 参考**: `ds4_vec_dot_q2_K_q8_K` (ds4.c) — 标量 nibble 提取 + 点积
**Xeon 已有**: `ds4_xeon_vec_dot_q2_K_vnni` (ds4_xeon.c:884) — **标量内循环, 无 AVX-512**

Benchmark:
- 输入: INT16 activation [256], Q2_K block (84 bytes), scale_y, y_sum
- 调用次数: down 投影 4096 行, 6 experts × 4096 = 24576 次/layer
- 验证: 计算 1 次 dot 的延迟, 以及 24576 次的总时间
- **预算**: down 共 24576 dots/layer, 每个 dot 目标 < 12ns → 总时间 < 300μs
- **注意**: 当前标量实现远慢于此, 必须向量化

### A.3 Q8_0 VNNI matvec (共享 FFN + Attention 投影)

**CPU 参考**: `matvec_q8_0` / `matvec_q8_0_decode_scratch` (ds4.c) — Q8_0 反量化 + FP32 FMA
**Xeon 缺失**: 需要实现 `ds4_xeon_matvec_q8_0_vnni` — Q8_0 权重已是 INT8, 用 VPDPBUSD

Q8_0 格式: 每 32 元素一个 block (int8_t[32] + float scale)
- 反量化: w_f32[i] = w_i8[i] * scale
- VNNI 路径: 直接 VPDPBUSD(activation_i8, weight_u8) → INT32 acc → scale 修正 → float

Benchmark (3 种维度):
- (a) 4096 → 2048 (shared FFN gate/up): 2048 行, 每行 4096 维
- (b) 2048 → 4096 (shared FFN down): 4096 行, 每行 2048 维
- (c) 4096 → 1024 (Q projection LoRA down): 1024 行
- (d) 1024 → 32768 (Q projection LoRA up): 32768 行
- (e) 4096 → 512 (KV projection): 512 行
- (f) 32768 → 4096 (output projection): 4096 行 (grouped: 8 groups × 4096→1024 + 8192→4096)
- 验证: 每个 matvec 延迟 vs CPU 标量, 加速比目标 >3x
- **预算**: shared FFN ~200μs, attention 投影 ~200μs, 合计 ~400μs

### A.4 RMS Norm

**CPU 参考**: `rms_norm_weight` (ds4.c:3294) — FP32 FMA
**Xeon 已有**: `ds4_xeon_rms_norm` (ds4_xeon.c:916) — AVX-512

Benchmark:
- 输入: float[4096], float weight[4096]
- 调用次数: 5-6 次/layer (attn_norm, ffn_norm ×2, q_a_norm, q_out_norm, kv_norm)
- 验证: 延迟 < 2μs/次
- **预算**: ~10μs/layer

### A.5 INT8 per-block 量化 (RMS Norm 输出 → INT8)

**CPU 参考**: `quantize_q8_0_activation` (ds4.c:3844) — 标量 max + 量化
**Xeon 已有**: `ds4_xeon_quantize_a8_per_block` (ds4_xeon.c:419) — AVX-512 reduce_max + round+pack

Benchmark:
- 输入: float[4096], block_size=64, 输出 int8[4096] + float scale[64]
- 调用次数: 2-3 次/layer (FFN norm 输出, attention norm 输出)
- 验证: 延迟 < 5μs/次
- **预算**: ~15μs/layer

### A.6 INT16 per-token 量化 (SwiGLU mid → INT16)

**CPU 参考**: `ds4_quantize_row_q8_K` (ds4.c) — Q8_K 量化 (256-block)
**Xeon 已有**: `ds4_xeon_quantize_a16_per_token` (ds4_xeon.c:472) — AVX-512 reduce_max

Benchmark:
- 输入: float[2048], 输出 int16[2048] + float scale (per-token)
- 调用次数: 6 experts × 1 = 6 次/layer
- 验证: 延迟 < 3μs/次
- **预算**: ~18μs/layer

### A.7 SwiGLU

**CPU 参考**: `swiglu` (ds4.c:5827) — FP32
**Xeon 已有**: `ds4_xeon_swiglu` (ds4_xeon.c:1126) — AVX-512

Benchmark:
- 输入: float gate[2048], float up[2048], 输出 float[2048]
- 调用次数: 7 次/layer (6 experts + 1 shared)
- 验证: 延迟 < 2μs/次
- **预算**: ~14μs/layer

### A.8 Per-Expert 完整 MoE (组合 A.1+A.2+A.6+A.7)

**Xeon 已有**: `ds4_xeon_routed_moe_one_expert` (ds4_xeon.c:971) — 组合 gate→up→SwiGLU→quantize→down
- 使用 `static` buffer (线程不安全, 需要修)
- gate/up 用 `ds4_xeon_vec_dot_iq2_xxs_vnni`
- down 用 `ds4_xeon_vec_dot_q2_K_vnni` (标量!)

Benchmark:
- 输入: float x[4096] (RMS-normed), expert 的 IQ2XXS gate/up blocks, Q2_K down blocks, expert_weight
- 输出: float out[4096] (累加到 accumulator)
- 调用次数: 6 次/layer
- 验证: 单 expert 延迟, 以及 6 experts 总时间
- **预算**: 6 experts < 1600μs, 单 expert < 267μs

### A.9 Attention Scores (QK^T + softmax + weighted sum)

**Xeon 已有**: `ds4_xeon_attn_scores` (ds4_xeon.c:1047) — AVX-512 FMA + 标量 softmax
- 注意: 当前实现有 heap allocation (FIXME comment), 需要修

Benchmark:
- 输入: float q[64][512], float kv[n_raw][512], uint32_t n_visible
- decode 时 n_raw = pos+1 (随上下文增长)
- 验证: pos=100/1000/10000/32768 时的延迟
- **预算**: short context ~50μs, long context ~500μs

### A.10 HC pre/post (Host Context)

**CPU 参考**: `hc_pre_from_state_one_scratch` + `hc_post_one` (ds4.c) — FP32 FMA, 小维度
**策略**: 保持 CPU 路径, HC 操作太小 (<10μs), VNNI 无收益

Benchmark:
- 验证: HC pre + post 总延迟 < 20μs

### A.11 Router (F16 matvec + softmax + top-k)

**CPU 参考**: `layer_router_probs_one` + `layer_topk_selected_experts_from_probs` (ds4.c)
**策略**: 保持 CPU 路径, 4096×256 维度太小, FMA 足够

Benchmark:
- 验证: router 总延迟 < 10μs

### A.12 Embedding + LM Head

**策略**: 保持 CPU 路径
- Embedding: F16 查表 (129280 × 4096), 1 次/token
- LM Head: Q8_0 matvec 4096→129280, 1 次/token, 可以后续优化

Benchmark:
- 验证: embedding < 10μs, LM head < 500μs

---

## Phase B: 预反量化 — 实现 + 与即时反量化对比

Phase A 测试即时反量化路径 (on-the-fly dequant from IQ2XXS/Q2_K blocks)。Phase B 实现预反量化路径, 然后对比两种方案在 decode 和 prefill 场景下的优劣。

### B.1 IQ2XXS → uint8 预反量化

**已有**: `ds4_xeon_dequant_iq2xxs_block_to_u8` (ds4_xeon.c:1287) — AVX-512 gather+LUT, 4.63 GB/s

Benchmark:
- 输入: 1 个 expert 的 gate (2048 rows × 4096 cols → 2048×16=32768 blocks)
- 测量: 单 expert 的 gate+up 预反量化时间 (MB processed, GB/s)
- **全模型**: 256 experts × 43 layers × 3 projections ≈ 大量 blocks
- 验证: 全部 256 experts × 43 layers 预反量化总时间, 目标 < 120s (一次性启动成本)

### B.2 Q2_K → int16 预反量化

**已有**: `ds4_xeon_dequant_q2k_block_to_i16` (ds4_xeon.c:1337) — 标量 2-bit 提取 + AVX-512 float 运算

Benchmark:
- 输入: 1 个 expert 的 down (4096 rows × 2048 cols → 4096×8=32768 blocks)
- 测量: 吞吐 (GB/s)
- 验证: 对标量参考 bit-exact

### B.3 INT8 VNNI matmul with 预反量化权重 (VPDPBUSD)

**已有**: `ds4_xeon_matmul_a8w8_vnni` (ds4_xeon.c:72) — 微基准 13.72 TOPS
- 但这个 kernel 用的是纯 uint8 weight, 实际使用时需要结合 Q8_0 scale

需要实现: `ds4_xeon_matmul_a8w8_q80` — INT8 activation (per-32 block scale) × Q8_0 weight (per-32 block scale) → float
- Q8_0 weight 的 scale 需要在 VNNI 累加后应用

Benchmark:
- (a) 4096×2048 matmul (gate/up): 延迟和有效 TOPS
- (b) 2048×4096 matmul (down): 延迟和有效 TOPS
- 验证: 输出 vs CPU 标量参考, 相对误差 < 1e-3

### B.4 INT16 VNNI matmul with 预反量化权重 (VPDPWSSD)

**已有**: `ds4_xeon_matmul_a16w16_vnni` (ds4_xeon.c:260) — 微基准 5.10 TOPS

Benchmark:
- 输入: INT16 activation (per-token scale) × INT16 weight (来自 B.2 预反量化)
- 维度: 2048×4096 (down projection)
- 验证: 延迟, 以及 vs 即时反量化 Q2_K 点积的对比

### B.5 即时反量化 vs 预反量化 — 端到端 per-expert 对比

用同一个 expert 对比两种路径的完整延迟:

路径 1 (即时): `ds4_xeon_routed_moe_one_expert` (INT16 量化输入 → IQ2XXS dot ×2048×2 → SwiGLU → INT16 量化 → Q2_K dot ×4096)
路径 2 (预反量化): INT8 量化输入 → VPDPBUSD matmul ×2048×2 → SwiGLU → INT16 量化 → VPDPWSSD matmul ×4096

Benchmark:
- 单 expert, batch=1 (decode 场景)
- 单 expert, batch=32 (prefill 场景)
- 验证: 延迟对比表, 选出 decode/prefill 分别的最优方案
- **决策点**: 如果预反量化在 decode 中因为内存流量增大而慢于即时反量化, decode 走即时, prefill 走预反量化

---

## Phase C: 逐层时间预算验证

把 Phase A+B 的 benchmark 结果汇总, 计算单层 decode 和单层 prefill 的预计时间, 对比预算。

### C.1 Decode 逐层时间表

| 操作 | 调用次数 | 单次延迟 (实测) | 总时间 | 预算 | 状态 |
|------|---------|---------------|--------|------|------|
| RMS norm | 5 | ? μs | ? μs | 10 μs | |
| INT8 quant | 2 | ? μs | ? μs | 15 μs | |
| Router | 1 | ? μs | ? μs | 10 μs | |
| Expert gate (即时) | 6×2048 dots | ? μs | ? μs | 400 μs | |
| Expert up (即时) | 6×2048 dots | ? μs | ? μs | 400 μs | |
| SwiGLU | 7 | ? μs | ? μs | 14 μs | |
| INT16 quant mid | 6 | ? μs | ? μs | 18 μs | |
| Expert down (即时) | 6×4096 dots | ? μs | ? μs | 300 μs | |
| Shared FFN (Q8_0 VNNI) | 3 matvecs | ? μs | ? μs | 300 μs | |
| Attn Q proj (Q8_0 VNNI) | 2 matvecs | ? μs | ? μs | 80 μs | |
| Attn KV proj (Q8_0 VNNI) | 1 matvec | ? μs | ? μs | 40 μs | |
| Attn scores | 1 | ? μs | ? μs | 50 μs | |
| Attn out proj (Q8_0 VNNI) | 2 matvecs | ? μs | ? μs | 80 μs | |
| HC pre/post | 2 | ? μs | ? μs | 20 μs | |
| Barrier | 3 | ? μs | ? μs | 50 μs | |
| **合计** | | | **? μs** | **2550 μs** | |

- 验证: 实测合计 < 2550μs → decode 可达 5 tok/s; < 1300μs → 可达 10 tok/s
- 如果超预算: 标记瓶颈算子, 优先优化

### C.2 Prefill (batch=32) 逐层时间表

类似的表, 但用 batched kernel:
- Expert gate/up: `ds4_xeon_vec_dot_iq2_xxs_vnni` 对每个 batch token 调用, 或 batched version
- 预反量化路径: `ds4_xeon_matmul_a8w8_vnni_batch` (batch tokens per expert)
- Shared FFN: batched Q8_0 matmul
- Attention: batched 投影 + batched attention scores

- 验证: 单层 < 25ms → interactive prefill 可达 30 tok/s

---

## Phase D: Xeon Decode 路径实现

基于 Phase C 的验证结果, 用选定的算子组装完整的 xeon decode 路径。

### D.1 实现 `ds4_xeon_decode_token`

文件: `ds4_xeon.c` (新函数)
参考: `forward_token_raw_swa_cpu_decode_scratch` (ds4.c:8419) + `layer_forward_raw_swa_one` (ds4.c:8243)

完整流程:
```
1. HC pre (attn) — CPU 路径
2. RMS norm (attn_norm) — ds4_xeon_rms_norm
3. INT8 quant norm → ds4_xeon_quantize_a8_per_block
4. Q projection (LoRA) → VNNI Q8_0 matvec (down + RMSnorm + up)
5. KV projection → VNNI Q8_0 matvec
6. RoPE (CPU)
7. KV cache push + compressor + indexer (CPU, 保持与 KV cache 兼容)
8. Attention scores → ds4_xeon_attn_scores (修掉 malloc)
9. Reverse RoPE (CPU)
10. Output projection (grouped LoRA) → VNNI Q8_0 matvec
11. HC post (attn) — CPU 路径
12. HC pre (ffn) — CPU 路径
13. RMS norm (ffn_norm) → ds4_xeon_rms_norm
14. INT8 quant → ds4_xeon_quantize_a8_per_block
15. Router → CPU 路径 (F16 matvec, softmax, top-k)
16. For each of 6 selected experts:
    a. ds4_xeon_routed_moe_one_expert (即时反量化, 或预反量化路径)
17. Shared FFN → VNNI Q8_0 matvec ×3
18. MoE + Shared 合并
19. HC post (ffn) — CPU 路径
```

验证:
- 单层延迟 vs Phase C 预算对比
- 逐 token 输出 vs `--cpu` 后端完全一致 (100+ tokens)

### D.2 接入 `ds4_session_sync`

文件: `ds4.c` (ds4_session_sync, line 18442-18470)
- 替换 `forward_token_raw_swa_cpu_decode_scratch` 调用为 `ds4_xeon_decode_token`
- 保留 checkpoint 逻辑不变

### D.3 Decode 端到端基准

- `./ds4 --backend xeon -p "test" -n 100`
- 验证: decode tok/s ≥ 5 (IQ2XXS 模型)
- 如不达标: 回 Phase C 时间表, 找出超预算的算子, 优化

---

## Phase E: Xeon Prefill 路径实现

### E.1 实现 `ds4_xeon_prefill_layer`

文件: `ds4_xeon.c` (新函数)
参考: `prefill_xeon_graph` (ds4.c:16582) + `prefill_layer_major_cpu` (ds4.c:8479)

关键差异 vs decode:
- Batch tokens: 需要 batched kernel
- Token regrouping (Phase 7): 按 expert 分组 token, 每 expert 的权重读取一次, 处理所有已路由 token
- 预反量化在此场景优势最大: batch token 摊销了 weight DRAM 读取, 预反量化消除 port 5 竞争

流程 (per layer):
```
1. Batch HC pre (attn) — CPU 路径 (已有 batch 版本)
2. Batch RMS norm → ds4_xeon_rms_norm per token
3. Batch INT8 quant → ds4_xeon_quantize_a8_per_block per token
4. Batch attention (Q/KV/Output 投影 + scores) — 用 VNNI batch kernel
5. Batch HC post (attn)
6. Batch HC pre (ffn)
7. Batch RMS norm + INT8 quant
8. Batch router (所有 token)
9. Token regrouping: 构建 expert → token list 倒排索引 (已有 T7.1)
10. Per expert, batched GEMM:
    a. Collect tokens routed to expert e
    b. ds4_xeon_matmul_a8w8_vnni_batch (gate, 预反量化 uint8 weights)
    c. ds4_xeon_matmul_a8w8_vnni_batch (up)
    d. ds4_xeon_swiglu per token
    e. ds4_xeon_quantize_a16_per_token per token
    f. ds4_xeon_matmul_a16w16_vnni_batch (down, 预反量化 int16 weights)
    g. Scatter results to per-token MoE buffer
11. Batch shared FFN: VNNI Q8_0 batch matmul ×3
12. MoE + Shared 合并
13. Batch HC post (ffn)
```

- 验证: 单层延迟 vs Phase C 预算对比
- 与 `--cpu` 输出逐 token 一致

### E.2 替换 `prefill_xeon_graph`

文件: `ds4.c` (prefill_xeon_graph, line 16582)
- 用 `ds4_xeon_prefill_layer` 替代 `layer_attention_raw_swa_batch` + `layer_ffn_batch`
- 移除 `numa_maps = true` (不再需要 — 等 NUMA 复制实现后再启用)
- 保留 `f32_cur`/`f32_next` HC state buffer 的使用

### E.3 Prefill 端到端基准

- `./ds4 --backend xeon -p "long test prompt with 32+ tokens" -n 1`
- 验证: interactive prefill ≥ 30 tok/s
- `./ds4 --backend xeon -p "$(cat long_prompt.txt)" -n 1` (batch=1024+)
- 验证: large-batch prefill ≥ 70 tok/s

---

## Phase F: 清理 & 优化

### F.1 移除未使用的 OpenMP 依赖

- `ds4_xeon.c` 的所有 VNNI kernel 中的 `#pragma omp parallel for` 改为 `ds4_parallel_for`
- Makefile 中移除 `-fopenmp` (如果确认所有 kernel 已迁移)
- 验证: 编译通过, benchmark 性能无回归

### F.2 NUMA 权重复制 (T4.2.2)

- 前提: 预反量化权重已完成 (Phase B)
- 实现: 将预反量化的 uint8/int16 权重 memcpy 到 socket 1 本地内存
- `ds4_xeon_set_numa_maps` 已就绪
- 验证: `perf stat -e LLC-misses` 对比, NUMA miss 率显著下降

### F.3 线程绑定 (T4.3.1)

- `ds4_xeon_threads_bind` 已实现
- 在 `ds4_xeon_threads_init` 中调用, 将 pthread pool 线程绑定到 NUMA nodes
- 确保不与 `ds4_parallel_for` 的 pthread pool 冲突

### F.4 修复 attention scores 的 heap allocation

- `ds4_xeon_attn_scores` (ds4_xeon.c:1047) 中有 `aligned_alloc` + FIXME comment
- 改为使用预分配的 buffer (来自 ds4_xeon_graph 或 scratch)

### F.5 1GB Hugepage (可选)

- 需要内核 hugetlbfs 配置
- 如不可用, 用 `MAP_HUGETLB` + fallback 到 2MB transparent hugepage

---

## 进度检查点

每个 Phase 结束后对照目标:
- [ ] **Phase A 完成**: 每个算子延迟明确, 可知单层最坏时间
- [ ] **Phase B 完成**: 即时反量化 vs 预反量化决策明确 (decode 用哪个, prefill 用哪个)
- [ ] **Phase C 完成**: 逐层时间表填满实测数据, 预计 decode tok/s 和 prefill tok/s 明确
- [ ] **Phase D 完成**: decode ≥ 5 tok/s, token 输出与 CPU 一致
- [ ] **Phase E 完成**: interactive prefill ≥ 30 tok/s, large-batch prefill ≥ 70 tok/s
- [ ] **Phase F 完成**: 性能无回归, 无内存泄漏, 无线程竞争

## 偏移检测

如果任何 Phase 的验证不达标:
1. 回到 Phase C 的时间表, 定位超预算的算子
2. 优化该算子 (向量化、缓存、算法改进)
3. 更新 benchmark, 重新验证时间预算
4. 如果优化后仍不达标, 评估目标是否在当前硬件上可达 (参考 plan Section 2.6 的保守估算)
