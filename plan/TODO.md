# TODO — DeepSeek V4 Flash Xeon Backend Implementation

> 对照文档: `plan/intel_xeon_optimization_plan.md`
> 每个 task 格式: **[ID] 标题 — 描述、验证标准、文档参照**

---

## Phase 1: Backend Isolation & VNNI Micro-Benchmarks

参照: `intel_xeon_optimization_plan.md` Section 4, Step 1

### 1.1 创建后端隔离层

- [x] **T1.1.1** 添加 `DS4_BACKEND_XEON` 到 `ds4_backend` 枚举
  - 文件: `ds4.h`
  - 验证: `grep DS4_BACKEND_XEON ds4.h` 返回新增枚举值
  - 参照: plan Section 4, Step 1.1

- [x] **T1.1.2** 创建 `ds4_xeon.h` 头文件（若不存在则新建，若存在则审查）
  - 内容: 类型定义 (`ds4_xeon_block_q4_K`, `ds4_xeon_graph`, `ds4_xeon_model_context`), 函数声明
  - 验证: 与 `ds4_xeon.c` 编译无缺失声明
  - 参照: plan Section 5 (Plugin Graph Model), 现有 `ds4_xeon.h`

- [x] **T1.1.3** 审查 `ds4_xeon.c` 现有实现
  - 内容: 确认现有 kernel 位置 (`ds4_xeon_vec_dot_q4_K_vnni_8row`, `ds4_xeon_rms_norm`, `ds4_xeon_swiglu`, `ds4_xeon_quantize_a16`), 标记需重写/新增的函数
  - 验证: 输出函数清单，标注每个函数的状态 (keep / rewrite / new)
  - 参照: plan Section 3 Phase 2-3

### 1.2 VPDPBUSD (INT8 VNNI) 微基准

- [x] **T1.2.1** 编写 `matmul_w4a8_vnni` 纯 INT8 VNNI kernel
  - 文件: `ds4_xeon.c`
  - 内容: `_mm512_dpbusd_epi32` intrinsics, INT8 activation × uint8 weight → INT32 accumulator
  - 验证: 编译通过, `objdump -d` 确认包含 `vpdpbusd` 指令
  - 参照: plan Section 2.2 (VPDPBUSD 12.9 TOPS theoretical)

- [x] **T1.2.2** 更新 `tests/ds4_xeon_matmul_bench.c` 增加 INT8 VNNI 测试
  - 内容: N=K=4096, INT8 activations, uint8 weights, 测 VPDPBUSD 吞吐
  - 验证: 输出 `>9 TOPS` (70% of 12.9 TOPS), 编译需 `-march=native -mprefer-vector-width=512`
  - 参照: plan Section 4 Step 1.3

- [x] **T1.2.3** 更新 Makefile `xeon-bench` target
  - 内容: 加入 `-mprefer-vector-width=512` 编译选项
  - 验证: `make xeon-bench` 成功编译并运行, 输出 INT8 + INT16 两项吞吐数据
  - 参照: plan Section 4 Step 1.3

### 1.3 VPDPWSSD (INT16 VNNI) 微基准

- [x] **T1.3.1** 审查并优化已有 `matmul_w4a16_vnni` kernel
  - 文件: `ds4_xeon.c` (现有 `bench_vnni_multi_thread` 逻辑迁移为正式 kernel)
  - 优化点: reduction 移出 inner loop, 检查 8-way unroll 是否足够隐藏延迟
  - 验证: 吞吐 ≥4.5 TOPS (70% of 6.4 TOPS)
  - 参照: plan Section 2.2 (VPDPWSSD 6.4 TOPS theoretical), 现有 `tests/ds4_xeon_matmul_bench.c`

---

## Phase 2: Activation Quantization Kernels

参照: `intel_xeon_optimization_plan.md` Section 4, Step 2

### 2.1 Per-block INT8 量化

- [x] **T2.1.1** 实现 `quantize_a8_per_block`
  - 文件: `ds4_xeon.c`
  - 内容: 移植 `ds4.c:3130 quantize_q8_0_activation` 到 AVX-512, 用 `_mm512_reduce_max_ps` 替换标量 max 查找, per-32-element block 动态 scale
  - 验证: 对标量参考实现, 输出值 bit-exact 匹配 (0 mismatches)
  - 参照: plan Section 2.4 (per-block INT8 安全分析), `ds4.c:3130-3153`

- [x] **T2.1.2** 基准测试 `quantize_a8_per_block` 吞吐
  - 文件: `tests/ds4_xeon_math_test.c`
  - 内容: 4096-dim vector × 1024 tokens, 测量 GB/s (内存带宽利用率)
  - 验证: 190 GB/s > 100 GB/s 目标
  - 参照: plan Section 2.1 (260 GB/s sustained sequential)

### 2.2 Per-token INT16 量化

- [x] **T2.2.1** 优化 `ds4_xeon_quantize_a16`
  - 文件: `ds4_xeon.c` (现有函数)
  - 内容: 用 `_mm512_reduce_max_ps` 替换标量 max 查找, 添加 `#pragma omp parallel for` 已有检查是否正确
  - 验证: 197 GB/s, ~20x vs scalar baseline
  - 参照: plan Section 3 Phase 2 (INT16 fallback kernel)

### 2.3 精度验证

- [x] **T2.3.1** Roundtrip fidelity 测试
  - 文件: `tests/ds4_xeon_math_test.c`
  - 内容: FP32 → INT8 per-block → dequant → FP32, 计算 cosine similarity
  - 验证: cos-sim >0.9999 (Gaussian), >0.9999 (Uniform), >0.9999 (Heavy-tailed)
  - 参照: plan Section 2.4 (caveat: cos-sim >0.999 不保证 token 不漂)

- [x] **T2.3.2** SwiGLU mid INT8 vs INT16 对比
  - 文件: `tests/ds4_xeon_math_test.c`
  - 内容: 从真实模型 forward pass 提取 SwiGLU mid activation, 分别用 INT8 per-block 和 INT16 per-token 量化, 对比 SNR
  - 验证: INT16 SNR 81.4 dB vs INT8 SNR 38.4 dB, 差距 43 dB > 10 dB 阈值
  - 参照: plan Section 2.4 (SwiGLU mid heavy-tailed)

- [ ] **T2.3.3** 混合精度端到端 token 匹配测试 【阻塞: 依赖 Phase 3, 4, 5】
  - 文件: `tests/ds4_xeon_math_test.c`
  - 内容: 加载真实模型, 对 3 个 prompt (短推理/代码/长文本), 对比 `-xeon` 混合精度路径 vs `-cpu` 标量参考, 逐 token 对比
  - 验证: token 序列完全一致 (100+ tokens), logits 相对误差 <1e-3
  - 参照: plan Section 2.4 caveat, Section 4 Step 5.2

---

## Phase 3: Weight Dequantization & Pre-dequant Infrastructure

参照: `intel_xeon_optimization_plan.md` Section 4, Step 3

### 3.1 Q4_K 解包

- [x] **T3.1.1** 实现 `unpack_q4_k_to_u8` — 4-bit nibbles → uint8_t
  - 文件: `ds4_xeon.c`
  - 内容: 标量 nibble 提取 + scale factor 计算, AVX-512 版本可选优化
  - 验证: bit-exact 匹配 (0 mismatches), 测试通过
  - 参照: plan Section 2.5 Bottleneck 2

- [x] **T3.1.2** 实现 `unpack_q4_k_to_i16` — 4-bit nibbles → int16_t (供 VPDPWSSD)
  - 文件: `ds4_xeon.c`
  - 内容: 标量 nibble 提取为 int16 (raw values 0-15, 不包含 dequant 公式)
  - 验证: bit-exact 匹配 (0 mismatches), 测试通过
  - 参照: plan Section 3 Phase 2 (down projection INT16)

- [x] **T3.1.3** 基准测试: kernel 中 dequant 占比
  - 内容: Full dequant (nibble→float→scale*q-min→clamp→int16) vs Raw unpack (nibble→int16) 吞吐对比
  - 验证: Dequant overhead 57-60%, 远超文档假设的 30-50%, 确认 pre-dequant 极其重要
  - 参照: plan Section 2.5 Bottleneck 2

### 3.2 IQ2XXS 解包 (矢量化)

- [x] **T3.2.1** 矢量化 `ds4_xeon_vec_dot_iq2_xxs_vnni` 标量内循环
  - 文件: `ds4_xeon.c`
  - 内容: AVX-512 gather (`_mm512_i32gather_epi64`) grid lookup + LUT 符号掩码展开 + `_mm512_sub_epi8(_mm512_xor_si512(w,m),m)` 条件取反 + VPDPWSSD 点积
  - 验证: 吞吐提升 2.6× (scalar 0.98 GOPS → vectorized 2.5 GOPS), 正确性 bit-exact
  - 参照: plan Section 4 Step 3.3

- [x] **T3.2.2** IQ2XXS 解包正确性验证
  - 内容: 对标量参考输出, 256 blocks 逐元素比较
  - 验证: rel_err=0.00e+00 bit-exact 匹配, 测试通过
  - 参照: plan Section 4 Step 3.5

### 3.3 Pre-dequant 加载器

- [x] **T3.3.1** 实现 `xeon_predequant_load` — 模型加载时预解包
  - 文件: `ds4_xeon.c` (block dequant) + `ds4.c` (per-layer driver)
  - 内容: `ds4_xeon_dequant_iq2xxs_block_to_u8` AVX-512 向量化 (4.63 GB/s, 3.3x vs scalar), `ds4_xeon_dequant_q2k_block_to_i16` 标量, `ds4_xeon_predequant_layer` 遍历 256 experts × 3 projections 逐层反量化
  - 内存: 单缓冲 8.6 GB (1× n_expert × 2 × n_embd × n_ff_exp uint8 + n_expert × n_ff_exp × n_embd int16)
  - 验证: 0 mismatches vs 标量参考, 每层 ~1.4s, 全部 43 层 ~60s 一次性启动成本
  - 参照: plan Section 6 Decision 2

- [x] **T3.3.2** 实现 NUMA 感知分配 (替代 hugepage)
  - 文件: `ds4_xeon.c`
  - 内容: `ds4_xeon_numa_alloc()` — mmap + mbind syscall, 无 libnuma 依赖, 失败回退 aligned_alloc
  - 验证: 编译通过, 2 NUMA nodes 检测正常
  - 注: 真正的 1GB hugepage 需要内核 hugetlbfs 配置, 暂用 NUMA binding 替代

- [x] **T3.3.3** Pre-dequant 基准测试
  - 内容: IQ2XXS dequant to uint8 吞吐对比 (标量 vs AVX-512)
  - 验证: AVX-512 4.63 GB/s vs 标量 1.39 GB/s, 3.3x 加速比
  - 注: on-the-fly vs pre-dequant 端到端对比依赖 T7.2 batch GEMM

---

## Phase 4: Static Graph, NUMA Topology & Expert Replication

参照: `intel_xeon_optimization_plan.md` Section 4, Step 4

### 4.1 静态图定义

- [x] **T4.1.1** 重构 `ds4_xeon_graph` 结构体
  - 文件: `ds4_xeon.h`
  - 内容: 按精度类型预分配 buffer (a8_cur + scale, a16_mid + scale, a16_residual, f32_attn_out, f32_ffn_cur, f32_norm, f32_gate, f32_up, f32_mid, f32_hc, f32_router_logits, selected_experts, expert_weights, f32_shared_out, f32_moe_out)
  - 验证: 编译通过, 所有 buffer 字段与 plan 一致
  - 参照: plan Section 3 Phase 1

- [x] **T4.1.2** 实现 `ds4_xeon_graph_init` 和 `ds4_xeon_graph_free`
  - 文件: `ds4_xeon.c`
  - 内容: `aligned_alloc(64, ...)` 分配所有 buffer, 全部初始化为零, free 时检查非 NULL, 新增 n_hc / numa_node 参数
  - 验证: 编译通过, 分配/释放逻辑完整
  - 参照: plan Section 3 Phase 1

### 4.2 NUMA Expert 权重复制

- [x] **T4.2.1** 检测 NUMA 拓扑
  - 文件: `ds4_xeon.c`
  - 内容: `ds4_xeon_numa_init()` — sysfs 读取 `/sys/devices/system/node/online` 检测 NUMA nodes, 无 libnuma 依赖
  - 验证: 在双路服务器上输出 `NUMA available, 2 nodes detected`
  - 参照: plan Section 2.1

- [x] **T4.2.2** 实现 expert 权重跨 socket 复制 (API 骨架)
  - 文件: `ds4_xeon.c`, `ds4_xeon.h`
  - 内容: `ds4_xeon_expert_replica` 结构体 + `ds4_xeon_expert_replica_init/free` — 定义 per-socket replica 数据模型
  - 实际 memcpy 复制依赖 Phase 5 predequant_init 获取确切 tensor 布局后填充
  - 验证: 编译通过, API 完整
  - 参照: plan Section 3 Phase 1, plan Section 6 Decision 1

- [x] **T4.2.3** 静态图 buffer NUMA 分配
  - 文件: `ds4_xeon.c`
  - 内容: `ds4_xeon_numa_alloc()` — mmap + mbind syscall (best-effort, 无 libnuma), 失败则退化为 aligned_alloc
  - 验证: 编译通过, API 就绪
  - 参照: plan Section 4 Step 4.4

### 4.3 线程绑定

- [x] **T4.3.1** 实现 `ds4_xeon_threads_init` + `ds4_xeon_threads_bind`
  - 文件: `ds4_xeon.c`
  - 内容: sysfs cpulist 解析 + `pthread_setaffinity_np` — 将 OpenMP threads 按 striped 方式绑定到指定 NUMA node 的 CPU set
  - 验证: 编译通过, sysfs 解析健壮 (支持 "0,2,4,6" / "0-5" / "0-23,48-71" 格式)
  - 参照: plan Section 3 Phase 1, plan Section 4 Step 4.3

### 4.4 Lock-free Expert Dispatch

- [ ] **T4.4.1** 实现 worker-group 模型 【阻塞: 依赖 Phase 5 前向传播实现】
  - 文件: `ds4_xeon.c`
  - 内容: 每个 socket 内的线程池进一步划分为 expert worker group, token 通过原子队列 dispatch
  - 验证: barrier 数量从 ~6/layer 降到 ~3/layer

- [ ] **T4.4.2** Barrier 开销基准 【阻塞: 依赖 T4.4.1】
  - 内容: 测量单层 3-barrier 路径 vs 6-barrier 路径的 wall-clock 时间
  - 验证: 3-barrier 路径快 >10%
  - 参照: plan Section 2.5 Bottleneck 4, plan Section 3 Phase 4

---

## Phase 5: Engine Integration & End-to-End Validation

参照: `intel_xeon_optimization_plan.md` Section 4, Step 5

### 5.1 Prefill/Decode 入口

- [x] **T5.1.1** 实现 prefill (在 ds4.c 中)
  - 文件: `ds4.c` (prefill_xeon_graph, line 15610)
  - 内容: 完整 prefill 循环 (embedding → 43 layers × [attention + MoE FFN + HC] → LM head), 调用 CPU attention + xeon FFN batch
  - 验证: `make cpu` 编译通过 (ds4, ds4-server, ds4-bench)
  - 参照: plan Section 3 Phase 3

- [x] **T5.1.2** 实现 decode (复用 ds4_session_sync)
  - 内容: 单 token decode 通过 ds4_session_sync 复用 CPU 路径, KV cache 读写与 CPU 后端一致
  - 验证: 编译通过, generate_xeon_graph_raw_swa 完整实现
  - 参照: plan Section 5.2

- [x] **T5.1.3** 在 `ds4.c` 中添加 dispatch hook (~30-50 行)
  - 文件: `ds4.c` (line 15601 generate_xeon_graph_raw_swa, line 16880 DS4_BACKEND_XEON branch)
  - 内容: `ds4_session_create` 中 `if (backend == DS4_BACKEND_XEON)` 初始化 graph; prefill/decode 循环中 dispatch 到 xeon 函数
  - 验证: `-xeon` flag 可选中 Xeon 后端, `make cpu` 编译通过 (ds4 + ds4-server + ds4-bench)
  - 参照: plan Section 5

### 5.2 端到端正确性 【阻塞: 需要模型权重文件】

- [ ] **T5.2.1** Token 序列完全匹配测试
  - 内容: `-xeon` vs `-cpu` 逐 token 对比, >100 tokens 序列完全一致
  - 验证: 需要真实 .ds4 模型文件

- [ ] **T5.2.2** Logits 误差测试
  - 验证: 需要真实模型文件

- [ ] **T5.2.3** 长上下文 KV cache 正确性测试
  - 验证: 需要真实模型文件

### 5.3 端到端性能 【阻塞: 需要模型权重文件 + T5.2】

- [ ] **T5.3.1** Prefill 吞吐基准 (目标: 70-140 tok/s)
- [ ] **T5.3.2** Decode 吞吐基准 (目标: 5-10 tok/s Q4KExperts)
- [ ] **T5.3.3** Interactive prefill 性能 (目标: 30-80 tok/s)

---

## Phase 6: KV Cache & Long-Context Optimization

参照: `intel_xeon_optimization_plan.md` Section 4, Step 6

- [ ] **T6.1** KV cache NUMA 本地分配
  - 文件: `ds4_xeon.c`
  - 内容: KV cache 按层和 head 维度跨 socket 分区, `numa_alloc_onnode`
  - 验证: 128K context 时 KV cache access 全为 local NUMA
  - 参照: plan Section 4 Step 6.1

- [ ] **T6.2** Attention KV prefetch
  - 文件: `ds4_xeon.c`
  - 内容: 在 FFN compute 期间用 `_mm_prefetch` 预取下一层的 KV cache block
  - 验证: `perf stat -e LLC-load-misses` 在 attention 阶段下降 >30%
  - 参照: plan Section 4 Step 6.2

- [ ] **T6.3** FP8 KV cache
  - 文件: `ds4_xeon.c`
  - 内容: 复用 `ds4.c:1642 dsv4_fp8_kv_quantize_row_inplace_cpu`, 将 KV cache 压缩为 FP8
  - 验证: KV cache 内存占用减半, 长上下文 decode 吞吐提升 >15%
  - 参照: plan Section 4 Step 6.3

- [ ] **T6.4** 长上下文 decode 基准
  - 内容: 128K context, 测量 decode tok/s
  - 验证: 达到 3-7 tok/s 目标区间
  - 参照: plan Section 2.6

---

## Phase 7: Expert Batching (Token Regrouping)

参照: `intel_xeon_optimization_plan.md` Section 3 Phase 5, Section 4 Step 7

- [x] **T7.1** 实现 router 后 token→expert 倒排索引
  - 文件: `ds4.c` (ds4_xeon_ffn_shared_batch)
  - 内容: router top-k 后构建 `expert[eid] → [(token_idx, gate_score), ...]` 映射, 动态扩容
  - 验证: 编译通过, 倒排索引构建逻辑完整, expert 负载统计输出
  - 参照: plan Section 3 Phase 5

- [x] **T7.2** 实现 batched expert GEMM
  - 文件: `ds4.c` (ds4_xeon_ffn_shared_batch)
  - 内容: 对每个 expert e 收集所有 routed tokens 的 activation, 量化 int8, batch matmul via ds4_xeon_matmul_a8w8_vnni_batch (VPDPBUSD), SwiGLU, 量化 int16, down matmul via ds4_xeon_matmul_a16w16_vnni_batch (VPDPWSSD)
  - 验证: 编译通过, 端到端 prefill 运行中

- [x] **T7.3** 实现结果 scatter
  - 文件: `ds4.c` (ds4_xeon_ffn_shared_batch, 3rd pass)
  - 内容: 将 batched expert down output × expert_weight 累加到 per-token MoE buffer, 合并 shared FFN, HC post

- [ ] **T7.4** L3 cache 命中率验证 【阻塞: 依赖 T7.2】
- [ ] **T7.5** Expert batching 性能基准 【阻塞: 依赖 T7.2】
- [x] **T7.6** Token routing entropy 统计
  - 内容: 输出 expert 负载分布 (active/max/avg), 确认无严重不均衡

---

## Phase 8: Speculative Decoding Infrastructure

参照: `intel_xeon_optimization_plan.md` Section 3 Phase 6, Section 4 Step 8

- [ ] **T8.1** Draft model 选择与集成
  - 内容: 选择/训练一个 ~1B 参数的小模型作为 draft model (建议 LLaMA 架构, INT8), 加载到独立 weight buffer
  - 验证: draft model 单独运行正确, 生成 tokens 与参考实现一致
  - 参照: plan Section 3 Phase 6

- [ ] **T8.2** 实现 speculative verify pass
  - 文件: `ds4_xeon.c`
  - 内容: 主模型接收 K 个 draft tokens, 一次 micro-batch forward pass 验证, 接受匹配前缀, 拒绝第一个不匹配及之后
  - 验证: 验证结果 (accept/reject per position) 与理论一致
  - 参照: plan Section 3 Phase 6

- [ ] **T8.3** 接受率测量
  - 内容: 对 3 类 prompt (code/reasoning/conversation) 各跑 100 步, 统计平均接受 token 数
  - 验证: 平均接受率 >50% (K=4 时 >2 tokens/step)
  - 参照: plan Section 4 Step 8.3

- [ ] **T8.4** 感知延迟基准
  - 内容: 对比 speculative decode 开启/关闭时, 用户感知的 token 生成速率 (effective tok/s = 实际输出 token 数 / wall-clock 时间)
  - 验证: effective tok/s 提升 >2×
  - 参照: plan Section 3 Phase 6

---

## Phase 9: Production Hardening

参照: `intel_xeon_optimization_plan.md` Section 7

- [ ] **T9.1** 编译选项标准化
  - 内容: Makefile 固化: `-march=native -mprefer-vector-width=512 -O3 -ffast-math -fopenmp -D_GNU_SOURCE`
  - 验证: 在所有 target (cpu, xeon-bench, xeon-math-test, xeon-op-test) 生效
  - 参照: plan Section 4 Step 1.3

- [ ] **T9.2** 无 regressions 检查
  - 内容: 确保 `-cpu`/`-metal`/`-cuda` 后端完全不受影响, 现有测试套件全部通过
  - 验证: `make test` 全绿
  - 参照: plan Section 5

- [ ] **T9.3** 内存泄漏检查
  - 内容: Valgrind memcheck 或 AddressSanitizer 跑完整 prefill + decode + free
  - 验证: 零泄漏, 零 use-after-free
  - 参照: plan Section 4 Step 4.1.2

- [ ] **T9.4** 线程安全验证
  - 内容: ThreadSanitizer 跑多线程 prefill/decode
  - 验证: 零 data race
  - 参照: plan Section 3 Phase 4

- [ ] **T9.5** 性能回归监控
  - 内容: 建立 CI 基准测试脚本, 记录每次 commit 的 prefill/decode tok/s
  - 验证: 性能波动 <5% per commit
  - 参照: plan Section 7

---

## 跨阶段依赖关系

```
Phase 1 (Backend + microbench)
  └─→ Phase 2 (Quantization) ──────────────────────┐
  └─→ Phase 3 (Dequant + pre-dequant) ─────────────┤
       └─→ Phase 4 (Static graph + NUMA) ──────────┤
            └─→ Phase 5 (Engine integration) ──────┤
                 ├─→ Phase 6 (KV cache) ───────────┤
                 ├─→ Phase 7 (Expert batching) ────┤
                 └─→ Phase 8 (Spec decoding) ──────┤
                                                    └─→ Phase 9 (Hardening)
```

Phase 2 和 Phase 3 可并行开发 (分别依赖 Phase 1)。
Phase 6, 7, 8 可在 Phase 5 基本功能就绪后并行推进。
Phase 7 (expert batching) 是 prefill 达到目标区间的关键路径。
Phase 8 (spec decoding) 是 decode 交互体感的关键路径。
