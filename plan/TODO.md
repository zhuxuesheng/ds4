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

- [ ] **T2.3.3** 混合精度端到端 token 匹配测试
  - 文件: `tests/ds4_xeon_math_test.c`
  - 内容: 加载真实模型, 对 3 个 prompt (短推理/代码/长文本), 对比 `-xeon` 混合精度路径 vs `-cpu` 标量参考, 逐 token 对比
  - 验证: token 序列完全一致 (100+ tokens), logits 相对误差 <1e-3
  - 参照: plan Section 2.4 caveat, Section 4 Step 5.2

---

## Phase 3: Weight Dequantization & Pre-dequant Infrastructure

参照: `intel_xeon_optimization_plan.md` Section 4, Step 3

### 3.1 Q4_K 解包

- [ ] **T3.1.1** 实现 `unpack_q4_k_to_u8` — 4-bit nibbles → uint8_t
  - 文件: `ds4_xeon.c`
  - 内容: AVX-512 `vpsrlw`/`vpand` (nibble 提取) + `vpshufb` (重排) → 256×uint8_t
  - 验证: 对标量解包参考, bit-exact 匹配
  - 参照: plan Section 2.5 Bottleneck 2 (port contention), 现有 `ds4_xeon_vec_dot_q4_K_vnni_8row` 中的解包逻辑

- [ ] **T3.1.2** 实现 `unpack_q4_k_to_i16` — 4-bit nibbles → int16_t (供 VPDPWSSD)
  - 文件: `ds4_xeon.c`
  - 内容: 同上 + `vpmovsxbw` 符号扩展
  - 验证: bit-exact 对标量
  - 参照: plan Section 3 Phase 2 (down projection INT16)

- [ ] **T3.1.3** 基准测试: kernel 中 dequant 占比
  - 内容: 分别在 pre-dequant 和 on-the-fly dequant 模式下测试 `matmul_w4a8_vnni` 吞吐
  - 验证: 量化 dequant 开销占 kernel 总时间的百分比, 确认文档假设 (30-50%)
  - 参照: plan Section 2.5 Bottleneck 2

### 3.2 IQ2XXS 解包 (矢量化)

- [ ] **T3.2.1** 矢量化 `ds4_xeon_vec_dot_iq2_xxs_vnni` 标量内循环
  - 文件: `ds4_xeon.c` (现有函数, ~30 行标量 for 循环)
  - 内容: AVX-512 gather (`_mm512_i32gather_epi64`) 做 grid lookup, `_mm512_xor_si512` + `_mm512_sign_epi8` 处理 sign mask, 输入 `VPDPBUSD`
  - 验证: 吞吐提升 >5× vs 当前标量实现
  - 参照: plan Section 4 Step 3.3

- [ ] **T3.2.2** IQ2XXS 解包正确性验证
  - 内容: 对标量参考输出, 逐元素比较
  - 验证: bit-exact 匹配
  - 参照: plan Section 4 Step 3.5

### 3.3 Pre-dequant 加载器

- [ ] **T3.3.1** 实现 `xeon_predequant_load` — 模型加载时预解包
  - 文件: `ds4_xeon.c`
  - 内容: 遍历所有层 × 256 experts × 3 projections, 将 Q4_K → 连续 INT8 buffer, 将 IQ2XXS → 连续 INT8 buffer, 将 Q4_K down projection → 连续 INT16 buffer
  - 内存预估: Q4KExperts ~15GB → ~30GB INT8 + ~8GB INT16 (down only) ≈ ~38GB; IQ2XXS ~7GB → ~14GB INT8 + ~7GB INT16 ≈ ~21GB
  - 验证: 加载后 buffer 内容 vs 标量逐 block 解包 bit-exact 匹配
  - 参照: plan Section 6 Decision 2

- [ ] **T3.3.2** 实现 1GB hugepage 分配
  - 文件: `ds4_xeon.c`
  - 内容: 对 pre-dequant weight buffer 使用 `mmap(MAP_HUGETLB | MAP_ANONYMOUS, ..., size, ...)` 或 `hugetlbfs` 挂载
  - 验证: `/proc/pid/smaps` 显示 `KernelPageSize: 1048576 kB`, TLB miss 计数 (`perf stat -e dTLB-load-misses`) 降低 >90%
  - 参照: plan Section 2.5 Bottleneck 5, plan Section 6 Decision 3

- [ ] **T3.3.3** Pre-dequant 基准测试
  - 内容: 分别测量 pre-dequant 和 on-the-fly dequant 两种模式下, 单层 FFN expert matmul 的 wall-clock 时间
  - 验证: pre-dequant 模式快 >30%
  - 参照: plan Section 6 Decision 2

---

## Phase 4: Static Graph, NUMA Topology & Expert Replication

参照: `intel_xeon_optimization_plan.md` Section 4, Step 4

### 4.1 静态图定义

- [ ] **T4.1.1** 重构 `ds4_xeon_graph` 结构体
  - 文件: `ds4_xeon.h`
  - 内容: 按精度类型预分配 buffer:
    ```c
    struct ds4_xeon_graph {
        uint32_t max_batch_size;
        int8_t  *a8_attn_in;    // RMS norm → attention input (per-block scale)
        float   *a8_attn_scale;
        int8_t  *a8_ffn_in;     // RMS norm → FFN gate/up input
        float   *a8_ffn_scale;
        int16_t *a16_mid;       // SwiGLU mid → down input (per-token scale)
        float   *a16_mid_scale;
        int16_t *a16_residual;  // residual accumulator
        float   *f32_router;    // router logits
        float   *f32_hc;        // HC sinkhorn
        // ... prefill batch 扩展 ...
    };
    ```
  - 验证: `ds4_xeon_graph_init` 分配所有 buffer, `sizeof(*g)` 与预估一致
  - 参照: plan Section 3 Phase 1

- [ ] **T4.1.2** 实现 `ds4_xeon_graph_init` 和 `ds4_xeon_graph_free`
  - 文件: `ds4_xeon.c`
  - 内容: `aligned_alloc(64, ...)` 分配所有 buffer, 全部初始化为零, free 时检查非 NULL
  - 验证: valgrind / AddressSanitizer 零泄漏
  - 参照: plan Section 3 Phase 1

### 4.2 NUMA Expert 权重复制

- [ ] **T4.2.1** 检测 NUMA 拓扑
  - 文件: `ds4_xeon.c`
  - 内容: `numa_available()` + `numa_max_node()` + `numa_num_configured_cpus()` 检测
  - 验证: 在双路服务器上输出 `NUMA nodes: 2, CPUs: 48 per node`
  - 参照: plan Section 2.1

- [ ] **T4.2.2** 实现 expert 权重跨 socket 复制
  - 文件: `ds4_xeon.c`
  - 内容: `numa_alloc_onnode(..., node0)` 分配 Socket 0 副本, `numa_alloc_onnode(..., node1)` 分配 Socket 1 副本, memcpy 两份完整 expert 权重
  - 验证: 两个副本的物理地址位于不同 NUMA node (`numa_move_pages` 或 `/proc/pid/numa_maps` 确认)
  - 参照: plan Section 3 Phase 1 (expert replication), plan Section 6 Decision 1

- [ ] **T4.2.3** 静态图 buffer NUMA 分配
  - 文件: `ds4_xeon.c`
  - 内容: 每个 socket 的动态 buffer (activation, residual) 用 `numa_alloc_onnode` 分配到本地 node
  - 验证: `numa_maps` 确认 buffer 物理页在对应的 NUMA node
  - 参照: plan Section 4 Step 4.4

### 4.3 线程绑定

- [ ] **T4.3.1** 实现 `ds4_xeon_threads_init`
  - 文件: `ds4_xeon.c` (已有占位实现)
  - 内容: `pthread_setaffinity_np` 将 threads 0-47 绑定到 Socket 0 的 CPU set, threads 48-95 绑定到 Socket 1
  - 验证: 每个线程内 `sched_getcpu()` 返回的 CPU ID 在预期的 socket 范围内
  - 参照: plan Section 3 Phase 1, plan Section 4 Step 4.3

### 4.4 Lock-free Expert Dispatch

- [ ] **T4.4.1** 实现 worker-group 模型
  - 文件: `ds4_xeon.c`
  - 内容: 每个 socket 内的线程池进一步划分为 expert worker group, 每个 group 持有若干 expert 的处理权, token 通过原子队列 dispatch 到对应 worker group
  - 验证: barrier 数量从 ~6/layer 降到 ~3/layer (用 `perf record -e sched:sched_switch` 统计同步点)
  - 参照: plan Section 3 Phase 4

- [ ] **T4.4.2** Barrier 开销基准
  - 内容: 测量单层 3-barrier 路径 vs 6-barrier 路径的 wall-clock 时间
  - 验证: 3-barrier 路径快 >10%
  - 参照: plan Section 2.5 Bottleneck 4, plan Section 3 Phase 4

---

## Phase 5: Engine Integration & End-to-End Validation

参照: `intel_xeon_optimization_plan.md` Section 4, Step 5

### 5.1 Prefill/Decode 入口

- [ ] **T5.1.1** 实现 `ds4_xeon_graph_prefill`
  - 文件: `ds4_xeon.c`
  - 内容: 完整 prefill 循环 (embedding → 43 layers × [attention + MoE FFN + HC] → LM head), 调用 Phase 1-4 的 kernel
  - 验证: 编译通过, 函数签名匹配 `ds4_xeon.h` 声明
  - 参照: plan Section 3 Phase 3 (per-layer dispatch 流程图)

- [ ] **T5.1.2** 实现 `ds4_xeon_graph_eval_token`
  - 文件: `ds4_xeon.c`
  - 内容: 单 token decode 循环, 复用 prefill 的 attention kernel (batch=1 路径)
  - 验证: 编译通过, KV cache 读写正确
  - 参照: plan Section 5.2

- [ ] **T5.1.3** 在 `ds4.c` 中添加 dispatch hook (~30-50 行)
  - 文件: `ds4.c`
  - 内容: `ds4_session_create` 中 `if (backend == DS4_BACKEND_XEON)` 初始化 graph; prefill/decode 循环中 dispatch 到 xeon 函数
  - 验证: `-xeon` flag 可选中 Xeon 后端, 不破坏现有 `-cpu`/`-metal`/`-cuda` 路径
  - 参照: plan Section 5

### 5.2 端到端正确性

- [ ] **T5.2.1** Token 序列完全匹配测试
  - 文件: `tests/ds4_xeon_op_test.c` (已有框架) 或 `tests/ds4_test.c`
  - 内容: 3 个推理 prompt (短推理/代码/长文本), `-xeon` vs `-cpu` 逐 token 对比
  - 验证: >100 tokens 序列完全一致 (非 "多数一致", 是 "完全一致")
  - 参照: plan Section 4 Step 5.2

- [ ] **T5.2.2** Logits 误差测试
  - 内容: 对每个 token step, 对比 `-xeon` logits 和 `-cpu` logits 的相对误差
  - 验证: max 相对误差 <1e-3 (99.9% 的 logit 值)
  - 参照: plan Section 4 Step 5.2

- [ ] **T5.2.3** 长上下文 KV cache 正确性测试
  - 内容: 128K token 输入, 验证 `-xeon` KV cache 内容逐层、逐 head 与 `-cpu` 一致
  - 验证: token 输出完全匹配, KV 值 bit-exact
  - 参照: plan Section 4 Step 5.2

### 5.3 端到端性能

- [ ] **T5.3.1** Prefill 吞吐基准
  - 内容: batch=1024, 测量 wall-clock prefill tok/s
  - 验证: 达到 70-140 tok/s 目标区间
  - 参照: plan Section 2.6

- [ ] **T5.3.2** Decode 吞吐基准
  - 内容: batch=1, 短上下文, 测量 wall-clock decode tok/s
  - 验证: 达到 5-10 tok/s (Q4KExperts) 或 7-15 tok/s (IQ2XXS) 目标区间
  - 参照: plan Section 2.6

- [ ] **T5.3.3** Interactive prefill 性能
  - 内容: batch=16-64 (模拟用户 prompt 大小), 测量 prefill tok/s
  - 验证: 达到 30-80 tok/s 目标区间
  - 参照: plan Section 2.9

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

- [ ] **T7.1** 实现 router 后 token→expert 倒排索引
  - 文件: `ds4_xeon.c`
  - 内容: router top-k 后, 构建 `expert[i] → [(token_id, gate_score), ...]` 映射, 每个 socket 独立构建
  - 验证: 对 batch=1024, 每个 expert 平均分配 ~24 tokens, stddev 在合理范围
  - 参照: plan Section 3 Phase 5

- [ ] **T7.2** 实现 batched expert GEMM
  - 文件: `ds4_xeon.c`
  - 内容: 对每个 expert e, 收集所有 routed tokens 的 activation, 做 batch GEMM (gate/up/down)
  - 验证: 输出与逐 token 迭代 bit-exact 匹配
  - 参照: plan Section 3 Phase 5

- [ ] **T7.3** 实现结果 scatter
  - 文件: `ds4_xeon.c`
  - 内容: 将 batched expert output scatter 回 per-token residual buffer (含 expert_weight 加权)
  - 验证: residual 与逐 token 迭代完全相同
  - 参照: plan Section 3 Phase 5

- [ ] **T7.4** L3 cache 命中率验证
  - 内容: `perf stat -e LLC-load-misses,LLC-loads -p <pid>` 对比 regroup 开启/关闭
  - 验证: regroup 开启后 LLC hit rate >80% (expert weight access), 关闭时 <20%
  - 参照: plan Section 3 Phase 5, plan Section 7 Priority 1

- [ ] **T7.5** Expert batching 性能基准
  - 内容: batch=1024, 对比 regroup 开启/关闭的 prefill tok/s
  - 验证: regroup 开启后吞吐提升 >30%
  - 参照: plan Section 3 Phase 5

- [ ] **T7.6** Token routing entropy 统计
  - 内容: 在真实推理中统计每个 expert 被选中的频次分布, 计算 entropy / load balance
  - 验证: 输出 expert 负载分布直方图, 确认无不均衡热点
  - 参照: plan Section 7 Priority 1.7

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
