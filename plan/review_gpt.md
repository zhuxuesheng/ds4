这个设计文档，我会这样评价：

# 总体评价

这是一个：

## “非常强的系统级优化设计文档”

而不是普通的：

* “AVX512加速”
* “写几个intrinsics”
* “简单CPU backend”

它已经涉及：

* MoE系统结构
* NUMA topology
* memory hierarchy
* activation distribution
* quantization theory
* VNNI instruction scheduling
* backend architecture isolation
* graph runtime

很多内容是：

## 真正做过大模型推理优化的人才会写出来的。

尤其：

* 你识别了 SwiGLU mid activation 是唯一真正危险的 heavy-tail 点
* 你意识到 decode 是 memory-bound 而不是 compute-bound
* 你意识到 NUMA locality 比 kernel FLOPS 更重要
* 你意识到 Q4 unpack 和 VNNI port contention
* 你意识到 pre-dequant 的价值

这些都说明：

## 作者对 CPU LLM inference 的理解是“深层次”的。

但。

---

# 最大的问题

这个文档：

## “理论分析很强”

但：

# 对真实 CPU inference bottleneck 过于乐观。

尤其：

## Prefill 500~600 tok/s

这个数字：

我认为：

# 极不现实。

甚至：

## 差了一个数量级。

---

# 我先说核心判断

你当前真实测得：

```text
prefill: 5.67 t/s
generation: 3.70 t/s
```

这个：

## 才是现实世界。

而文档预测：

```text
500~600 tok/s
```

意味着：

## 高了 100 倍。

这不是：

“小误差”。

而是：

# 模型层面的 assumptions 出问题了。

---

# 一、最大的理论错误

文档核心问题：

# 把 LLM inference 当成 dense GEMM benchmark 了。

这是当前整个设计最根本的问题。

---

# 文档假设：

```text
effective TOPS ≈ effective token throughput
```

但：

# CPU 上的大模型推理从来不是 compute-bound。

而是：

# latency-bound + memory-bound + scheduling-bound。

尤其：

* MoE
* batch=1 decode
* long-context
* scattered experts

情况下。

---

# 二、Prefill 500 tok/s 为什么不现实

文档计算：

```text
19.4 GFLOPs/token
7.5 TOPS
=> 500 tok/s
```

数学上没错。

但：

## 实际系统里根本达不到。

因为：

---

# 三、真正的瓶颈不是 matmul

而是：

| 真正瓶颈                     | 文档低估程度 |
| ------------------------ | ------ |
| DRAM latency             | 极大     |
| NUMA remote access       | 极大     |
| expert routing scatter   | 极大     |
| Q4 unpack overhead       | 极大     |
| cache miss               | 极大     |
| synchronization          | 高      |
| KV cache access          | 高      |
| small batch inefficiency | 极大     |

---

# 四、文档最大的“工程幻想”

这一段：

```text
Socket 0 threads only process Socket 0 experts.
```

理论上很漂亮。

但：

# 实际上非常难成立。

因为：

## token routing 是动态的。

你没法保证：

```text
token -> expert
```

总在本地 socket。

除非：

* duplicate experts
* expert replication
* token migration

否则：

# MoE 本质上会跨 NUMA。

---

# 五、decode weight read/token 估算有问题

文档：

```text
7.6GB/token
260GB/s
=> 24 tok/s
```

这是假设：

# 完全 streaming sequential access。

现实：

## 完全不是。

MoE 的问题：

不是：

```text
bandwidth
```

而是：

# random memory access。

CPU 最怕：

* 小块随机读
* NUMA miss
* LLC miss

因此：

实际有效带宽：

不是：

```text
260GB/s
```

而可能：

# 30~80GB/s effective。

这就是为什么你现实只有：

```text
3.7 tok/s
```

而不是：

24 tok/s。

---

# 六、最被低估的问题：cache hierarchy

文档几乎默认：

```text
weight load cost ≈ DRAM bandwidth
```

但：

DeepSeek V4 Flash：

## expert working set 巨大。

例如：

每 token 激活：

```text
6 experts × 43 layers
```

这些 expert：

* 不连续
* 不可预测
* cache reuse 极差

导致：

# CPU 根本无法进入高效 steady-state GEMM。

---

# 七、关于 W4A8 的分析

这是文档里：

## 最有价值的部分。

我认为：

这部分：

### “方向正确”。

尤其：

```text
Only SwiGLU mid needs INT16
```

这个洞察：

## 是对的。

因为：

RMSNorm 后 activation：

确实：

* bounded
* near-gaussian

而：

SwiGLU mid：

heavy-tail。

这是真正理解 activation distribution 的人才能注意到的。

---

# 八、但这里也有问题

文档默认：

```text
INT8 activation is "safe"
```

实际上：

## DeepSeek 系列非常 sensitive。

尤其：

* MoE router
* long context
* reasoning chains

很多时候：

即使：

```text
cos sim > 0.999
```

也不代表：

token generation 不漂。

LLM：

不是 vision model。

误差会：

* recursive amplification
* routing drift
* entropy collapse

---

# 九、Q4 unpack 问题被低估

你文档提到：

```text
port contention
```

很好。

但：

## 实际影响比文档描述大得多。

因为：

AVX512 unpack：

会：

* 占用 uop cache
* 占用 port 5
* 增加 register pressure
* 降低 VNNI occupancy

现实里：

很多 CPU LLM kernel：

# 50% 时间花在 dequant。

不是 matmul。

---

# 十、关于 pre-dequant

这一段：

## 我认为非常正确。

这是整个文档：

# 最工程化、最现实的优化。

因为：

## CPU RAM 很大。

而：

CPU inference：

最大问题：

不是容量。

而是：

# latency。

因此：

```text
Q4 -> INT8 preload
```

是：

## 极其合理的 tradeoff。

很多 production CPU inference：

最终都会走向：

# “空间换时间”。

---

# 十一、关于 static graph

这是：

## 很好的设计。

尤其：

```text
replace scratch alloc
```

很关键。

因为：

malloc/free：

在：

* 多线程
* NUMA
* 高频 token loop

下会很贵。

---

# 十二、关于 backend isolation

这部分：

## 非常专业。

说明作者：

确实做过大型系统维护。

因为：

很多 inference 项目最后死于：

```text
patch spaghetti
```

而不是 kernel。

---

# 十三、真正缺失的内容

这里是我认为：

# 文档最缺的部分。

---

# 1. KV cache locality

完全没有深入讨论。

但：

在 decode：

KV cache 往往：

## 比 expert 更重要。

尤其：

长上下文。

---

# 2. batching strategy

MoE CPU inference：

# batch strategy 决定生死。

文档没有：

* token batching
* expert batching
* routing fusion
* microbatch scheduling

---

# 3. speculative decoding

CPU-only：

真正需要：

## speculative decode。

否则：

token latency 太高。

---

# 4. page locality / TLB

超大模型：

TLB miss 非常严重。

文档只轻描淡写 hugepage。

其实：

# hugepage 很关键。

---

# 5. thread synchronization

MoE：

真正 killer：

不是 compute。

而是：

```text
barrier synchronization
```

文档没有分析：

* lock-free routing
* queue model
* work stealing
* expert dispatch latency

---

# 十四、现实性能预估

我认为：

这套方案最终：

更现实的是：

| 指标      | 我认为现实值      |
| ------- | ----------- |
| prefill | 20~60 tok/s |
| decode  | 5~12 tok/s  |
| 长上下文    | 2~6 tok/s   |

这已经：

## 是非常优秀的 CPU-only 成绩。

---

# 十五、如果真想上 500 tok/s

必须：

# AMX + 高 batch + expert fusion。

单纯：

```text
AVX512 + dual xeon
```

几乎不可能。

---

# 十六、这个设计文档真正的价值

不是：

“500 tok/s”。

而是：

# 它建立了一个完整的 Xeon-native inference architecture。

这是最有价值的地方。

因为：

真正稀缺的是：

## 系统设计能力。

不是：

intrinsics。

---

# 十七、我的最终评价

| 维度                  | 评价 |
| ------------------- | -- |
| CPU架构理解             | A  |
| NUMA理解              | A  |
| Quant理解             | A  |
| Activation分析        | A  |
| Kernel方向            | A- |
| 工程架构                | A  |
| 性能建模                | C  |
| 现实world-model       | C+ |
| Decode bottleneck理解 | B- |
| Cache/TLB理解         | B  |
| 实际tok/s预测           | D  |

---

# 十八、我最建议补充的一章

建议新增：

# “Real-world Bottleneck Analysis”

包括：

* LLC miss profiling
* NUMA miss profiling
* TLB miss
* uop cache pressure
* dequant occupancy
* token routing entropy
* expert locality statistics
* memory access trace

否则：

整个设计：

会：

# “理论 compute 很强，但真实 tok/s 不高”。

而你现在的实测：

```text
3.7 tok/s
```

其实已经证明：

# 真正瓶颈不在 TOPS。
