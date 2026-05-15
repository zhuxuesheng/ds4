这次修改后，我认为：

# 文档质量提升非常大

已经从：

```text id="ztzqth"
“理论型优化方案”
```

变成了：

# “真正可信的系统架构文档”。

尤其：

你已经补上了之前最缺失的：

* real-world bottleneck
* NUMA reality
* cache hierarchy
* TLB
* synchronization
* KV cache
* dequant contention
* token routing locality

现在：

## world model 是正确的。

这是最大的进步。

---

# 一、最大的改进：你终于把“TOPS ≠ tok/s”写进去了

这是最关键的。

你新增的：

## 2.5 Real-world Bottleneck Analysis

这一章：

# 非常好。

因为：

你终于承认：

```text id="6m50ib"
CPU inference 是 memory orchestration problem
```

而不是：

```text id="l5ag1t"
dense GEMM benchmark
```

这让整个文档：

## 可信度暴涨。

---

# 二、我最认可的修改

# 1. Effective bandwidth 30~80 GB/s

这是：

## 非常 realistic 的数字。

以前：

```text id="tb44fh"
260 GB/s sustained
```

那个其实是 STREAM benchmark。

不是：

MoE inference。

现在：

```text id="q4l4gb"
30~80 GB/s effective
```

这个：

# 非常接近现实。

---

# 三、Expert replication 是这版最关键升级

之前：

```text id="ghvv5j"
expert partition by NUMA
```

实际上不可行。

因为：

router 是动态的。

现在：

你改成：

# “full expert replication”

这一下：

## 整个 NUMA model 就成立了。

这是：

# 非常大的工程层级提升。

---

# 四、这个 tradeoff 非常正确

你现在明确：

```text id="a54k06"
RAM is abundant
latency is scarce
```

这是：

# CPU inference 的核心哲学。

很多 GPU 人会本能追求：

* 更小模型
* 更少 RAM

但 CPU：

真正 scarce 的是：

# memory latency。

所以：

```text id="2i2c2y"
memory for locality
```

是正确方向。

---

# 五、Pre-dequant 现在终于变成“核心设计”

之前只是提到。

现在：

你把：

# “runtime dequant is catastrophic”

说清楚了。

这是对的。

实际上：

很多 CPU LLM kernel：

## 真的一半时间在 unpack。

尤其：

Q4_K。

---

# 六、TLB 那一章非常好

这部分：

## 非常专业。

尤其：

```text id="9bgm9r"
165GB / 4KB = 41M pages
```

这个意识：

说明你已经开始：

# 从 OS/VM subsystem 层面思考 inference。

这是：

真正系统优化工程师的视角。

---

# 七、KV cache 终于被正确重视了

之前：

KV 基本缺席。

现在：

你终于意识到：

# 长上下文时，KV 会超过 expert。

这是非常正确的。

尤其：

```text id="7dj5qt"
128K context
=> 2.75GB KV read / decode step
```

这个量级：

完全可能。

---

# 八、Barrier overhead 分析很好

这部分：

## 非常接近真实 production inference。

尤其：

你开始意识到：

```text id="8bgs5o"
MoE synchronization
```

可能比 compute 更贵。

这是很多人忽略的。

---

# 九、最重要的修正：性能目标终于可信了

现在：

```text id="jop0bw"
Prefill: 70~140 tok/s
Decode: 5~15 tok/s
```

这个：

# 已经是 realistic engineering target。

终于不再：

```text id="qz3vws"
500 tok/s AVX512 fantasy
```

了。

---

# 十、我认为现在的性能区间

## “基本可信”

尤其：

### decode

```text id="1gfrh6"
5~15 tok/s
```

这个：

## 非常合理。

与你实测：

```text id="6s6d5s"
3.7 tok/s
```

之间：

有现实可优化空间。

---

# 十一、Prefill 70~140 tok/s

这里：

我认为：

## 偏乐观，但不离谱了。

如果：

* batch=1024
* perfect locality
* pre-dequant
* static graph
* hugepage
* fused routing

实验室 benchmark：

### 有机会。

但：

interactive prefill：

我认为：

更现实：

```text id="r4h9w8"
30~80 tok/s
```

---

# 十二、你现在最大的亮点

# “系统架构意识”

已经很强。

尤其：

你现在开始：

## 明确区分：

| 层级                 | 内容            |
| ------------------ | ------------- |
| Kernel             | VPDPBUSD      |
| Memory             | hugepage      |
| NUMA               | replication   |
| Scheduler          | worker group  |
| Runtime            | static graph  |
| VM subsystem       | TLB           |
| Inference behavior | routing drift |

这说明：

你已经不是：

“写intrinsics的人”。

而是在：

# 设计 inference runtime。

---

# 十三、但还有几个重要缺失

这里是：

我认为：

## 下一版应该继续加强的。

---

# 1. LLC-aware scheduling

现在：

你有：

```text id="k2s8b5"
NUMA-aware
```

但：

还缺：

# cache-aware。

例如：

## token regrouping by expert

真正关键的：

不是：

```text id="od0u3j"
NUMA local
```

而是：

# “多个 token 连续命中同一个 expert”

否则：

L3 reuse 仍然很差。

---

# 十四、你还缺：

# “Expert batching strategy”

这个：

## 极其关键。

例如：

```text id="sq6h9h"
token -> router
```

之后：

不是立刻 matmul。

而是：

# regroup:

```text id="5zc1dn"
expert_7:
  token 3
  token 18
  token 52
```

然后：

做：

# batched expert GEMM。

这会：

## dramatically 提升 cache reuse。

---

# 十五、Speculative decoding 还不够深入

你提到了。

但：

实际上：

# CPU-only inference 的未来核心。

因为：

CPU：

最大的敌人：

是：

```text id="3e0xlr"
single-token latency
```

Spec decode：

能直接：

# 把 latency amortize。

---

# 十六、我认为你现在最缺的一章

建议新增：

# “Execution Timeline Analysis”

例如：

```text id="i2hr7w"
token decode timeline:
  router
  dispatch
  gate/up
  swiglu
  down
  kv attention
  barrier
```

然后：

估算：

* DRAM stall
* cache miss
* barrier
* compute

占比。

否则：

现在：

还是：

## throughput-oriented。

但：

CPU inference：

更需要：

# latency accounting。

---

# 十七、你现在已经接近 production design doc 了

我现在会这样评价：

| 维度                         | 评分 |
| -------------------------- | -- |
| CPU architecture           | A  |
| NUMA understanding         | A  |
| Memory system              | A  |
| Quantization understanding | A  |
| Runtime architecture       | A  |
| Systems engineering        | A  |
| Real-world modeling        | A- |
| Scheduler/runtime          | B+ |
| Cache-local batching       | B  |
| Speculative decode         | B  |
| Latency analysis           | B- |

---

# 十八、最重要的是：

这版文档：

## 已经不再“像论文”。

而开始：

# “像真正能做出来的系统”。

这是最大的进步。

---

# 十九、我现在最认可的一句话

这句：

```text id="z9go6w"
The value of this design lies in the system architecture
as much as in the individual kernels.
```

这是：

# 完全正确的。

因为：

LLM CPU inference：

真正难的：

从来不是：

```text id="3jghfk"
vpdpbusd
```

而是：

# 整个 memory/scheduling/runtime architecture。

---

# 二十、我认为下一步真正值得做的

不是继续写文档。

而是：

# 立刻开始 microbenchmark。

优先级：

| 优先 | 内容                      |
| -- | ----------------------- |
| 1  | pre-dequant benchmark   |
| 2  | hugepage impact         |
| 3  | NUMA replication impact |
| 4  | expert regroup batching |
| 5  | barrier profiling       |
| 6  | LLC miss profiling      |
| 7  | token routing entropy   |

因为：

你现在：

## 最大风险已经不是“设计错误”。

而是：

# “真实 profiling 数据会不会推翻某些 assumptions”。

这是一个好现象。
