# 一个 token 的完整旅程

<p class="lead">这一章把整本手册串成一条线：从用户输入的一句话开始，跟着它走过分词、嵌入、28 层 Transformer、输出层、采样和反分词，在真实模型上记录每一站的张量形状；然后在每一站停下来，问一句"推理引擎在这里做了什么优化"。最后用一张屋顶线分析表回答"decode 的时间到底花在了哪里"。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 对于 Qwen3-0.6B，prefill 17 个 token 时，`k_proj` 的输出形状是什么？第 0 层 KV Cache 的形状呢？
    2. prefill 时，输出层需要为全部 17 个位置计算 logits 吗？
    3. batch 从 1 增大到 64，decode 中哪类算子的算术强度随之增大，哪类不变？为什么这决定了大 batch 下的瓶颈？
    4. 哪些推理优化在数学上不改变输出，哪些会改变？"数学上不变"是否意味着结果逐位相同？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `k_proj` 输出 `[1, 17, 1024]`（8 个 KV 头 × 128 维）；拆成多头后第 0 层的 K、V 各是 `[1, 8, 17, 128]`。
    2. 不需要。生成只用最后一个位置的 logits，只算它能省下 LM Head 的大部分计算（除非要返回提示词的 logprobs）。
    3. 权重类算子（投影、FFN）的算术强度随 batch 增大，因为同一份权重被更多 token 复用；注意力读的是每个请求自己的 KV，强度只等于 GQA 的分组数，与 batch 无关。所以 batch 大了以后，读 KV 成为主要瓶颈。
    4. 数学上不变：KV Cache、分页、FlashAttention、前缀缓存、分块 prefill、连续批处理、贪心验证的投机解码等；会改变：量化、KV 量化、稀疏注意力、近似的草稿接受规则等。"数学上不变"不等于逐位相同：计算顺序和 batch 组成不同，浮点结果会有微小差异。

## 追踪张量形状

用 PyTorch 的 forward hook 记录第 0 层各个模块的输入输出形状，先 prefill 整个提示词，再 decode 一步：

```python
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

text = tok.apply_chat_template([{"role": "user", "content": "中国的首都是哪里？"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids

layer0 = model.layers[0]
watch = {
    "embed_tokens": model.embed_tokens, "input_layernorm": layer0.input_layernorm,
    "q_proj": layer0.self_attn.q_proj, "k_proj": layer0.self_attn.k_proj, "v_proj": layer0.self_attn.v_proj,
    "self_attn": layer0.self_attn, "post_attention_layernorm": layer0.post_attention_layernorm,
    "gate_proj": layer0.mlp.gate_proj, "up_proj": layer0.mlp.up_proj, "down_proj": layer0.mlp.down_proj,
    "norm": model.norm, "lm_head": model.lm_head,
}
shapes = {}

def recorder(name):
    def hook(module, args, output):
        shapes.setdefault(name, []).append((tuple(args[0].shape), tuple(output.shape)))
    return hook

handles = [m.register_forward_hook(recorder(n)) for n, m in watch.items()]
cache = KVCache(model.cfg.num_hidden_layers)
with torch.no_grad():
    logits = model(ids, cache)                                        # prefill
    next_id = logits[:, -1].argmax(-1)
    kv_after_prefill = tuple(cache.k[0].shape)
    logits = model(next_id[:, None], cache)                           # decode 一步
for h in handles:
    h.remove()

print(f"{'模块':26s}{'prefill 输入 → 输出':34s}decode 输入 → 输出")
for name in watch:
    (pi, po), (di, do) = shapes[name]
    print(f"{name:26s}{str(pi) + ' → ' + str(po):34s}{di} → {do}")
print("第 0 层的 K Cache", kv_after_prefill, "→", tuple(cache.k[0].shape))
print("生成：", repr(tok.decode(next_id)), repr(tok.decode(logits[:, -1].argmax(-1))))
```

在本手册的环境里运行得到：

```text
模块                        prefill 输入 → 输出                   decode 输入 → 输出
embed_tokens              (1, 17) → (1, 17, 1024)           (1, 1) → (1, 1, 1024)
input_layernorm           (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
q_proj                    (1, 17, 1024) → (1, 17, 2048)     (1, 1, 1024) → (1, 1, 2048)
k_proj                    (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
v_proj                    (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
self_attn                 (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
post_attention_layernorm  (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
gate_proj                 (1, 17, 1024) → (1, 17, 3072)     (1, 1, 1024) → (1, 1, 3072)
up_proj                   (1, 17, 1024) → (1, 17, 3072)     (1, 1, 1024) → (1, 1, 3072)
down_proj                 (1, 17, 3072) → (1, 17, 1024)     (1, 1, 3072) → (1, 1, 1024)
norm                      (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
lm_head                   (1, 17, 1024) → (1, 17, 151936)   (1, 1, 1024) → (1, 1, 151936)
第 0 层的 K Cache (1, 8, 17, 128) → (1, 8, 18, 128)
生成： '中国的' '首'
```

几个值得注意的地方：

- 除了注意力，**每个模块都是逐 token 独立计算的**：prefill 与 decode 的形状只差在 T（17 对 1）。这就是 prefill 是矩阵乘法（GEMM）、decode 是矩阵乘向量（GEMV）的原因。
- `k_proj`、`v_proj` 的输出是 1024 维（8 个 KV 头 × 128），`q_proj` 有 2048 维（16 个头 × 128）：这是 [GQA](../transformer/attention-variants.md#mqa-与-gqa)，两个 query 头共用一组 K、V。KV Cache 每层每个 token 存 2 × 1024 个数。注意 `q_proj` 的输出比隐藏维度 1024 还宽——Qwen3 的头维单独配置成 128。
- 注意力是唯一一个"跨 token"的操作：它把当前 token 与 KV Cache 里所有历史 token 联系起来。decode 时 query 只有 1 个，key 有 18 个。
- `lm_head` 输出 151936 维，是整个模型最宽的一层。

## 逐站解读

下表把每一站的计算与推理引擎中对应的优化放在一起。左边两列是"模型做了什么"，右边一列是"工程上怎么让它更快"。

![图：一个 token 的旅程——每一站的张量形状，prefill 与 decode 对照](../assets/figures/token-journey.svg){.aig-svg}

| 站点 | 形状（prefill） | 计算的本质 | 推理引擎中的优化 |
| --- | --- | --- | --- |
| 对话模板、分词 | 文本 → `[1, 17]` | CPU 上的字符串处理 | 模板必须与训练时一致（[对话模板](../basics/tokenization.md#特殊-token-与对话模板)）；分词放在独立进程，避免阻塞 GPU 调度 |
| 嵌入 | `[1, 17]` → `[1, 17, 1024]` | 按 id 查表 | 张量并行时按词表切分（vocab parallel） |
| RMSNorm + 残差 | `[1, 17, 1024]` | 逐元素 + 归约，访存密集 | 与残差相加融合成一个 kernel（fused add + RMSNorm），参见 CUDA 手册的 [Softmax 与归一化](cuda://kernels/softmax-norm/) |
| Q、K、V 投影 | → `[1, 17, 2048]`、`[1, 17, 1024]` × 2 | GEMM / GEMV | 三个矩阵拼成一个 `qkv_proj`，一次 GEMM；[权重量化](../inference/quantization.md)；张量并行按头切分 |
| QK-Norm、RoPE | 形状不变 | 逐元素（按头归一化、旋转） | 位置由 KV Cache 长度决定；与 QKV 或注意力 kernel 融合 |
| 写入 KV Cache | `[1, 8, 17, 128]` × 2 | 拷贝 | 写入[分页](../inference/kv-cache.md#kv-cache-的显存管理)的块中（`reshape_and_cache`）；可存成 FP8 |
| 注意力 | scores `[1, 16, 17, 17]` | prefill：计算密集；decode：读 KV，访存密集 | prefill 用 FlashAttention，不物化 scores；decode 用分页 decode kernel、split-KV（Flash-Decoding）；GQA 在 kernel 内共享 KV，参见 CUDA 手册的 [FlashAttention 与推理算子](cuda://advanced/attention/) |
| O 投影 | → `[1, 17, 1024]` | GEMM | 张量并行时按行切分，之后一次 all-reduce |
| SwiGLU MLP | → `[1, 17, 3072]` → `[1, 17, 1024]` | 三个 GEMM，参数最多 | `gate_proj` 与 `up_proj` 合并；`silu(gate) * up` 融合成一个 kernel；MoE 用分组 GEMM（[按专家分组](../transformer/moe.md#实现逐-token-与按专家分组)） |
| 最终 RMSNorm + 输出层 | → `[1, 17, 151936]` | 最宽的 GEMM | **prefill 只算最后一个位置**（见下文）；词表按列切分 |
| 采样 | `[1, 151936]` → 1 个 id | softmax、排序、随机数 | [温度、top-p、惩罚](../inference/decoding.md)在 GPU 上批量完成；约束解码在这里屏蔽不合法的 token；投机解码在这里做验证 |
| 反分词 | id → 文本 | CPU | [增量反分词](../basics/tokenization.md#流式输出与增量反分词)，处理不完整的 UTF-8 字节，流式返回 |

此外，decode 阶段每一步都要启动几百个 kernel（28 层 × 每层十几个），对小模型来说 CPU 的启动开销甚至超过 GPU 的计算时间，所以 decode 通常用 **CUDA Graphs** 把整步前向录制下来一次性重放，参见 CUDA 手册的[流、并发与 CUDA Graphs](cuda://tools/streams/)。

### prefill 时只需要最后一个位置的 logits

上面的 `mini_llm` 为了教学，对全部 17 个位置都计算了 logits。但生成时只用得到最后一个位置（训练时才需要全部位置来计算损失）。对这个模型来说，这省下的计算量相当可观：

```python
T, d, V = ids.shape[1], model.cfg.hidden_size, model.cfg.vocab_size
n_params = sum(p.numel() for p in model.parameters())                  # 共享的嵌入只数一次
body_flops = 2 * T * (n_params - V * d)                                # 28 层 Transformer
print(f"prefill {T} 个 token：主体 {body_flops / 1e9:.1f} GFLOP，"
      f"全部位置的输出层 {2 * T * d * V / 1e9:.1f} GFLOP，只算最后一个位置 {2 * d * V / 1e9:.2f} GFLOP")
```

```text title="输出"
prefill 17 个 token：主体 15.0 GFLOP，全部位置的输出层 5.3 GFLOP，只算最后一个位置 0.31 GFLOP
```

小模型的词表占比大，这一项优化能省掉 prefill 约四分之一的计算，还省下了 `[17, 151936]` 的 logits 显存（提示词很长时，这个张量可以达到几个 GB）。推理引擎里的 `LogitsProcessor` 会先按每个请求的最后位置取出隐藏状态，再做输出层。

## decode 的时间花在哪里

把 LLaMA-3-8B 的 decode 一步拆成几类算子，分别计算它们的计算量、访存量和[算术强度](../inference/estimation.md#延迟的下限)，与 H100 的屋脊点（约 295 FLOP/字节）比较。每个算子的时间下限取"算完所需时间"和"读完所需时间"中较大的一个：

```python
from estimate import H100

llama3_8b = dict(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=8)

def decode_ops(c, batch, context, wbytes=2, kvbytes=2):
    """decode 一步中各类算子的计算量（FLOP）与访存量（字节），对整个模型求和。"""
    d, dff, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = d // nh
    ops = {}

    def gemm(name, k, n, times=L):                                     # [batch, k] @ [k, n]
        flop = 2 * batch * k * n * times
        byte = (k * n * wbytes + batch * (k + n) * 2) * times         # 权重 + 输入输出激活
        ops[name] = (flop, byte)

    gemm("qkv_proj", d, (nh + 2 * nkv) * hd)
    gemm("o_proj", nh * hd, d)
    gemm("gate_up_proj", d, 2 * dff)
    gemm("down_proj", dff, d)
    gemm("lm_head", d, V, times=1)
    ops["attention"] = (4 * batch * nh * hd * context * L,            # QK^T 与 PV
                        2 * batch * nkv * hd * context * kvbytes * L)   # 读 K 和 V
    return ops

totals = {}
for batch in (1, 64):
    print(f"batch = {batch}，上下文 4096")
    print(f"{'算子':14s}{'GFLOP':>9s}{'GB':>8s}{'强度':>8s}{'时间下限 ms':>12s}")
    totals[batch] = {}
    for name, (flop, byte) in decode_ops(llama3_8b, batch, 4096).items():
        t = max(flop / (H100.tflops * 1e12), byte / (H100.bw_tbs * 1e12)) * 1e3
        totals[batch][name] = t
        print(f"{name:14s}{flop / 1e9:9.1f}{byte / 1e9:8.2f}{flop / byte:8.1f}{t:12.2f}")
    print(f"{'合计':14s}{'':25s}{sum(totals[batch].values()):12.2f}\n")
```

```text title="输出"
batch = 1，上下文 4096
算子                GFLOP      GB      强度     时间下限 ms
qkv_proj            1.6    1.61     1.0        0.48
o_proj              1.1    1.07     1.0        0.32
gate_up_proj        7.5    7.52     1.0        2.24
down_proj           3.8    3.76     1.0        1.12
lm_head             1.1    1.05     1.0        0.31
attention           2.1    0.54     4.0        0.16
合计                                             4.64

batch = 64，上下文 4096
算子                GFLOP      GB      强度     时间下限 ms
qkv_proj          103.1    1.65    62.4        0.49
o_proj             68.7    1.11    62.1        0.33
gate_up_proj      481.0    7.65    62.9        2.28
down_proj         240.5    3.83    62.7        1.14
lm_head            67.2    1.07    63.0        0.32
attention         137.4   34.36     4.0       10.26
合计                                            14.83
```

这张表是全书最重要的一张表，值得逐行读懂：

1. **batch = 1 时，所有算子的强度都远低于 295**，全部受访存限制，时间几乎全花在读权重上（MLP 占大头）。这就是 [decode 访存受限](../inference/kv-cache.md#prefill-与-decode)的具体含义。
2. **权重类算子（GEMM）的强度约等于 batch**：权重读一次，被 batch 里所有请求共用。batch 从 1 增大到 64，它们的计算量涨了 64 倍，时间几乎不变。这就是[批处理](../inference/kv-cache.md#批处理让多个请求分摊权重读取)有效的原因。
3. **注意力的强度恒等于 GQA 的分组数（32 / 8 = 4），与 batch 无关**：每个请求读的是自己的 KV Cache，没有任何共享。batch 增大，读的 KV 成比例增加。batch = 64 时，注意力占了 70% 的时间，成为新的瓶颈。
4. 所以，大 batch、长上下文下优化的主战场是 **KV Cache**：
    - 减少 KV 的量：[GQA/MLA](../transformer/attention-variants.md)（MLA 让 128 个头共享一个潜在向量，注意力强度提高几十倍）、KV Cache 量化、滑动窗口；
    - 让 KV 读得更快：分页 decode kernel、Flash-Decoding 的 split-KV 并行；
    - 让 KV 被多个请求共享：[前缀缓存](../inference/serving.md#kv-cache-管理与前缀缓存)（共享前缀的 KV 可以只读一次，如 cascade attention）。

!!! inference "推理视角"
    这张表可以直接用来回答很多面试题。比如"为什么 MoE 模型推理更依赖大 batch？"：MoE 每个专家只被 batch 中的一部分 token 使用，权重 GEMM 的强度是"每个专家分到的 token 数"而不是 batch，所以需要更大的 batch（或者专家并行把更多请求汇集到一起）才能摆脱访存瓶颈。又比如"为什么投机解码在大 batch 下收益变小？"：大 batch 下权重类算子的强度已经接近屋脊点，验证多个 token 的计算不再"免费"。

## 把整本手册串起来

所有推理优化都可以从两个根本事实推导出来：

```text
事实一：语言模型是自回归的，一次只能生成一个 token
  │
  ├─→ 生成必须串行 → 每个 token 都要一次完整前向
  │     └─→ 前向中的历史 K、V 不变 → KV Cache
  │           ├─→ 推理分成 prefill（计算密集）与 decode（访存密集）两个阶段
  │           │     ├─→ 两者互相干扰 → 分块 prefill、PD 分离
  │           │     └─→ decode 每步只算 1 个 token → 投机解码"猜几个、验一次"
  │           └─→ KV Cache 随上下文与并发线性增长，成为显存与带宽的主要消耗
  │                 ├─→ 减少 KV：GQA、MLA、KV 量化、滑动窗口
  │                 ├─→ 管理 KV：PagedAttention、前缀缓存、抢占与换出
  │                 └─→ 读取 KV：FlashAttention、Flash-Decoding
  │
事实二：decode 的每一步都要读一遍全部权重，而算术强度只有 batch 那么大
  │
  ├─→ 增大 batch 分摊权重读取 → 连续批处理（请求长度不一，必须逐步调度）
  ├─→ 减少权重字节数 → 权重量化（INT4/FP8），decode 速度约与位宽成反比
  ├─→ 权重太大放不进一张卡 → 张量并行 / 专家并行，通信成为新开销
  └─→ 小 batch 时 kernel 启动开销显著 → 算子融合、CUDA Graphs
```

每一章都是这张图上的一个节点。读推理引擎的源码或论文时，先问自己"它在这张图的哪个位置、解决了哪个瓶颈、付出了什么代价"，大部分设计就都能看懂了。

## 哪些优化会改变输出

| 类别 | 优化 | 输出 |
| --- | --- | --- |
| 数学上等价 | KV Cache、前缀缓存、分块 prefill、连续批处理、PagedAttention、FlashAttention、算子融合、CUDA Graphs、张量并行 | 与原始计算相同（浮点误差范围内） |
| 保持分布 | 投机解码（贪心时逐 token 相同，采样时分布相同） | 同上 |
| 有损 | 权重 / 激活 / KV Cache 量化、KV 淘汰（丢弃不重要的 token）、稀疏注意力、剪枝、蒸馏 | 与原模型不同，需要评测精度 |

"数学上等价"不等于"逐位相同"。浮点加法不满足结合律，不同的 kernel、不同的并行切分、甚至**不同的 batch 大小**都会改变累加顺序：

```python
with torch.no_grad():
    alone = model(ids)[0, -1]
    batched = model(ids.repeat(4, 1))[0, -1]                           # 同一条序列，放进 batch = 4 里
print(f"单独计算与在 batch 中计算的 logits 最大差异：{(alone - batched).abs().max().item():.1e}，"
      f"逐位相同：{torch.equal(alone, batched)}")
assert (alone - batched).abs().max() < 1e-3
```

```text title="输出"
单独计算与在 batch 中计算的 logits 最大差异：3.1e-05，逐位相同：False
```

在 GPU 上用 BF16 推理时，这种差异更大。当两个候选 token 的概率几乎相同时，微小的差异就会让贪心解码走上不同的路径，而且一旦分叉，后面的文本就完全不同了。所以线上服务"同一个请求、温度为 0，两次结果不同"是正常现象：请求所在的 batch 大小在变。需要严格可复现时（比如强化学习训练中推理与训练的对齐），要使用专门的 batch 不变（batch-invariant）kernel，代价是一定的性能。

!!! inference "推理视角"
    这也决定了推理优化的**测试方法**：数学上等价的优化，用"与参考实现的 logits 误差在阈值内"来验证（就像本手册对 `mini_llm` 做的那样），而不是要求生成文本逐字相同；有损的优化，要在下游任务上评测精度（困惑度、MMLU、GSM8K 等）。

!!! interview "怎么讲清楚"
    这一章就是"从请求到 token 的全链路"的标准讲法：分词与对话模板 → 嵌入（gather）→ 每层 RMSNorm、QKV 投影（prefill 是 GEMM、decode 是 GEMV）、RoPE、注意力（prefill 用 FlashAttention、decode 读 KV）、SwiGLU → 只算最后位置的 logits → GPU 上采样 → 增量反分词。关键数字：decode 时权重类算子的算术强度约等于 batch，注意力的强度等于 GQA 分组数、与 batch 无关，所以大 batch 下读 KV 成为瓶颈。最后说明哪些优化数学上等价、哪些有损，以及"等价不等于逐位相同"。

## 练习

**1. 形状推导。** 对 LLaMA-3-8B（hidden 4096、32 个头、8 个 KV 头、head_dim 128、ffn 14336、32 层、词表 128256），batch = 8 的请求每个都已有 1000 个 token 的上下文，现在 decode 一步。写出 `q_proj` 输出、第 0 层 K Cache（更新后）、注意力 scores、`gate_proj` 输出、`lm_head` 输出的形状。

??? success "参考答案"
    - `q_proj` 输出：`[8, 1, 4096]`，变形后为 `[8, 32, 1, 128]`；
    - 第 0 层 K Cache：`[8, 8, 1001, 128]`；
    - scores：`[8, 32, 1, 1001]`；
    - `gate_proj` 输出：`[8, 1, 14336]`；
    - `lm_head` 输出：`[8, 1, 128256]`。

    实际的推理引擎不会用 `[B, T, ...]` 这种带填充的布局，而是把所有请求的 token 拼成一维 `[num_tokens, hidden]`（这里 num_tokens = 8），用每个请求的序列长度和块表来描述 KV 的位置。这样 prefill 和 decode 的 token 可以混在同一个 batch 里。

**2. 用屋顶线表做决策。** 在上面的表中，如果把 KV Cache 量化成 FP8，batch = 64 时 decode 一步的时间下限大约变成多少？如果改为把权重量化成 INT4 呢？哪个收益大？

??? success "参考答案"
    ```python
    fp8_kv = sum(totals[64].values()) - totals[64]["attention"] / 2
    int4_w = sum(totals[64].values()) - sum(t for n, t in totals[64].items() if n != "attention") * 0.75
    print(f"原始 {sum(totals[64].values()):.1f} ms，KV FP8 约 {fp8_kv:.1f} ms，权重 INT4 约 {int4_w:.1f} ms")
    ```

    ```text title="输出"
    原始 14.8 ms，KV FP8 约 9.7 ms，权重 INT4 约 11.4 ms
    ```

    KV FP8 让注意力的读取量减半（10.3 → 5.1 ms）；权重 INT4 让权重类算子的读取量变为 1/4（4.6 → 1.1 ms），但它们本来就只占 4.6 ms。所以**大 batch、长上下文时，KV 量化的收益更大；小 batch 时，权重量化的收益更大**。（这是忽略了反量化开销的粗略估计；另外 batch = 64 时权重 GEMM 的强度已经是 63，INT4 后强度变为 4 倍，仍低于屋脊点，估计依然成立。）

## 小结

- [x] 除注意力外，每个模块都逐 token 独立计算；prefill 是 GEMM，decode 是 GEMV。
- [x] 每一站都有对应的推理优化：融合、合并投影、分页 KV、FlashAttention、只算最后位置的 logits、GPU 采样、增量反分词、CUDA Graphs。
- [x] decode 的权重类算子强度约等于 batch，注意力强度等于 GQA 分组数且与 batch 无关；大 batch 下 KV Cache 是瓶颈。
- [x] 所有推理优化都可以追溯到"自回归串行"和"decode 访存受限"两个事实。
- [x] 数学上等价的优化也不保证逐位相同，batch 大小会改变浮点结果；测试时用误差阈值，有损优化要评测精度。
