# 主流模型架构巡礼

<p class="lead">LLaMA、Qwen、DeepSeek、Gemma、gpt-oss……名字很多，但骨架几乎都一样：Pre-Norm 的解码器、RoPE、SwiGLU、GQA，再加上可选的 MoE。它们的区别集中在少数几个"旋钮"上。这一章先教你读懂 <code>config.json</code>，再用 meta 设备精确数出十个主流模型的参数量和 KV Cache，最后逐个看各家的改动，以及每个改动给推理引擎带来的影响。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 拿到一个新模型的 `config.json`，你能在一分钟内说出它的参数量、激活参数量、每个 token 的 KV Cache 大小吗？
    2. Gemma 3、gpt-oss、Qwen3-Next 分别用什么方法减少长上下文的 KV Cache？
    3. 为什么 Qwen2.5-7B 不能在 8 张卡上做张量并行？
    4. 为什么 DeepSeek 系列模型的部署通常用 DP Attention + 专家并行，而不是纯张量并行？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 能：参数量按嵌入 + 每层注意力和 FFN（MoE 按专家数）算；激活参数只算每个 token 实际经过的专家；每 token 的 KV = 2 × 层数 × KV 头数 × 头维 × 字节数（MLA 按潜向量维度，滑窗层和线性层另算）。
    2. Gemma 3：局部（滑动窗口 1024）与全局注意力 5 : 1 交替，大部分层的 KV 不随上下文增长；gpt-oss：全注意力层与窗口 128 的滑窗层交替，并加入注意力汇聚；Qwen3-Next：每 4 层里 3 层是 Gated DeltaNet（线性注意力，固定大小的状态），1 层是门控的标准注意力。
    3. 它有 28 个注意力头（4 个 KV 头），28 不能被 8 整除，头没法平均分到 8 张卡上（KV 头更少，还得复制）。
    4. MLA 的潜向量被所有头共享，张量并行按头切不开它，每张卡都得存完整的一份 KV，卡越多浪费越多；所以注意力部分用数据并行（DP Attention，各卡服务不同的请求），MoE 部分用专家并行。

## 读懂 config.json

这是本手册一直在用的 Qwen3-0.6B 的配置（节选重要字段）：

```text
"hidden_size": 1024,                d：残差流的宽度
"intermediate_size": 3072,          FFN 中间层宽度，3 d
"num_hidden_layers": 28,            层数 L
"num_attention_heads": 16,          query 头数
"num_key_value_heads": 8,           KV 头数：GQA，每 2 个 query 头共享一组 K、V
"head_dim": 128,                    每个头的维度；注意 16 × 128 = 2048 ≠ d，头维是单独配置的
"vocab_size": 151936,               词表大小
"tie_word_embeddings": true,        输出层与嵌入层共享权重
"rope_theta": 1000000,              RoPE 的基数，越大越适合长上下文
"max_position_embeddings": 40960,   训练时支持的最大上下文
"rms_norm_eps": 1e-06,              RMSNorm 的 ε
"hidden_act": "silu",               SwiGLU 中的激活函数
"torch_dtype": "bfloat16"           发布权重的精度
```

config.json 里看不出来的结构差异要读模型代码才知道：Qwen3 的 q、k 在 RoPE 之前各做一次 RMSNorm（QK-Norm），而且去掉了 q、k、v 投影的偏置（Qwen2 有）。

每个字段都对应着手册中的一章，也对应着推理时的一项成本：

| 字段 | 章节 | 推理时决定了什么 |
| --- | --- | --- |
| `hidden_size`、`intermediate_size`、`num_hidden_layers` | [组装大模型](../transformer/build-llm.md#各部分的参数量) | 参数量 → 权重显存、decode 的权重读取量 |
| `num_attention_heads`、`num_key_value_heads`、`head_dim` | [注意力变体](../transformer/attention-variants.md) | KV Cache 大小；张量并行的切分方式与上限 |
| `vocab_size`、`tie_word_embeddings` | [嵌入层与输出层](../transformer/embedding.md) | 输出层的计算与 logits 的显存；采样的开销 |
| `rope_theta`、`rope_scaling`、`max_position_embeddings` | [位置编码](../transformer/position.md#长上下文扩展) | 支持的上下文长度；注意力 kernel 需要实现的 RoPE 变体 |
| `num_experts`、`num_experts_per_tok`、`moe_intermediate_size` | [MoE](../transformer/moe.md) | 总参数与激活参数；是否需要专家并行 |
| `sliding_window`、`layer_types` | 本章 | 哪些层的 KV Cache 随上下文增长 |
| `torch_dtype`、`quantization_config` | [量化](../inference/quantization.md) | 权重字节数、需要的 GEMM kernel |

在推理引擎里，这些字段决定了加载哪个模型类、分配多少 KV Cache、选择哪个注意力后端。一个模型能否被推理引擎支持，很大程度上就是看这些字段组合出的结构有没有对应的 kernel。

## 十个模型的体检表

用 transformers 自带的配置类在 meta 设备上构建模型，精确地数出总参数与每个 token 激活的参数，再计算 KV Cache 的大小。对于用了滑动窗口的层，KV Cache 最多保存窗口内的 token；线性注意力层只有固定大小的状态，不随上下文增长：

```python
import torch
import transformers as T
from transformers import AutoModelForCausalLM

configs = {
    "LLaMA-3-8B": T.LlamaConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                                num_attention_heads=32, num_key_value_heads=8),
    "Qwen2.5-7B": T.Qwen2Config(vocab_size=152064, hidden_size=3584, intermediate_size=18944, num_hidden_layers=28,
                                num_attention_heads=28, num_key_value_heads=4),
    "Qwen3-8B": T.Qwen3Config(vocab_size=151936, hidden_size=4096, intermediate_size=12288, num_hidden_layers=36,
                              num_attention_heads=32, num_key_value_heads=8, head_dim=128),
    "Gemma-3-27B": T.Gemma3TextConfig(vocab_size=262208, hidden_size=5376, intermediate_size=21504, num_hidden_layers=62,
                                      num_attention_heads=32, num_key_value_heads=16, head_dim=128, sliding_window=1024),
    "Mixtral-8x7B": T.MixtralConfig(),                      # 以下几个配置类的默认值就是对应模型的配置
    "Qwen3-30B-A3B": T.Qwen3MoeConfig(hidden_size=2048, num_hidden_layers=48, head_dim=128, moe_intermediate_size=768),
    "Qwen3-235B-A22B": T.Qwen3MoeConfig(hidden_size=4096, num_hidden_layers=94, num_attention_heads=64, head_dim=128,
                                        moe_intermediate_size=1536),
    "gpt-oss-120b": T.GptOssConfig(),
    "DeepSeek-V3": T.DeepseekV3Config(),
    "Qwen3-Next-80B-A3B": T.Qwen3NextConfig(),
}

def count(cfg):
    """在 meta 设备上构建模型，返回（总参数，每个 token 激活的参数）。"""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    routed = sum(p.numel() for n, p in model.named_parameters() if ".experts." in n)   # 路由专家（不含共享专家）
    if not routed:
        return total, total
    n_experts = getattr(cfg, "n_routed_experts", None) or getattr(cfg, "num_local_experts", None) or cfg.num_experts
    return total, total - routed * (1 - cfg.num_experts_per_tok / n_experts)

def kv_bytes(cfg, context, bytes_per_elem=2):
    """一个请求在给定上下文长度下的 KV Cache 字节数。"""
    if hasattr(cfg, "kv_lora_rank"):                                   # MLA：每层只存潜在向量和 RoPE 部分
        per_token_layer = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    else:
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        per_token_layer = 2 * cfg.num_key_value_heads * head_dim
    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
    window = getattr(cfg, "sliding_window", None) or context
    tokens = {"full_attention": context, "sliding_attention": min(context, window),
              "linear_attention": 0}                                    # 线性注意力只有固定大小的状态
    return sum(tokens[t] for t in layer_types) * per_token_layer * bytes_per_elem

ctx = 131072
print(f"{'模型':20s}{'总参数':>9s}{'激活':>9s}{'KV/token':>10s}{'128K 上下文的 KV':>18s}")
results = {}
for name, cfg in configs.items():
    total, active = count(cfg)
    results[name] = (total, active)
    print(f"{name:20s}{total / 1e9:8.1f}B{active / 1e9:8.1f}B{kv_bytes(cfg, ctx) / ctx / 1024:8.0f}KB"
          f"{kv_bytes(cfg, ctx) / 1e9:14.1f} GB")
assert round(results["DeepSeek-V3"][0] / 1e9) == 671 and round(results["gpt-oss-120b"][0] / 1e9, 2) == 116.83
```

```text title="输出"
模型                        总参数       激活  KV/token      128K 上下文的 KV
LLaMA-3-8B               8.0B     8.0B     128KB          17.2 GB
Qwen2.5-7B               7.6B     7.6B      56KB           7.5 GB
Qwen3-8B                 8.2B     8.2B     144KB          19.3 GB
Gemma-3-27B             27.0B    27.0B      83KB          11.2 GB
Mixtral-8x7B            46.7B    12.9B     128KB          17.2 GB
Qwen3-30B-A3B           30.5B     3.4B      96KB          12.9 GB
Qwen3-235B-A22B        235.1B    22.2B     188KB          25.2 GB
gpt-oss-120b           116.8B     5.7B      36KB           4.8 GB
DeepSeek-V3            671.0B    37.6B      69KB           9.2 GB
Qwen3-Next-80B-A3B      79.7B     3.9B      24KB           3.2 GB
```

总参数都与官方公布的数字一致。激活参数比官方数字略大一些（比如 gpt-oss-120b 官方为 5.1B），是因为这里把嵌入表也算进去了。嵌入层只是查表，不参与计算，有的官方统计不计入它：gpt-oss 的嵌入表 201088 × 2880 ≈ 0.58B，正好是差值。

把这十个模型画在一张图上，两条主线（MoE 让激活参数远小于总参数；GQA、MLA、滑动窗口、线性注意力让 KV 越来越小）一眼就能看出来：

<div class="aig-widget" data-widget="model-map"></div>

从这张表能读出这几年模型设计的两条主线：

1. **MoE 越来越稀疏**：激活比例从 Mixtral 的 28%，到 DeepSeek-V3 的 5.6%，再到 gpt-oss 和 Qwen3-Next 的 5% 左右。计算量按激活参数算，显存按总参数算。
2. **KV Cache 越来越省**：同样是 128K 上下文，全注意力的 Qwen3-8B 要 19 GB；DeepSeek-V3 用 MLA，参数量是它的 80 倍，KV 却只要一半；gpt-oss 和 Qwen3-Next 通过让一部分层"不保存全部历史"，把 KV 压到了 3～5 GB。

下面逐个看各家的具体设计。

## LLaMA：标准模板

LLaMA（2023）确立了今天几乎所有开源模型的模板：**Pre-Norm + RMSNorm、SwiGLU、RoPE、无偏置**。之后的演进：

- LLaMA 2：70B 用上了 GQA；
- LLaMA 3：所有尺寸都用 GQA（8 个 KV 头），词表从 32K 扩大到 128K（同样的文本 token 数更少，但输出层更大），`rope_theta` 调到 500000；
- LLaMA 3.1：用 `rope_scaling`（`rope_type: llama3`，对不同频率分段缩放）把上下文扩展到 128K。

!!! inference "推理视角"
    KV 头数决定了张量并行的"舒适上限"。LLaMA-3-8B 有 8 个 KV 头，TP = 8 时每张卡恰好一个 KV 头；如果 TP 大于 KV 头数，KV 头就必须在多张卡上复制，KV Cache 的总显存随之翻倍。

## Qwen：从偏置到 QK-Norm，再到混合注意力

- **Qwen2 / 2.5**：在 LLaMA 模板上给 Q、K、V 投影加了偏置（`mini_llm` 的 `attention_bias`），小尺寸模型共享嵌入权重，`rope_theta` 为 1e6。
- **Qwen3**：去掉 QKV 偏置，加上 **QK-Norm**：在 RoPE 之前对每个头的 q 和 k 各做一次 RMSNorm，防止注意力分数的数值过大，让训练更稳定。MoE 版本（30B-A3B、235B-A22B）用 128 个专家、每个 token 选 8 个，没有共享专家。
- **Qwen3-Next**：每 4 层中 3 层用 **Gated DeltaNet（线性注意力）**，1 层用门控的标准注意力；512 个专家选 10 个，外加 1 个共享专家；带有 MTP 模块。

线性注意力把"对所有历史 token 做注意力"换成一个固定大小的状态矩阵，每来一个 token 就更新一次状态。它的 decode 代价与上下文长度无关，但表达能力弱于标准注意力，所以通常与标准注意力层混合使用。

!!! inference "推理视角"
    - QK-Norm 需要融合进 QKV 投影之后、RoPE 之前的 kernel 中；
    - 混合架构让推理引擎的缓存管理变复杂：标准注意力层有按 token 增长、可以分页的 KV Cache，线性注意力层则是每个请求一份固定大小的状态。前缀缓存也变难了：线性注意力的状态只对应"某个确切长度的前缀"，想复用就得在特定位置保存状态的快照。vLLM、SGLang 都为此单独实现了这类状态的缓存管理。
    - Qwen2.5-7B 有 28 个 query 头，不能被 8 整除，所以不能做 8 卡张量并行（推理引擎要求头数能被 TP 整除）。

## Mixtral：MoE 进入主流

Mixtral-8x7B（2023）在 Mistral-7B 的骨架上，把每层的 FFN 换成 8 个专家、每个 token 选 2 个。总参数 46.7B，激活 12.9B：推理速度接近 13B 的稠密模型，效果接近更大的模型。它是开源社区第一个被广泛使用的 MoE 模型，推动了 fused MoE kernel 的普及。

## DeepSeek-V3：MLA + 细粒度 MoE

DeepSeek-V3（2024 年底）集中了多项影响推理的设计：

- **[MLA](../transformer/attention-variants.md#mla多头潜在注意力)**：128 个头共享一个 512 维的 KV 潜在向量（外加 64 维的 RoPE 部分），每层每个 token 只存 576 个数；
- **细粒度 MoE**：61 层中前 3 层是稠密 FFN，其余每层 256 个路由专家选 8 个，外加 1 个共享专家；路由用 sigmoid 打分，并通过给每个专家加一个动态调整的偏置来做负载均衡（不依赖辅助损失）；
- **MTP**：额外预测下一个之后的 token，推理时可以作为投机解码的草稿；
- 用 FP8 混合精度训练，发布的权重就是 FP8（按 128×128 的块量化）。

后续的 DeepSeek-R1 沿用了同样的结构；DeepSeek-V3.2 在此基础上引入了稀疏注意力（DSA），用一个轻量的索引器为每个 query 挑选最相关的一部分 token 做注意力。Kimi K2 也基本沿用了 V3 的结构（见练习）。

!!! inference "推理视角"
    - MLA 在 decode 时使用权重吸收，注意力变成"128 个 query 头对同一份 576 维的潜在 KV"，强度很高，需要专门的 kernel（如 FlashMLA）；
    - MLA 的潜在 KV 只有一份，无法按头切分，做张量并行时每张卡都要存完整的 KV，TP 越大浪费越多。所以常用 **DP Attention**：注意力部分按请求做数据并行（每张卡处理不同的请求，各存各的 KV），MoE 部分做专家并行；
    - 256 个专家分布在很多卡上，每层两次 all-to-all，需要 DeepEP 这类通信库，以及冗余专家、负载均衡放置（EPLB）；
    - PD 分离与大规模 EP 结合（prefill 和 decode 用不同的并行规模）是目前部署这类模型的主流方案。

## Gemma：大词表、软截断、滑动窗口

Gemma 系列的特点：

- 词表 256K，`head_dim` 为 256，嵌入向量在输入时乘以 $\sqrt{d}$，FFN 用 GeGLU（GELU 版的门控）；
- 每个子层前后各有一个 RMSNorm（"三明治"式的 Pre + Post Norm）；
- **Gemma 2**：局部注意力（滑动窗口 4096）与全局注意力逐层交替；对注意力分数和最终 logits 做**软截断（soft-capping）**：$\text{cap} \cdot \tanh(x / \text{cap})$，把数值平滑地限制在 $(-\text{cap}, \text{cap})$ 内；
- **Gemma 3**：局部与全局之比提高到 5 : 1，窗口缩小到 1024，用 QK-Norm 替代了软截断，上下文 128K。

软截断的效果：

```pycon
>>> import torch
>>> x = torch.tensor([10.0, 50.0, 100.0, 1000.0])
>>> [round(v, 2) for v in (30 * torch.tanh(x / 30)).tolist()]
[9.65, 27.93, 29.92, 30.0]
```

小的值几乎不变，大的值被压到 30 以内。

滑动窗口对 KV Cache 的效果非常显著。Gemma-3-27B 的 62 层中只有 10 层是全局注意力，另外 52 层只保存最近 1024 个 token：

```python
g = configs["Gemma-3-27B"]
all_global = ctx * g.num_hidden_layers * 2 * g.num_key_value_heads * g.head_dim * 2
print(f"128K 上下文：实际 {kv_bytes(g, ctx) / 1e9:.1f} GB，若全部是全局注意力则为 {all_global / 1e9:.1f} GB")
```

```text title="输出"
128K 上下文：实际 11.2 GB，若全部是全局注意力则为 66.6 GB
```

!!! inference "推理视角"
    软截断改变了注意力分数的计算方式，注意力 kernel 必须原生支持它（FlashAttention 2 和 FlashInfer 都专门加了这个选项），否则只能退回到物化 scores 的慢速实现。`head_dim` 为 256 同样需要 kernel 支持。滑动窗口层的 KV 可以循环覆盖，但与分页 KV、前缀缓存结合时，推理引擎要为不同类型的层分别管理缓存。

## gpt-oss：原生 4 位的 MoE

OpenAI 的 gpt-oss（2025）有 120b 和 20b 两个尺寸：

- MoE：120b 有 36 层、每层 128 个专家选 4 个；
- 注意力：全注意力层与窗口仅 128 个 token 的滑动窗口层交替；64 个 query 头、8 个 KV 头；
- **注意力汇聚（attention sink）**：每个头有一个可学习的标量，作为一个"虚拟 token"参与 softmax 的归一化，但不对应任何 value。这样一个头可以"什么都不看"，而不必像[注意力一章](../transformer/attention.md#真实模型里的注意力注意力汇聚)那样把注意力堆到第一个 token 上；
- 专家权重以 **MXFP4**（4 位浮点，每 32 个数共享一个缩放因子）发布，120b 可以放进一张 80 GB 的 GPU；
- 使用新的对话格式 harmony。

注意力汇聚的计算：

```pycon
>>> scores = torch.tensor([2.0, 1.0, 0.5, 0.2])   # 当前 query 对 4 个 key 的分数
>>> sink = torch.tensor([3.0])                     # 这个头学到的汇聚分数
>>> p = torch.cat([scores, sink]).softmax(-1)[:-1] # 汇聚参与归一化，然后丢掉
>>> [round(v, 3) for v in p.tolist()], round(p.sum().item(), 3)
([0.223, 0.082, 0.05, 0.037], 0.393)
```

这个头对真实 token 的注意力之和只有 0.39，其余的"注意力"被汇聚吸收了。

!!! inference "推理视角"
    注意力汇聚同样需要 kernel 支持：在 online softmax 的最后，把汇聚项加进分母即可，开销很小。MXFP4 的 MoE 权重在 Blackwell 上有原生的 Tensor Core 支持，在 Hopper 上需要先在 kernel 里反量化。

## 趋势小结

| 方向 | 代表 | 推理引擎要跟上的 |
| --- | --- | --- |
| 减少 KV Cache | GQA → MLA；局部/全局混合（Gemma 3、gpt-oss）；线性注意力混合（Qwen3-Next）；稀疏注意力（DeepSeek-V3.2） | 新的注意力 kernel；按层类型分别管理的缓存；前缀缓存的新实现 |
| 更稀疏的 MoE | 专家数 8 → 128 → 256 → 512，激活比例降到 5% 以下 | fused MoE、专家并行、all-to-all 通信、负载均衡 |
| 原生低精度 | FP8 训练（DeepSeek-V3）、MXFP4 发布（gpt-oss） | 对应格式的 GEMM kernel |
| 训练稳定性技巧 | QK-Norm、软截断、注意力汇聚、门控注意力 | kernel 里的额外计算 |
| 为推理而设计 | MTP（可作为投机解码的草稿）、更少的注意力头 | 投机解码框架 |

模型架构和推理系统是相互塑造的：模型越来越多地为了推理效率而设计，推理引擎则要不断支持新的结构。读一个新模型的论文或配置时，按本章的方法做一遍"体检"：参数量、激活参数、KV Cache、注意力类型、特殊算子，你就能判断它对推理系统意味着什么。

!!! interview "怎么讲清楚"
    讲"拿到一个新模型的 `config.json` 你看什么"：一分钟内算出参数量、激活参数、每 token 的 KV；再看注意力类型（GQA、MLA、滑窗、线性注意力混合、稀疏注意力）、MoE 配置（专家数、top-k、共享专家）、RoPE 与上下文长度、特殊结构（软截断、注意力汇聚、QK-Norm）。然后推出部署上的影响：KV 头数少于张量并行的卡数要复制 KV，MLA 模型用 DP Attention + 专家并行，线性注意力要有按请求的状态池。

## 练习

**1. Kimi K2。** Kimi K2 基本沿用了 DeepSeek-V3 的结构，主要改动是：路由专家从 256 个增加到 384 个，注意力头从 128 个减少到 64 个，只有第一层是稠密 FFN，词表 163840。数一数它的总参数、激活参数和每个 token 的 KV Cache，并思考：为什么减少注意力头对推理有利？

??? success "参考答案"
    ```python
    k2 = T.DeepseekV3Config(vocab_size=163840, num_attention_heads=64, n_routed_experts=384, first_k_dense_replace=1)
    total, active = count(k2)
    print(f"总参数 {total / 1e9:.0f}B，激活 {active / 1e9:.1f}B，KV {kv_bytes(k2, ctx) / ctx / 1024:.0f} KB/token")
    ```

    ```text title="输出"
    总参数 1026B，激活 32.9B，KV 69 KB/token
    ```

    总参数约 1T，激活约 32B。KV Cache 与 DeepSeek-V3 完全相同：MLA 存的是所有头共享的潜在向量，与头数无关。

    减少头数的好处在注意力的**计算**上：MLA 在 decode 时，每个头都要与 576 维的潜在 KV 做点积，注意力的计算量与头数成正比；prefill 长上下文时，注意力的 $O(n^2)$ 部分也与头数成正比。头数减半，长上下文下的注意力开销就减半。这是一个为推理效率而做的设计选择。

**2. 部署方案。** 你要在 8 张 H100（每张 80 GB）上部署 Qwen3-235B-A22B（BF16）。权重放得下吗？如果换成 FP8 呢？除了张量并行，你还会考虑什么并行方式？

??? success "参考答案"
    BF16 权重约 235 × 2 = 470 GB，8 张卡共 640 GB，放得下，但只剩约 170 GB 给 KV Cache 和激活，而这个模型 128K 上下文的 KV 就要 25 GB，并发会受到很大限制。FP8 权重约 235 GB，剩余显存宽裕得多，decode 读取的权重量也减半。

    它有 4 个 KV 头，TP = 8 时每个 KV 头要复制到两张卡上，KV Cache 的显存效率减半。更好的方案是注意力部分用 TP = 4 或者 DP Attention，MoE 部分用专家并行（128 个专家分到 8 张卡，每张 16 个）。具体选择要结合负载实测。

## 小结

- [x] `config.json` 的每个字段都对应一章原理和一项推理成本；看配置就能估算参数、激活参数和 KV Cache。
- [x] 主流模型共用 Pre-Norm + RoPE + SwiGLU + GQA 的骨架，差异集中在注意力类型、MoE 配置和少数稳定性技巧上。
- [x] 减少 KV 的路线：GQA → MLA → 局部/全局混合 → 线性注意力混合 → 稀疏注意力。
- [x] MoE 越来越稀疏，推理需要 fused MoE、专家并行和负载均衡。
- [x] 每一种新结构（软截断、注意力汇聚、MLA、线性注意力）都要求推理引擎有对应的 kernel 和缓存管理。
