# （九）解码层与整模型

<p class="lead">零件全有了。这一章把它们拼起来：一个解码层 = 归一化 → 混合器（全注意力或线性注意力）→ 残差 → 归一化 → 前馈 → 残差；整模型 = 嵌入 → 24 层 → 归一化 → 输出层。从这一章起，<strong>前向里没有一行 Hugging Face 的代码</strong>，连 tokenizer 都换成 <code>tokenizers</code> 库直接读 tokenizer.json。最后用它生成第 0 步那句话，逐 token 一致。</p>

!!! abstract "学完这章你能"
    1. 说清一个解码层和整模型的结构：Pre-Norm、混合器、前馈网络、两条残差。
    2. 把权重文件里的名字一一对到自己写的模块上，整模型和 HF 的 logits 对到 1e-5。
    3. 用整模型往里看：门在真实输入下开多大，前馈网络里的一个神经元存着什么。

!!! note "这一章跑什么"
    `ch09_model.py`，约 30 秒（整模型和 HF 对 logits，再生成 40 个 token——还没有 KV cache，所以慢）；`ch09_inside.py`，约 10 秒（往整模型里面看）。

@@files@@

## 解码层

```python title="qwen/layer.py"
"""解码层：一层 = 一个 token 混合器（全注意力或线性注意力）+ 一个前馈网络，各带一个前置 RMSNorm 和一条残差。"""
import torch.nn as nn

from qwen.attention import Attention
from qwen.deltanet import GatedDeltaNet
from qwen.mlp import MLP
from qwen.norm import RMSNorm


class DecoderLayer(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.kind = kind                                                   # "full_attention" 或 "linear_attention"
        if kind == "full_attention":
            self.self_attn = Attention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin) if self.kind == "full_attention" else self.linear_attn(h)
        x = x + h                                                          # 残差：混合器只往残差流里"加"东西
        return x + self.mlp(self.post_attention_layernorm(x))
```

残差流是理解 Transformer 的主线：`x` 从嵌入层出来，经过 24 层，每层只往上**加**两样东西（混合器的输出、前馈的输出），从不覆盖。所以 `x` 的尺度会越走越大——这就是每个子层前面要有 RMSNorm（Pre-Norm）的原因：子层看到的是归一化后的副本，残差流本身一路畅通。

混合器两种，按 `layer_types` 决定：`linear_attention` 层的属性叫 `linear_attn`，`full_attention` 层的叫 `self_attn`——名字故意和权重文件里一致，`load_state_dict` 才能直接对上。

## 整模型

```python title="qwen/model.py"
"""整模型：嵌入 -> 24 个解码层 -> 最后的 RMSNorm -> 输出层。从这个文件起，前向里没有一行 Hugging Face 的代码。"""
import torch
import torch.nn as nn

from qwen.embedding import Embedding, LMHead
from qwen.layer import DecoderLayer
from qwen.norm import RMSNorm
from qwen.rope import rope_tables
from qwen.weights import Config, find_model, load_weights


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, kind) for kind in cfg.layer_types)
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = LMHead(self.embed_tokens.weight)                       # 和嵌入层共用权重

    @classmethod
    def from_pretrained(cls, path=None, dtype=torch.float32):
        path = path or find_model()
        model = cls(Config.load(path))
        missing, unexpected = model.load_state_dict(load_weights(path, dtype=dtype), strict=False, assign=True)   # assign=True：不复制，直接接管
        assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)   # 文件里没有 lm_head：它就是嵌入层
        model.lm_head.weight = model.embed_tokens.weight                                  # 接管之后嵌入表换了对象，输出层重新指过去
        return model.to(dtype).eval()

    @torch.no_grad()
    def forward(self, ids):
        """ids: [B, T] -> logits [B, T, 词表]"""
        x = self.embed_tokens(ids)
        cos, sin = rope_tables(self.cfg, torch.arange(ids.shape[1], device=ids.device))
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))
```

`from_pretrained` 里那句 `assert`：权重文件里没有 `lm_head.weight`（共用嵌入表），所以 `strict=False`，但除此之外不许缺、不许多——这是防止"名字写错了一个、模块悄悄用了随机初始化"的保险。

## 全书唯一一次和 HF 对，然后扔掉 HF

```python title="ch09_model.py" ci="loose"
"""第 9 章：整模型拼起来；全书唯一一次和 HF 对 logits；然后只用 tokenizers 库和自己的模型生成第 0 步那句话"""
import os
import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
model = Model.from_pretrained(path)
print(f"自己的模型：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M 参数，{len(model.layers)} 层")

hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 1. 全书唯一一次和 HF 对：整模型的 logits
tok = AutoTokenizer.from_pretrained(path)
for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 40)):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = hf(ids).logits
    print(f"{ids.shape[1]:3d} 个 token：logits 最大差 {diff(model(ids), ref):.1e}，下一个 token 的 top-3 一样：{model(ids)[0, -1].topk(3).indices.tolist() == ref[0, -1].topk(3).indices.tolist()}")
del hf

tk = Tokenizer.from_file(str(path / "tokenizer.json"))              # 2. 不要 transformers 了：tokenizer.json 用 tokenizers 库直接读
prompt = "<|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # 第 12 章手写这个模板
ids = torch.tensor([tk.encode(prompt).ids])
print(f"提示词 {ids.shape[1]} 个 token，和第 0 步一样")
STOP = {tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>")}

t0 = time.time()                                                    # 3. 第 0 步的循环，换成自己的模型
out = ids
for _ in range(40):
    nxt = model(out)[0, -1].argmax()
    out = torch.cat([out, nxt.view(1, 1)], dim=1)
    if nxt.item() in STOP:
        break
new = out[0, ids.shape[1]:].tolist()
print(f"生成 {len(new)} 个 token，{time.time() - t0:.1f} 秒（每步都重算整段，第 10 章解决）")
print(tk.decode(new))
```

```text title="输出"
自己的模型：752.4M 参数，24 层
  6 个 token：logits 最大差 1.3e-05，下一个 token 的 top-3 一样：True
279 个 token：logits 最大差 2.6e-05，下一个 token 的 top-3 一样：True
提示词 17 个 token，和第 0 步一样
生成 38 个 token，18.7 秒（每步都重算整段，第 10 章解决）
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。
```

前面八章没有和任何参考实现对过答案，这里一次对上：整段前向和 Hugging Face 的 logits 差 1e-5 量级（来源只有两处：注意力矩阵乘的顺序、线性注意力分块求和的顺序），下一个 token 的 top-3 完全一样，生成的句子和第 0 步一字不差。这说明前面每一个零件都写对了——如果哪一章写错了，这里会差出 0.1 以上。

## 拼好以后往里看：门和前馈网络在做什么

第 6 章讲门、第 4 章讲前馈网络时，都有一件事没法看：那时没有整模型，喂进去的不是真实的中间结果。现在可以了：用钩子（`register_forward_pre_hook`）把某一层真正收到的输入抓下来，再按那一章的公式手算一遍。

```python title="ch09_inside.py" ci="loose"
"""第 9 章（二）：整模型拼好了，往里看两件事：门在真实输入下开多大、前馈网络里存着什么"""
import math
import os

import torch
from tokenizers import Tokenizer

from qwen.model import Model
from qwen.rope import apply_rope, rope_tables
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path).eval()
tk = Tokenizer.from_file(str(path / "tokenizer.json"))

text = "推理优化的核心是减少访存。大模型在解码阶段每生成一个词，都要读取全部的权重和缓存。"   # 和大模型原理手册注意力一章同一句话
ids = torch.tensor([tk.encode(text).ids])
T = ids.shape[1]
inputs = {}                                                          # 1. 抓 6 个全注意力层的真实输入
for i, layer in enumerate(model.layers):
    if hasattr(layer, "self_attn"):
        layer.self_attn.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args[0]))
with torch.no_grad():
    model(ids)
    cos, sin = rope_tables(model.cfg, torch.arange(T))
    print(f"{T} 个 token。每个全注意力层：门的均值、门值小于 0.1 的比例、后一半 query 平均分给第一个 token 的注意力")
    for i, x in inputs.items():
        a = model.layers[i].self_attn
        q, gate = a.q_proj(x).view(1, T, a.n_head, 2 * a.d).chunk(2, dim=-1)
        k = a.k_proj(x).view(1, T, a.n_kv, a.d)
        q, k = apply_rope(a.q_norm(q), cos, sin), apply_rope(a.k_norm(k), cos, sin)
        q, k = q.transpose(1, 2), k.transpose(1, 2).repeat_interleave(a.n_head // a.n_kv, dim=1)
        s = (q @ k.transpose(-1, -2) / math.sqrt(a.d)).masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        first = torch.softmax(s, dim=-1)[0, :, T // 2:, 0].mean()
        g = torch.sigmoid(gate)
        print(f"  第 {i:2d} 层：门均值 {g.mean():.2f}，小于 0.1 的 {(g < 0.1).float().mean():.0%}，给第一个 token {first:.0%}")

E = model.embed_tokens.weight                                       # 2. 前馈网络里存着什么：输出层和嵌入表共用这份权重
scale = 1 + model.norm.weight                                        # 最后那个零中心 RMSNorm 的逐维缩放
ids = torch.tensor([tk.encode("中国的首都是").ids])
target = tk.encode("北京").ids[0]
acts = {}
for i, layer in enumerate(model.layers):                             # 抓每层前馈网络中间那 3584 个神经元在最后一个位置的激活
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, args, i=i: acts.__setitem__(i, args[0][0, -1]))
with torch.no_grad():
    p = model(ids)[0, -1].softmax(-1)
    print("「中国的首都是」的下一个词，排前三的是", "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p.topk(3))))
    u = E[target] * scale                                            # "北京"在输出层上的方向
    best = []
    for L, act in acts.items():                                      # 每个神经元的直接贡献 = 激活 × 它写回残差流的那一列在"北京"方向上的分量
        c = act * (model.layers[L].mlp.down_proj.weight.T @ u)
        j = c.argmax().item()
        best.append((c[j].item(), L, j, act[j].item()))
    best.sort(reverse=True)
    for c, L, j, act in best[:2]:
        col = model.layers[L].mlp.down_proj.weight[:, j] * scale     # 这个神经元写回残差流的那一列，按输出层读成词
        words = list(dict.fromkeys(tk.decode([t]).strip() for t in (E @ col).topk(12).indices.tolist()))[:8]
        print(f"  贡献最大的之一：第 {L} 层第 {j} 号神经元，激活 {act:+.2f}；它写回的方向最像这些词：{' '.join(words)}")
    L, j = best[0][1], best[0][2]
    h = model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, args: (args[0].index_fill(-1, torch.tensor([j]), 0.0),))
    p2 = model(ids)[0, -1].softmax(-1)
    top = "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p2.topk(3)))
    print(f"  只把第 {L} 层第 {j} 号神经元关掉：P(北京) = {p2[target]:.1%}，排前三的变成 {top}")
    h.remove()
```

```text title="输出"
25 个 token。每个全注意力层：门的均值、门值小于 0.1 的比例、后一半 query 平均分给第一个 token 的注意力
  第  3 层：门均值 0.07，小于 0.1 的 77%，给第一个 token 2%
  第  7 层：门均值 0.17，小于 0.1 的 33%，给第一个 token 12%
  第 11 层：门均值 0.14，小于 0.1 的 44%，给第一个 token 21%
  第 15 层：门均值 0.12，小于 0.1 的 56%，给第一个 token 27%
  第 19 层：门均值 0.13，小于 0.1 的 50%，给第一个 token 5%
  第 23 层：门均值 0.19，小于 0.1 的 49%，给第一个 token 3%
「中国的首都是」的下一个词，排前三的是 哪个 18.6%、北京 18.6%、哪里 6.9%
  贡献最大的之一：第 23 层第 275 号神经元，激活 +5.86；它写回的方向最像这些词：杭州 重庆 广州 南京 Shanghai 成都 天津 武汉
  贡献最大的之一：第 18 层第 2065 号神经元，激活 +3.30；它写回的方向最像这些词：London paris Paris Londres Washington 伦敦 大城市 华盛顿
  只把第 23 层第 275 号神经元关掉：P(北京) = 1.7%，排前三的变成 哪个 29.5%、____ 9.3%、？ 6.5%
```

**门大多数时候关着。** 6 个全注意力层的门均值只有 0.07～0.19，33%～77% 的门值小于 0.1；后一半 query 分给第一个 token 的注意力是 2%～27%。同一句话在没有门的 Qwen3-0.6B 上，第 6 层以后是 40%～77%（[大模型原理手册的注意力一章](llm://transformer/attention/#真实模型里的注意力注意力汇聚)）。第 6 章那张"没有门 vs 有门"的图，用的就是这两组数。

**前馈网络像一本"读到什么就写回什么"的规则本。** 前馈网络中间有 3584 个神经元（第 4 章）。读到"中国的首都是"时，第 23 层第 275 号神经元的激活是 +5.86，它写回残差流的那一列，按输出层读出来最像"杭州 重庆 广州 南京 Shanghai 成都 天津 武汉"，是一个"中国城市"的方向；第 18 层第 2065 号推的是"London Paris Washington 伦敦"这些首都。只关掉第 23 层那一个神经元，"北京"的概率就从 18.6% 掉到 1.7%，排第一的变成"哪个"：模型把这句话当成了问句的开头。第 4 章那张"规则本"的图，用的就是这里的数。

这里的"贡献"只算了一条直通路：神经元的激活 × 它写回的那一列 × 输出层上"北京"的方向，忽略了后面各层对它的再加工，所以只用来挑候选；真正说明问题的是最后那个"关掉它"的实验。`gate_proj`、`up_proj` 的一行负责"认出现在是什么情况"，`down_proj` 的一列负责"这时往残差流里写什么"，所以常把前馈网络比作一张键值表：键在 gate、up 里，值在 down 的列里。

!!! interview "怎么讲清楚"
    讲整模型的结构，一句话：嵌入 → N 个（Pre-Norm + 混合器 + 残差，Pre-Norm + 前馈 + 残差）→ 归一化 → 输出层。再强调两点：残差流只加不覆盖；每层的混合器可以是不同的东西（Qwen3.5 是 3 层线性 + 1 层全注意力循环），前馈和归一化是一样的。能把"权重文件里的名字 → 模块树上的属性"这条线讲清楚，就说明真的拼过。

## 小结

- [x] 解码层 = 两个 Pre-Norm 子层 + 两条残差；整模型 = 嵌入 + 24 层 + 归一化 + 共用权重的输出层。
- [x] 全书唯一一次和 HF 对：整段前向差 1e-5 量级；用 `tokenizers` 直接读 tokenizer.json，生成的句子和第 0 步一字不差。
- [x] 用钩子抓真实输入往里看：门的均值只有 0.07～0.19；第 23 层一个"中国城市"神经元，关掉它 P(北京) 从 18.6% 掉到 1.7%。
- [x] 阶段一结束：`qwen/` 里 9 个文件，不再依赖 transformers。

## 思考题

**1. 整模型和 HF 的 logits 差 1e-5，如果某一章写错了（比如 RMSNorm 忘了零中心的 1 + w），大概会差多少？怎么定位是哪一层？**

??? success "参考答案"
    会差到 0.1 以上，下一个 token 的 top-3 也会变。定位办法是逐层对：两边各挂钩子抓每一层的输出，从第 0 层开始比，第一个差出 1e-3 以上的层就是出错的零件所在；再在那一层里按子层（归一化、混合器、前馈）细分。

**2. 为什么"直接贡献最大"的神经元不一定是对结果影响最大的？**

??? success "参考答案"
    直接贡献只看神经元写回残差流的那一列直接投到输出层上的分量，忽略了后面各层会读到这个方向、再加工它；也忽略了神经元之间的相互抵消。要判断因果，得做干预：把它关掉（或换成别的输入下的值）再看输出变了多少。

**这一章新建的文件**：`qwen/layer.py`、`qwen/model.py`。
