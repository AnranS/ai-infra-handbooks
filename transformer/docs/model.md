# （九）解码层与整模型

<p class="lead">零件全有了。这一章把它们拼起来：一个解码层 = 归一化 → 混合器（全注意力或线性注意力）→ 残差 → 归一化 → 前馈 → 残差；整模型 = 嵌入 → 24 层 → 归一化 → 输出层。从这一章起，<strong>前向里没有一行 Hugging Face 的代码</strong>，连 tokenizer 都换成 <code>tokenizers</code> 库直接读 tokenizer.json。最后用它生成第 0 步那句话，逐 token 一致。</p>

!!! note "这一章跑什么"
    `ch09_model.py`，约 30 秒（整模型和 HF 对 logits，再生成 40 个 token——还没有 KV cache，所以慢）。

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

!!! interview "怎么讲清楚"
    讲整模型的结构，一句话：嵌入 → N 个（Pre-Norm + 混合器 + 残差，Pre-Norm + 前馈 + 残差）→ 归一化 → 输出层。再强调两点：残差流只加不覆盖；每层的混合器可以是不同的东西（Qwen3.5 是 3 层线性 + 1 层全注意力循环），前馈和归一化是一样的。能把"权重文件里的名字 → 模块树上的属性"这条线讲清楚，就说明真的拼过。

## 小结

- [x] 解码层 = 两个 Pre-Norm 子层 + 两条残差；整模型 = 嵌入 + 24 层 + 归一化 + 共用权重的输出层。
- [x] 全书唯一一次和 HF 对：整段前向差 1e-5 量级；用 `tokenizers` 直接读 tokenizer.json，生成的句子和第 0 步一字不差。
- [x] 阶段一结束：`qwen/` 里 9 个文件，不再依赖 transformers。

**这一章新建的文件**：`qwen/layer.py`、`qwen/model.py`。
