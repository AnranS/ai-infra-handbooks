# 9. The decoder layer and the whole model

<p class="lead">All the parts exist. This chapter puts them together: a decoder layer = normalize → mixer (full or linear attention) → residual → normalize → feed-forward → residual; the whole model = embedding → 24 layers → normalize → output layer. From this chapter on, <strong>there is not a single line of Hugging Face code in the forward pass</strong>; even the tokenizer is replaced by the <code>tokenizers</code> library reading tokenizer.json directly. Finally it generates step 0's sentence, token for token the same.</p>

!!! note "What this chapter runs"
    `ch09_model.py`, about 30 seconds (compare the whole model's logits with HF, then generate 40 tokens — there is no KV cache yet, so it is slow).

## The decoder layer {#解码层}

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
        self.kind = kind                                                   # "full_attention" or "linear_attention"
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
        x = x + h                                                          # residual: the mixer only *adds* to the residual stream
        return x + self.mlp(self.post_attention_layernorm(x))
```

The residual stream is the thread for understanding a Transformer: `x` leaves the embedding, passes through 24 layers, and each layer only **adds** two things to it (the mixer's output, the feed-forward's output), never overwriting it. So the scale of `x` keeps growing — which is why every sublayer has an RMSNorm in front of it (Pre-Norm): the sublayer sees a normalized copy, while the residual stream itself stays clear all the way.

There are two kinds of mixer, chosen by `layer_types`: the attribute is called `linear_attn` in `linear_attention` layers and `self_attn` in `full_attention` layers — deliberately the same names as in the weight file, so that `load_state_dict` lines up directly.

## The whole model {#整模型}

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
        self.lm_head = LMHead(self.embed_tokens.weight)                       # shares weights with the embedding

    @classmethod
    def from_pretrained(cls, path=None, dtype=torch.float32):
        path = path or find_model()
        model = cls(Config.load(path))
        missing, unexpected = model.load_state_dict(load_weights(path, dtype=dtype), strict=False, assign=True)   # assign=True: take over the tensors instead of copying
        assert not unexpected and missing == ["lm_head.weight"], (missing, unexpected)   # no lm_head in the file: it is the embedding
        model.lm_head.weight = model.embed_tokens.weight                                  # after the take-over the embedding table is a new object; point the output layer at it again
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

The `assert` in `from_pretrained`: the weight file has no `lm_head.weight` (it shares the embedding table), hence `strict=False`, but nothing else may be missing or extra — insurance against "one name misspelled and a module silently using random initialization".

## Compare with HF, then drop HF {#和-hf-对然后扔掉-hf}

```python title="ch09_model.py"
"""第 9 章：整模型和 HF 对 logits；只用 tokenizers 库和自己的模型生成第 0 步那句话"""
import os
import time

import torch
from tokenizers import Tokenizer

from qwen.harness import load_hf, logits, max_diff
from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
print(f"自己的模型：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M 参数，{len(model.layers)} 层")

hf, tok = load_hf(path)                                             # 1. the last comparison with HF
for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 40)):
    ids = tok(text, return_tensors="pt")["input_ids"]
    print(f"{ids.shape[1]:3d} 个 token：logits 最大差 {max_diff(model(ids), logits(hf, ids)):.1e}")
del hf

tk = Tokenizer.from_file(str(path / "tokenizer.json"))              # 2. no more transformers: read tokenizer.json with the tokenizers library
prompt = "<|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # chapter 12 writes this template by hand
ids = torch.tensor([tk.encode(prompt).ids])
print(f"提示词 {ids.shape[1]} 个 token，和第 0 步一样")
STOP = {tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>")}

t0 = time.time()                                                    # 3. the loop of step 0, with our own model
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

```text title="output"
自己的模型：752.4M 参数，24 层
  6 个 token：logits 最大差 1.3e-05
279 个 token：logits 最大差 2.6e-05
提示词 17 个 token，和第 0 步一样
生成 38 个 token，19.1 秒（每步都重算整段，第 10 章解决）
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。
```

Word for word the same as step 0. The error of the whole forward pass is in the 1e-5 range, from exactly two sources: the order of the attention matrix multiplications and the order of summation in the chunked linear attention.

!!! interview "How to explain it"
    The structure of the whole model in one sentence: embedding → N × (Pre-Norm + mixer + residual, Pre-Norm + feed-forward + residual) → normalize → output layer. Then stress two points: the residual stream only adds, never overwrites; the mixer of each layer can be a different thing (Qwen3.5 cycles 3 linear + 1 full attention), while the feed-forward and the normalizations are the same. Being able to trace "names in the weight file → attributes on the module tree" shows you have really assembled one.

## Summary {#小结}

- [x] A decoder layer = two Pre-Norm sublayers + two residuals; the whole model = embedding + 24 layers + normalization + a tied output layer.
- [x] The whole forward pass differs from HF by at most about 1e-5; tokenizer.json is read with `tokenizers` directly, and the generated sentence is word for word step 0's.
- [x] Stage 1 is complete: 9 files in `qwen/`, no dependency on transformers any more.

**Files created in this chapter**: `qwen/layer.py`, `qwen/model.py`.
