# 9. The decoder layer and the whole model

<p class="lead">All the parts exist. This chapter puts them together: a decoder layer = normalize → mixer (full or linear attention) → residual → normalize → feed-forward → residual; the whole model = embedding → 24 layers → normalize → output layer. From this chapter on, <strong>there is not a single line of Hugging Face code in the forward pass</strong>; even the tokenizer is replaced by the <code>tokenizers</code> library reading tokenizer.json directly. Finally it generates step 0's sentence, token for token the same.</p>

!!! abstract "After this chapter you can"
    1. Describe a decoder layer and the whole model: Pre-Norm, the mixer, the feed-forward network, two residual connections.
    2. Map every name in the weight files onto the modules you wrote, and match HF's logits to 1e-5 with the whole model.
    3. Look inside the whole model: how open the gates are on real input, and what one neuron of the feed-forward network stores.

!!! note "What this chapter runs"
    `ch09_model.py`, about 30 seconds (compare the whole model's logits with HF, then generate 40 tokens — there is no KV cache yet, so it is slow); `ch09_inside.py`, about 10 seconds (looking inside the whole model).

@@files@@

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

## The book's only comparison with HF, then drop HF {#全书唯一一次和-hf-对然后扔掉-hf}

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

hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 1. the book's only comparison with HF: the whole model's logits
tok = AutoTokenizer.from_pretrained(path)
for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 40)):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = hf(ids).logits
    print(f"{ids.shape[1]:3d} 个 token：logits 最大差 {diff(model(ids), ref):.1e}，下一个 token 的 top-3 一样：{model(ids)[0, -1].topk(3).indices.tolist() == ref[0, -1].topk(3).indices.tolist()}")
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
  6 个 token：logits 最大差 1.3e-05，下一个 token 的 top-3 一样：True
279 个 token：logits 最大差 2.6e-05，下一个 token 的 top-3 一样：True
提示词 17 个 token，和第 0 步一样
生成 38 个 token，18.7 秒（每步都重算整段，第 10 章解决）
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。
```

The eight chapters before this one never checked against a reference implementation; here everything is checked at once: the whole forward pass differs from Hugging Face's logits by about 1e-5 (from exactly two sources: the order of the attention matrix multiplications and the order of summation in the chunked linear attention), the top-3 next tokens are identical, and the generated sentence is word for word step 0's. That means every part before this was written correctly — had any chapter got it wrong, the difference here would be 0.1 or more.

## Looking inside: what the gates and the feed-forward network do {#拼好以后往里看门和前馈网络在做什么}

When chapter 6 covered the gate and chapter 4 the feed-forward network, one thing could not be seen: there was no whole model yet, so what went in was not a real intermediate result. Now it can be done: capture the input a layer really receives with a hook (`register_forward_pre_hook`), then redo that chapter's formula by hand.

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

text = "推理优化的核心是减少访存。大模型在解码阶段每生成一个词，都要读取全部的权重和缓存。"   # the same sentence as the attention chapter of the LLM handbook
ids = torch.tensor([tk.encode(text).ids])
T = ids.shape[1]
inputs = {}                                                          # 1. capture the real inputs of the 6 full-attention layers
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

E = model.embed_tokens.weight                                       # 2. what the feed-forward network stores: the output layer shares this weight with the embedding table
scale = 1 + model.norm.weight                                        # the per-dimension scale of the final zero-centred RMSNorm
ids = torch.tensor([tk.encode("中国的首都是").ids])
target = tk.encode("北京").ids[0]
acts = {}
for i, layer in enumerate(model.layers):                             # capture the activations of the 3584 neurons in the middle of each feed-forward network at the last position
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, args, i=i: acts.__setitem__(i, args[0][0, -1]))
with torch.no_grad():
    p = model(ids)[0, -1].softmax(-1)
    print("「中国的首都是」的下一个词，排前三的是", "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p.topk(3))))
    u = E[target] * scale                                            # the direction of "北京" (Beijing) in the output layer
    best = []
    for L, act in acts.items():                                      # each neuron's direct contribution = activation × the component, along "北京", of the column it writes back to the residual stream
        c = act * (model.layers[L].mlp.down_proj.weight.T @ u)
        j = c.argmax().item()
        best.append((c[j].item(), L, j, act[j].item()))
    best.sort(reverse=True)
    for c, L, j, act in best[:2]:
        col = model.layers[L].mlp.down_proj.weight[:, j] * scale     # the column this neuron writes back to the residual stream, read as words through the output layer
        words = list(dict.fromkeys(tk.decode([t]).strip() for t in (E @ col).topk(12).indices.tolist()))[:8]
        print(f"  贡献最大的之一：第 {L} 层第 {j} 号神经元，激活 {act:+.2f}；它写回的方向最像这些词：{' '.join(words)}")
    L, j = best[0][1], best[0][2]
    h = model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda m, args: (args[0].index_fill(-1, torch.tensor([j]), 0.0),))
    p2 = model(ids)[0, -1].softmax(-1)
    top = "、".join(f"{tk.decode([t])} {v:.1%}" for v, t in zip(*p2.topk(3)))
    print(f"  只把第 {L} 层第 {j} 号神经元关掉：P(北京) = {p2[target]:.1%}，排前三的变成 {top}")
    h.remove()
```

```text title="output"
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

**The gates are closed most of the time.** The gates of the 6 full-attention layers average only 0.07–0.19, and 33%–77% of the gate values are below 0.1; the second half of the queries give 2%–27% of their attention to the first token. On Qwen3-0.6B, which has no gate, the same sentence gives 40%–77% from layer 6 on ([the attention chapter of the LLM handbook](llm://transformer/attention/#真实模型里的注意力注意力汇聚)). Chapter 6's "without a gate vs with a gate" figure uses exactly these two sets of numbers.

**The feed-forward network is a rule book of "when you read this, write back that".** The feed-forward network has 3584 neurons in the middle (chapter 4). When reading "中国的首都是" ("The capital of China is"), neuron 275 of layer 23 has an activation of +5.86, and the column it writes back to the residual stream, read through the output layer, looks most like "杭州 重庆 广州 南京 Shanghai 成都 天津 武汉" (Hangzhou, Chongqing, Guangzhou, Nanjing, Shanghai, Chengdu, Tianjin, Wuhan): a "Chinese cities" direction; neuron 2065 of layer 18 pushes capitals such as "London Paris Washington 伦敦". Switching off just that one neuron of layer 23 drops the probability of "北京" (Beijing) from 18.6% to 1.7%, and "哪个" ("which") moves to the top: the model now reads the sentence as the start of a question. Chapter 4's "rule book" figure uses exactly these numbers.

The "contribution" here follows only one direct path: the neuron's activation × the column it writes back × the direction of "北京" in the output layer, ignoring how the later layers process it further, so it is only used to pick candidates; what really settles the question is the final "switch it off" experiment. A row of `gate_proj` and `up_proj` "recognizes the situation", and a column of `down_proj` decides "what to write into the residual stream then", which is why the feed-forward network is often likened to a key-value table: the keys live in gate and up, the values in the columns of down.

!!! interview "How to explain it"
    The structure of the whole model in one sentence: embedding → N × (Pre-Norm + mixer + residual, Pre-Norm + feed-forward + residual) → normalize → output layer. Then stress two points: the residual stream only adds, never overwrites; the mixer of each layer can be a different thing (Qwen3.5 cycles 3 linear + 1 full attention), while the feed-forward and the normalizations are the same. Being able to trace "names in the weight file → attributes on the module tree" shows you have really assembled one.

## Summary {#小结}

- [x] A decoder layer = two Pre-Norm sublayers + two residuals; the whole model = embedding + 24 layers + normalization + a tied output layer.
- [x] The book's only comparison with HF: the whole forward pass differs by about 1e-5; tokenizer.json is read with `tokenizers` directly, and the generated sentence is word for word step 0's.
- [x] Looking inside with hooks on real inputs: the gates average only 0.07–0.19; a "Chinese cities" neuron in layer 23, and switching it off drops P(北京) from 18.6% to 1.7%.
- [x] Stage 1 is complete: 9 files in `qwen/`, no dependency on transformers any more.

## Questions {#思考题}

**1. The whole model's logits differ from HF's by 1e-5. If some chapter were wrong (say RMSNorm forgot the 1 + w of zero centring), roughly how large would the difference be? How would you find the layer?**

??? success "Answer"
    It would exceed 0.1, and the top-3 of the next token would change too. To locate it, compare layer by layer: capture every layer's output on both sides with hooks and compare from layer 0; the first layer that differs by more than 1e-3 holds the broken part, which you can then narrow down by sublayer (normalization, mixer, feed-forward).

**2. Why is the neuron with the "largest direct contribution" not necessarily the one with the largest effect on the result?**

??? success "Answer"
    The direct contribution only measures how much of the column the neuron writes back projects straight onto the output layer; it ignores that later layers read this direction and process it further, and that neurons can cancel each other. Judging cause and effect takes an intervention: switch it off (or replace it with its value under another input) and see how much the output changes.

**Files created in this chapter**: `qwen/layer.py`, `qwen/model.py`.

