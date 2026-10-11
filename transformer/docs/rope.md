# （五）RoPE：旋转位置编码

<p class="lead">注意力本身不知道 token 的先后——把一句话打乱，每对 (query, key) 的点积不变。RoPE 把"第几个位置"编成一个角度，<strong>把 query 和 key 转过这个角</strong>，于是两者的点积只和它们的距离有关。Qwen3.5 的版本有三个不常见的细节：只旋转每个头前 1/4 的维度、基频 10,000,000、以及为图像和视频准备的三轴位置（M-RoPE）。这一章把它写出来，换掉 HF 的 <code>rotary_emb</code>。</p>

!!! note "这一章跑什么"
    `ch05_rope.py`，约 10 秒。

@@files@@

## 转一个角

```python title="qwen/rope.py"
"""旋转位置编码（RoPE）：把"第几个位置"转进 query 和 key 里，让注意力分数只跟相对距离有关。

Qwen3.5 的三个细节：只旋转每个头前 1/4 的维度（256 维里的 64 维，partial_rotary_factor=0.25）；
基频 theta = 10,000,000；为了图像和视频，位置有三个轴（时间、高、宽，M-RoPE），32 个频率按 11/11/10 交错分给三个轴。
纯文本时三个轴的位置相同，M-RoPE 就退化成普通的 RoPE——这个文件写的就是这个退化版。
"""
import torch


def rope_tables(cfg, positions):
    """positions: [T]（或每行不同的 [B, T]）的整数位置 -> cos, sin 各 [T, rot_dim]（rot_dim = head_dim * partial_rotary_factor = 64）"""
    rot = int(cfg.head_dim * cfg.partial_rotary_factor)
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rot, 2).float() / rot))   # 32 个频率：从 1 一路降到 1/theta
    angles = positions.float()[..., None] * inv_freq                                  # [T, 32]：位置 × 频率 = 转角
    angles = torch.cat([angles, angles], dim=-1)                                      # [T, 64]：前后两半配成一对，用同一个角
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, T, 头数, head_dim]。只转前 rot_dim 维，剩下的维度原样拼回去。"""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    cos, sin = cos[..., None, :].to(x.dtype), sin[..., None, :].to(x.dtype)         # 对批和头广播；表是 fp32 算的，用的时候转成 x 的精度
    x_rot = x_rot * cos + rotate_half(x_rot) * sin                                    # 二维旋转：(a, b) -> (a cos - b sin, b cos + a sin)
    return torch.cat([x_rot, x_pass], dim=-1)
```

把每个头的 256 维看成 128 对 $(a, b)$，对第 $i$ 对，位置 $p$ 对应的转角是 $p \cdot \theta_i$，其中 $\theta_i = \text{base}^{-2i/d}$ 从 1 一路降到 $1/\text{base}$——前几对转得快（分辨相邻位置），后几对转得极慢（分辨几千几万个位置外的远近）。两个被转过的向量的点积：

$$
\langle R_{p}\,q,\; R_{p'}\,k \rangle = \langle q,\; R_{p'-p}\,k \rangle
$$

只依赖 $p' - p$。这就是 RoPE 的全部：绝对位置加在向量上，相对位置出现在点积里。

**只旋转 1/4 维。** `partial_rotary_factor = 0.25`：256 维里只有前 64 维（32 对）被旋转，其余 192 维不带位置信息。这是 Qwen3.5 从 Qwen3-Next 继承的做法：留一部分维度专门做"和位置无关的匹配"，长上下文下更稳。

**基频 10,000,000。** 经典值是 10,000；基频越大，最慢的那几对转得越慢，262K 的上下文里才不会"转过一圈"把远近混淆。

**M-RoPE。** 多模态时位置有三个轴：时间（文本的先后、视频的帧）、图像的高、图像的宽。32 个频率按 `mrope_section = [11, 11, 10]` 交错分给三个轴（第 0、3、6 … 个频率归时间轴，第 1、4、7 … 归高，第 2、5 … 归宽）。纯文本时三个轴的位置相同，三张表完全一样，所以上面的代码直接按一个轴写——第 1 章 `RotaryAdapter` 从 HF 的 `[3, B, T]` 位置里取第 0 轴，就是这个道理。

## 换上去，对答案

```python title="ch05_rope.py" ci="loose"
"""第 5 章：自己算 cos/sin 表，和 HF 的逐位对；换掉 rotary_emb；看一眼"相对位置"是怎么出现在点积里的"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import RotaryAdapter, load_hf, logits, max_diff, status
from qwen.mlp import MLP
from qwen.norm import RMSNorm
from qwen.rope import apply_rope, rope_tables
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

pos = torch.arange(ids.shape[1])
cos, sin = rope_tables(cfg, pos)
hf_cos, hf_sin = hf.model.rotary_emb(torch.zeros(1), pos.view(1, 1, -1).expand(3, 1, -1))   # HF 要三轴位置 [3, B, T]
print(f"cos 表 {tuple(cos.shape)}（HF 的是 {tuple(hf_cos.shape)}），和 HF 最大差 {max(max_diff(cos, hf_cos[0]), max_diff(sin, hf_sin[0])):.1e}")
rot = int(cfg.head_dim * cfg.partial_rotary_factor)
print(f"每个头 {cfg.head_dim} 维，只旋转前 {rot} 维（{rot // 2} 对）；最快的一对每个位置转 {1:.2f} 弧度，最慢的一对转 {1 / cfg.rope_theta ** ((rot - 2) / rot):.1e} 弧度")

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 第 2–4 章换过的
hf.lm_head = LMHead(hf.model.embed_tokens.weight)
hf.model.norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
for i, layer in enumerate(hf.model.layers):
    layer.input_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.input_layernorm.")
    layer.post_attention_layernorm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, f"layers.{i}.post_attention_layernorm.")
    layer.mlp = load(MLP(cfg), W, f"layers.{i}.mlp.")
hf.model.rotary_emb = RotaryAdapter(cfg, rope_tables)                                          # 这一章
print(f"换掉 rotary_emb 后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

torch.manual_seed(0)                                              # 相对位置：同一对 q、k，放在不同位置但距离相同，点积相同
q, k = torch.randn(1, 1, 1, cfg.head_dim), torch.randn(1, 1, 1, cfg.head_dim)
def dot(pq, pk):
    cq, sq = rope_tables(cfg, torch.tensor([pq]))
    ck, sk = rope_tables(cfg, torch.tensor([pk]))
    return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()
print(f"q 在 3、k 在 1：{dot(3, 1):.4f}；q 在 103、k 在 101：{dot(103, 101):.4f}；q 在 3、k 在 2：{dot(3, 2):.4f}")
```

```text title="输出"
cos 表 (6, 64)（HF 的是 (1, 6, 64)），和 HF 最大差 0.0e+00
每个头 256 维，只旋转前 64 维（32 对）；最快的一对每个位置转 1.00 弧度，最慢的一对转 1.7e-07 弧度
换掉 rotary_emb 后，logits 最大差 0.0e+00
已替换 7/9：embed_tokens✓  input_layernorm✓  post_attention_layernorm✓  mlp✓  rotary_emb✓  self_attn✗  linear_attn✗  norm✓  lm_head✓
q 在 3、k 在 1：-20.3131；q 在 103、k 在 101：-20.3131；q 在 3、k 在 2：-19.6979
```

最后一行是 RoPE 的"相对性"：(3, 1) 和 (103, 101) 的点积一样，因为距离都是 2；(3, 2) 不一样，因为距离变了。

!!! interview "怎么讲清楚"
    讲 RoPE 先说它解决什么：注意力对顺序不敏感，需要位置信息，而加法式的位置编码让点积里混进"绝对位置"项。再说做法：按位置转角，点积只剩相对位置。最后说两个数字的含义：基频决定能分辨多长的距离（Qwen3.5 用 1e7 配 262K 上下文），部分旋转留一些维度不带位置。能顺手提一句 M-RoPE——三轴位置、纯文本退化为一轴——说明你看过这一代多模态模型的实现。

## 小结

- [x] RoPE 把位置 $p$ 编成转角 $p\theta_i$，旋转 query 和 key；点积只依赖相对距离。
- [x] Qwen3.5：只转每头前 64 维、基频 1e7、M-RoPE 三轴（纯文本退化成一轴）；我们的 cos/sin 表和 HF 逐位相等。
- [x] 骨架 9 个位置换了 7 个。

**这一章新建的文件**：`qwen/rope.py`。
