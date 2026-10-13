# 1. Weights and configuration

<p class="lead">The first step of writing a model from scratch is not writing an operator; it is reading the <strong>configuration</strong> and the <strong>weights</strong>: the few dozen hyperparameters in config.json decide the shape of every tensor, and the 320 tensors in the safetensors file map by name onto the module tree. This chapter writes a single file, <code>weights.py</code>, and then works out the parameter budget module by module — every later chapter writes one module, and first has to know what that module's weights are called, how big they are and what they look like.</p>

!!! note "What this chapter runs"
    `ch01_weights.py`, about 10 seconds (read the weights, work out the parameter budget by module). From this chapter on, every script imports from the `qwen/` package and sits next to it: `python ch01_weights.py`.

@@files@@

## config.json: where the shapes come from {#configjson形状从哪来}

Qwen3.5 is a multimodal model, so the top level of config.json has three parts: `architectures`, `vision_config` and `text_config`; everything about the language model is under `text_config`. We pick the fields we use into a dataclass:

```python title="qwen/weights.py"
"""读配置和权重：config.json 里的超参数，safetensors 里的 488 个张量。

权重文件里有三部分：model.language_model.*（语言模型，这本书要手写的）、model.visual.*（视觉编码器，不碰）
和 mtp.*（多 token 预测的草稿头，第 14 章用）。这里把前缀去掉，剩下的名字和模块树一一对应。
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


def find_model(name="Qwen3.5-0.8B"):
    """从当前目录往上找 models/<name>；也可以用环境变量 QWEN_MODEL 直接指定"""
    if os.environ.get("QWEN_MODEL"):
        return Path(os.environ["QWEN_MODEL"])
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "models" / name).is_dir():
            return d / "models" / name
    raise FileNotFoundError(f"没找到 models/{name}，先下载：hf download Qwen/{name} --local-dir models/{name}")


@dataclass
class Config:
    hidden_size: int                 # 1024: the width of the residual stream
    num_hidden_layers: int           # 24
    layer_types: list                # whether each layer is linear_attention or full_attention
    num_attention_heads: int         # full attention: 8 query heads
    num_key_value_heads: int         # 2 key/value groups (GQA)
    head_dim: int                    # 256
    intermediate_size: int           # 3584: the FFN's intermediate width
    vocab_size: int                  # 248320
    rms_norm_eps: float
    rope_theta: float                # 1e7
    partial_rotary_factor: float     # 0.25: only the first quarter of each head is rotated
    mrope_section: list              # [11, 11, 10]: how many frequencies each of the three position axes gets
    linear_num_key_heads: int        # linear attention: 16 key heads
    linear_num_value_heads: int      # 16 value heads
    linear_key_head_dim: int         # 128
    linear_value_head_dim: int       # 128
    linear_conv_kernel_dim: int      # 4: the short convolution's window
    tie_word_embeddings: bool        # the output layer shares weights with the embedding
    eos_token_id: int

    @classmethod
    def load(cls, path):
        c = json.loads((Path(path) / "config.json").read_text())["text_config"]   # a multimodal model: the language model's config lives under text_config
        rope = c["rope_parameters"]
        fields = {k: c[k] for k in cls.__dataclass_fields__ if k in c}
        return cls(**fields, rope_theta=rope["rope_theta"], partial_rotary_factor=rope["partial_rotary_factor"],
                   mrope_section=rope["mrope_section"])


def load_weights(path, part="model.language_model.", dtype=torch.float32):
    """读 safetensors，返回 {去掉前缀的名字: 张量}。part 可以是 "model.language_model."、"mtp." 或 "model.visual."。"""
    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.startswith(part):
                    out[k[len(part):]] = st.get_tensor(k).to(dtype)   # the file is bf16; read it in and convert to fp32 (the reason is below)
    return out


def sub(weights, prefix):
    """取一个子模块的权重：sub(W, "layers.0.mlp.") -> {"gate_proj.weight": ..., "up_proj.weight": ..., ...}"""
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

def load(module, weights, prefix):
    """把一个子模块的权重灌进去：load(MLP(cfg), W, "layers.0.mlp.")。我们的属性名和文件里的一致，所以 state_dict 直接对上。"""
    module.load_state_dict(sub(weights, prefix), assign=True)   # assign=True: take over these tensors directly instead of copying them
    return module
```

Two details:

- **safetensors is not pickle.** The file starts with a JSON header (the name, dtype, shape and byte offset of every tensor) followed by raw data, so `safe_open` can read one tensor, or even one slice, which is why step 0's `map.py` could list every shape without loading any data.
- **The file is bf16; we convert to fp32.** Not because fp32 is "more correct", but because chapter 9 checks the logits against the reference implementation (HF) with both sides in fp32 — only then can the error be pushed down to the 1e-5 range where a bug is obvious at a glance. Real deployments on a GPU use bf16 (chapter 15).

## The weight names are the module tree {#权重名就是模块树}

The names of the 320 tensors follow a pattern: `layers.0.mlp.gate_proj.weight` reads as "layer 0 → feed-forward → gate projection → weight". Everything before the last dot is the module tree, with the same module names as the Hugging Face implementation (`model.layers[0].mlp`). When the later chapters write their own modules, the attributes get the same names (`gate_proj`, `self_attn`, `linear_attn`, …), so `load_state_dict` pours the weights in by name, with no mapping table to write.

![Figure: the module tree: the prefix of a weight name is the module name, and each module has its chapter](assets/figures/skeleton.svg){.aig-svg}

The eight chapters of stage 1 never check against a reference implementation: each chapter finishes one part and runs a small experiment to see what it does — who is close to whom in embedding space, where normalization puts the scale, what the attention matrix looks like, why the memory table does not grow with length. Chapter 9 assembles the parts into the whole model and, for the only time in the book, checks the logits against HF (differences around 1e-5); if they match, every part before it was written correctly.

## Run it: the parameter budget and what the weights look like {#跑一遍参数账和权重的样子}

Run it and look at four things: the numbers in the configuration, the parameter budget by module, the tensors inside one layer, and what the values of a few weights look like.

```python title="ch01_weights.py"
"""第 1 章：读配置和权重，按模块算一笔参数账，看看权重长什么样"""
import os
import re
from collections import OrderedDict

import torch

from qwen.weights import Config, find_model, load_weights, sub

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
print(f"{cfg.num_hidden_layers} 层，hidden {cfg.hidden_size}，词表 {cfg.vocab_size}，RoPE theta {cfg.rope_theta:.0e}，只旋转 {cfg.partial_rotary_factor:.0%} 的维度")
print("层的排法：", "".join("A" if t == "full_attention" else "L" for t in cfg.layer_types))

W = load_weights(path)
total = sum(v.numel() for v in W.values())
print(f"语言模型 {len(W)} 个张量，{total / 1e6:.1f}M 参数；文件里是 bf16（{total * 2 / 2**30:.2f} GB），读进内存转成 fp32（{total * 4 / 2**30:.2f} GB）")
print("lm_head.weight 在文件里吗：", "lm_head.weight" in W, "；tie_word_embeddings =", cfg.tie_word_embeddings, "（输出层就是嵌入表）")

groups = OrderedDict()                                           # 1. the parameter budget by module: layer numbers in the names become N
for k, v in W.items():
    groups.setdefault(re.sub(r"\.\d+\.", ".N.", k).rsplit(".", 1)[0], []).append(v.numel())
print("模块                             张量数  参数量   占比")
for name, sizes in groups.items():
    print(f"  {name:<30} ×{len(sizes):<3} {sum(sizes) / 1e6:7.1f}M {sum(sizes) / total:5.1%}")

print("第 0 层（线性注意力）：", sorted(sub(W, "layers.0.")))                            # 2. what a layer contains: the prefix of a name is the module
print("第 3 层（全注意力）：  ", sorted(sub(W, "layers.3.")))

for name in ("embed_tokens.weight", "layers.0.mlp.gate_proj.weight", "layers.3.self_attn.q_proj.weight", "norm.weight", "layers.0.linear_attn.A_log"):
    v = W[name]                                                   # 3. what the values look like: the matrices are tiny, the normalization weights are not
    print(f"  {name:<36} {str(tuple(v.shape)):<16} 均值 {v.mean():+.4f}  标准差 {v.std():.4f}  范围 [{v.min():+.2f}, {v.max():+.2f}]")
```

```text title="输出"
24 层，hidden 1024，词表 248320，RoPE theta 1e+07，只旋转 25% 的维度
层的排法： LLLALLLALLLALLLALLLALLLA
语言模型 320 个张量，752.4M 参数；文件里是 bf16（1.40 GB），读进内存转成 fp32（2.80 GB）
lm_head.weight 在文件里吗： False ；tie_word_embeddings = True （输出层就是嵌入表）
模块                             张量数  参数量   占比
  embed_tokens                   ×1     254.3M 33.8%
  layers.N.input_layernorm       ×24      0.0M  0.0%
  layers.N.linear_attn           ×36      0.0M  0.0%
  layers.N.linear_attn.conv1d    ×18      0.4M  0.1%
  layers.N.linear_attn.in_proj_a ×18      0.3M  0.0%
  layers.N.linear_attn.in_proj_b ×18      0.3M  0.0%
  layers.N.linear_attn.in_proj_qkv ×18    113.2M 15.1%
  layers.N.linear_attn.in_proj_z ×18     37.7M  5.0%
  layers.N.linear_attn.norm      ×18      0.0M  0.0%
  layers.N.linear_attn.out_proj  ×18     37.7M  5.0%
  layers.N.mlp.down_proj         ×24     88.1M 11.7%
  layers.N.mlp.gate_proj         ×24     88.1M 11.7%
  layers.N.mlp.up_proj           ×24     88.1M 11.7%
  layers.N.post_attention_layernorm ×24      0.0M  0.0%
  layers.N.self_attn.k_norm      ×6       0.0M  0.0%
  layers.N.self_attn.k_proj      ×6       3.1M  0.4%
  layers.N.self_attn.o_proj      ×6      12.6M  1.7%
  layers.N.self_attn.q_norm      ×6       0.0M  0.0%
  layers.N.self_attn.q_proj      ×6      25.2M  3.3%
  layers.N.self_attn.v_proj      ×6       3.1M  0.4%
  norm                           ×1       0.0M  0.0%
第 0 层（线性注意力）： ['input_layernorm.weight', 'linear_attn.A_log', 'linear_attn.conv1d.weight', 'linear_attn.dt_bias', 'linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight', 'linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight', 'linear_attn.norm.weight', 'linear_attn.out_proj.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight']
第 3 层（全注意力）：   ['input_layernorm.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight', 'self_attn.k_norm.weight', 'self_attn.k_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.q_proj.weight', 'self_attn.v_proj.weight']
  embed_tokens.weight                  (248320, 1024)   均值 +0.0002  标准差 0.0197  范围 [-0.19, +0.23]
  layers.0.mlp.gate_proj.weight        (3584, 1024)     均值 +0.0000  标准差 0.0121  范围 [-0.13, +0.18]
  layers.3.self_attn.q_proj.weight     (4096, 1024)     均值 -0.0002  标准差 0.0198  范围 [-0.24, +0.25]
  norm.weight                          (1024,)          均值 +3.3092  标准差 0.6692  范围 [-0.78, +5.31]
  layers.0.linear_attn.A_log           (16,)            均值 -1.9795  标准差 1.4952  范围 [-6.59, +0.26]
```

A few numbers worth remembering: the vocabulary is 34%; linear attention's `in_proj_qkv` (1024 → 6144) alone is 15%, far more than full attention's `q_proj` — because there are 18 such layers; the three feed-forward matrices are 11.7% each. The standard deviation of every matrix is 0.01–0.02 (the scale of the initialization), while the final `norm.weight` has a mean of 3.3 — it is not a small zero-mean number, and chapter 3 explains why it is stored as $w-1$ and why it scales things up. `A_log` is 16 numbers (one per head); chapter 7 explains that it is linear attention's "forgetting speed".

!!! interview "How to explain it"
    To explain "what is in a model file", the answer is a table: name → shape → parameter count. Being able to split the 0.8B's 752M parameters into "vocabulary 254M + 18 linear-attention layers of 21.5M each + 6 full-attention layers of 18.4M each", and to name the operator each block belongs to, shows you have actually read the weights rather than only called an API. Add a sentence on why safetensors replaced pickle: it has an index, it can be read partially without copying, and it cannot execute arbitrary code.

## Summary {#小结}

- [x] `Config.load` takes only the fields of `text_config` we use; `load_weights` takes one of the three parts (language model / vision / MTP) by prefix and strips the prefix.
- [x] The budget of the 752M parameters: vocabulary 34%, 18 linear-attention layers 25%, 24 feed-forward networks 35%, 6 full-attention layers 6%; `lm_head` is not in the file, it is shared with the embedding.
- [x] The prefix of a weight name is the module name; our modules use the same names, so the weights are poured in by name.

**File created in this chapter**: `qwen/weights.py`.
