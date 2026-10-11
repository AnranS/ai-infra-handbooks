# 1. Weights and configuration

<p class="lead">The first step of taking the black box apart is not writing an operator; it is reading the <strong>configuration</strong> and the <strong>weights</strong>: the few dozen hyperparameters in config.json decide the shape of every tensor, and the 320 tensors in the safetensors file map by name onto the module tree. This chapter writes two files: <code>weights.py</code> reads, and <code>harness.py</code> "plugs our own modules into the HF model and checks the answer" — every one of the next eight chapters uses it.</p>

!!! note "What this chapter runs"
    `ch01_weights.py`, about 10 seconds (read the weights, load the HF model, one forward pass). From this chapter on, every script imports from the `qwen/` package and sits next to it: `python ch01_weights.py`.

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
                    out[k[len(part):]] = st.get_tensor(k).to(dtype)   # the file is bf16; converting to fp32 lets us match the fp32 reference element by element
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
- **The file is bf16; we convert to fp32.** Not because fp32 is "more correct", but because the reference implementation (HF) also runs in fp32, and only with both sides in fp32 can the error be pushed down to the 1e-5 range where a bug is obvious at a glance. Real deployments on a GPU use bf16 (chapter 15).

## harness.py: the replacement skeleton {#harnesspy替换骨架}

Taking the black box apart works like this: the HF model is a module tree (`model.embed_tokens`, `model.layers[i].mlp`, `model.norm`, `lm_head`, …); once a chapter has written a module, we replace the corresponding node of the tree with ours and compute the logits again to compare with the original. Two things get in the way:

1. HF calls attention with its own set of arguments (`position_embeddings`, `attention_mask`, `past_key_values`, …), which our modules do not want to know about;
2. HF's rotary embedding returns cos/sin with a batch dimension, in the multimodal three-axis form.

So harness.py holds three **adapters** that translate HF's calls into our signatures. They are only used in chapters 2–8; once chapter 9 has assembled the whole model, they are no longer needed.

```python title="qwen/harness.py"
"""替换骨架：把 Hugging Face 的实现当黑盒，一块一块换成自己写的，每换一块就和原来的 logits 对一次。

HF 调用注意力和线性注意力时会传一堆它自己的参数（mask、cache、position_embeddings），
这里用几个适配器把它们翻译成我们的签名。换掉的模块越多，骨架里剩下的 HF 代码就越少，第 9 章全部换完。
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen.weights import find_model

SLOTS = ["embed_tokens", "input_layernorm", "post_attention_layernorm", "mlp", "rotary_emb", "self_attn", "linear_attn", "norm", "lm_head"]


def load_hf(path=None):
    """文本部分 Qwen3_5ForCausalLM，fp32；视觉编码器和 MTP 头在加载时被丢掉"""
    path = path or find_model()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    return model, AutoTokenizer.from_pretrained(path)


@torch.no_grad()
def logits(model, ids):
    out = model(ids)
    return out.logits if hasattr(out, "logits") else out


def max_diff(a, b):
    return (a - b).abs().max().item()


def status(hf):
    """骨架里哪些位置已经换成了自己写的：类定义在 qwen 包里的就算"""
    mine = lambda m: type(m).__module__.startswith("qwen")
    done = {"embed_tokens": mine(hf.model.embed_tokens), "norm": mine(hf.model.norm), "lm_head": mine(hf.lm_head),
            "rotary_emb": mine(hf.model.rotary_emb)}
    for layer in hf.model.layers:
        for name in ("input_layernorm", "post_attention_layernorm", "mlp", "self_attn", "linear_attn"):
            if hasattr(layer, name):
                m = getattr(layer, name)
                done[name] = done.get(name, True) and mine(m.mine if hasattr(m, "mine") else m)
    n = sum(done.values())
    return f"已替换 {n}/{len(SLOTS)}：" + "  ".join(f"{s}{'✓' if done[s] else '✗'}" for s in SLOTS)


class AttentionAdapter(nn.Module):
    """HF 这样调注意力：self_attn(hidden_states=..., position_embeddings=(cos, sin), attention_mask=..., ...)，要返回 (输出, 注意力权重)"""

    def __init__(self, mine):
        super().__init__()
        self.mine = mine

    def forward(self, hidden_states, position_embeddings, **kwargs):
        cos, sin = position_embeddings            # HF's cos/sin carry a batch dim [B, T, 64]; for pure text every sample has the same positions, so take the first
        return self.mine(hidden_states, cos[0], sin[0]), None


class LinearAttentionAdapter(nn.Module):
    """线性注意力的调用是 linear_attn(hidden_states=..., cache_params=..., attention_mask=...)"""

    def __init__(self, mine):
        super().__init__()
        self.mine = mine

    def forward(self, hidden_states, **kwargs):
        return self.mine(hidden_states)


class RotaryAdapter(nn.Module):
    """HF 的 rotary_emb(x, position_ids) 返回 (cos, sin)；position_ids 是多模态的三轴位置 [3, B, T]，纯文本三轴相同"""

    def __init__(self, cfg, tables):
        super().__init__()
        self.cfg, self.tables = cfg, tables

    def forward(self, x, position_ids):
        cos, sin = self.tables(self.cfg, position_ids[0, 0])
        B = position_ids.shape[1]
        return cos.expand(B, -1, -1).to(x.dtype), sin.expand(B, -1, -1).to(x.dtype)
```

## Run it: do we read what HF reads? {#跑一遍我们读到的和-hf-的一样吗}

This chapter replaces nothing yet; first confirm two things: the 320 tensors we read by name are equal, one by one, to what the HF model's `state_dict()` holds after loading, and record the logits before any replacement — every later chapter compares against them.

```python title="ch01_weights.py"
"""第 1 章：读配置和权重，和 HF 加载的逐个对；记下替换前的 logits"""
import os

import torch

from qwen.harness import load_hf, logits, max_diff, status
from qwen.weights import Config, find_model, load_weights, sub

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
print(f"{cfg.num_hidden_layers} 层，hidden {cfg.hidden_size}，词表 {cfg.vocab_size}，RoPE theta {cfg.rope_theta:.0e}，只旋转 {cfg.partial_rotary_factor:.0%} 的维度")

W = load_weights(path)
print(f"语言模型 {len(W)} 个张量，{sum(v.numel() for v in W.values()) / 1e6:.1f}M 参数（文件 bf16 -> 内存 fp32）")
print("lm_head.weight 在文件里吗：", "lm_head.weight" in W, "；tie_word_embeddings =", cfg.tie_word_embeddings)
print("第 0 层（线性注意力）的张量：", sorted(sub(W, "layers.0.").keys()))
print("第 3 层（全注意力）的张量：  ", sorted(sub(W, "layers.3.").keys()))

hf, tok = load_hf(path)                                                   # compare one by one with what HF loaded
sd = hf.state_dict()
mismatch = [k for k, v in W.items() if not torch.equal(v, sd["model." + k])]
print(f"和 HF 的 state_dict 对比：{len(W) - len(mismatch)} 个相同，{len(mismatch)} 个不同；HF 多出来的键：",
      sorted(set(k[6:] for k in sd if k.startswith("model.")) - set(W)) + [k for k in sd if not k.startswith("model.")])

ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)
print(f"替换前的 logits：形状 {tuple(ref.shape)}，下一个 token 的 top-3：", [tok.decode(i) for i in ref[0, -1].topk(3).indices])
print(status(hf))
```

```text title="output"
24 层，hidden 1024，词表 248320，RoPE theta 1e+07，只旋转 25% 的维度
语言模型 320 个张量，752.4M 参数（文件 bf16 -> 内存 fp32）
lm_head.weight 在文件里吗： False ；tie_word_embeddings = True
第 0 层（线性注意力）的张量： ['input_layernorm.weight', 'linear_attn.A_log', 'linear_attn.conv1d.weight', 'linear_attn.dt_bias', 'linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight', 'linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight', 'linear_attn.norm.weight', 'linear_attn.out_proj.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight']
第 3 层（全注意力）的张量：   ['input_layernorm.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight', 'self_attn.k_norm.weight', 'self_attn.k_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.q_proj.weight', 'self_attn.v_proj.weight']
和 HF 的 state_dict 对比：320 个相同，0 个不同；HF 多出来的键： ['lm_head.weight']
替换前的 logits：形状 (1, 6, 248320)，下一个 token 的 top-3： ['散步', '玩', '吧']
已替换 0/9：embed_tokens✗  input_layernorm✗  post_attention_layernorm✗  mlp✗  rotary_emb✗  self_attn✗  linear_attn✗  norm✗  lm_head✗
```

Two confirmations: the 320 tensors are equal one by one, so every later difference comes from our own operators and never from reading the weights; HF's `state_dict` has a `lm_head.weight` but the file does not — that is the shared embedding table (chapter 2).

!!! interview "How to explain it"
    To explain "what is in a model file", the answer is a table: name → shape → parameter count. Being able to split the 0.8B's 752M parameters into "vocabulary 254M + 18 linear-attention layers of 21.5M each + 6 full-attention layers of 18.4M each", and to name the operator each block belongs to, shows you have actually read the weights rather than only called an API. Add a sentence on why safetensors replaced pickle: it has an index, it can be read partially without copying, and it cannot execute arbitrary code.

## Summary {#小结}

- [x] `Config.load` takes only the fields of `text_config` we use; `load_weights` takes one of the three parts (language model / vision / MTP) by prefix and strips the prefix.
- [x] The 320 tensors are equal one by one to what HF loads; `lm_head` is not in the file, it is shared with the embedding.
- [x] The three adapters in `harness.py` translate HF's calls into our signatures; `status()` shows how many slots have been replaced.

**Files created in this chapter**: `qwen/weights.py`, `qwen/harness.py`.
