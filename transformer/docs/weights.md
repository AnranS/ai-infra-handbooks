# （一）权重与配置

<p class="lead">拆黑盒的第一步不是写算子，是把<strong>配置</strong>和<strong>权重</strong>读进来：config.json 里的几十个超参数决定每个张量的形状，safetensors 里的 320 个张量按名字对应到模块树上。这一章写两个文件：<code>weights.py</code> 负责读，<code>harness.py</code> 负责"把自己写的模块装进 HF 的模型里对答案"——后面八章每一章都用它。</p>

!!! note "这一章跑什么"
    `ch01_weights.py`，约 10 秒（读权重 + 加载 HF 模型 + 一次前向）。从这一章起，所有脚本都从 `qwen/` 包里 import，脚本本身放在包旁边：`python ch01_weights.py`。

## config.json：形状从哪来

Qwen3.5 是多模态模型，config.json 的顶层是 `architectures`、`vision_config`、`text_config` 三部分；语言模型的配置全在 `text_config` 下。我们只挑用得到的字段放进一个 dataclass：

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
    hidden_size: int                 # 1024：残差流的宽度
    num_hidden_layers: int           # 24
    layer_types: list                # 每层是 linear_attention 还是 full_attention
    num_attention_heads: int         # 全注意力：8 个 query 头
    num_key_value_heads: int         # 2 组 key/value（GQA）
    head_dim: int                    # 256
    intermediate_size: int           # 3584：前馈网络的中间宽度
    vocab_size: int                  # 248320
    rms_norm_eps: float
    rope_theta: float                # 1e7
    partial_rotary_factor: float     # 0.25：每个头只旋转前 1/4 维
    mrope_section: list              # [11, 11, 10]：三个位置轴各分到几个频率
    linear_num_key_heads: int        # 线性注意力：16 个 key 头
    linear_num_value_heads: int      # 16 个 value 头
    linear_key_head_dim: int         # 128
    linear_value_head_dim: int       # 128
    linear_conv_kernel_dim: int      # 4：短卷积的窗口
    tie_word_embeddings: bool        # 输出层和嵌入层共用权重
    eos_token_id: int

    @classmethod
    def load(cls, path):
        c = json.loads((Path(path) / "config.json").read_text())["text_config"]   # 多模态模型：语言模型的配置在 text_config 下
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
                    out[k[len(part):]] = st.get_tensor(k).to(dtype)   # 文件里是 bf16；转成 fp32 才能和 fp32 的参考实现逐位对
    return out


def sub(weights, prefix):
    """取一个子模块的权重：sub(W, "layers.0.mlp.") -> {"gate_proj.weight": ..., "up_proj.weight": ..., ...}"""
    return {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

def load(module, weights, prefix):
    """把一个子模块的权重灌进去：load(MLP(cfg), W, "layers.0.mlp.")。我们的属性名和文件里的一致，所以 state_dict 直接对上。"""
    module.load_state_dict(sub(weights, prefix), assign=True)   # assign=True：直接接管这些张量，不再复制一份
    return module
```

两个细节：

- **safetensors 不是 pickle。** 文件头是一段 JSON（每个张量的名字、dtype、形状、字节偏移），后面是裸数据，`safe_open` 可以只读一个张量、甚至只读一个切片，所以第 0 步的 `map.py` 不加载数据也能列出全部形状。
- **文件里是 bf16，我们转成 fp32。** 不是因为 fp32 更"对"，而是参考实现（HF）也用 fp32 跑，两边都用 fp32 才能把误差压到 1e-5 量级，一眼看出写错没有。GPU 上实际部署用 bf16（第 15 章）。

## harness.py：替换骨架

拆黑盒的玩法是这样的：HF 的模型是一棵模块树（`model.embed_tokens`、`model.layers[i].mlp`、`model.norm`、`lm_head` ……），每一章写好一个模块，就把树上对应的节点换成自己的，再算一次 logits 和原来比。换的时候有两个麻烦：

1. HF 调注意力时传的是它自己的一套参数（`position_embeddings`、`attention_mask`、`past_key_values` ……），我们的模块不想认识这些；
2. HF 的旋转位置编码返回的 cos/sin 带批维，还是多模态的三轴形式。

所以 harness.py 里放三个**适配器**：把 HF 的调用翻译成我们的签名。它们只在第 2–8 章用，第 9 章整模型拼起来之后就不需要了。

![图：HF 模型的模块树与每个位置在哪一章被换掉](assets/figures/skeleton.svg){.aig-svg}

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
        cos, sin = position_embeddings            # HF 的 cos/sin 带批维 [B, T, 64]；纯文本时每个样本的位置相同，取第 0 个
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

## 跑一遍：我们读到的和 HF 的一样吗

这一章还没替换任何模块，先确认两件事：我们按名字读出来的 320 个张量，和 HF 模型加载后 `state_dict()` 里的逐个相等；以及记下替换前的 logits，后面每章都要和它比。

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

hf, tok = load_hf(path)                                                   # 和 HF 加载的逐个对
sd = hf.state_dict()
mismatch = [k for k, v in W.items() if not torch.equal(v, sd["model." + k])]
print(f"和 HF 的 state_dict 对比：{len(W) - len(mismatch)} 个相同，{len(mismatch)} 个不同；HF 多出来的键：",
      sorted(set(k[6:] for k in sd if k.startswith("model.")) - set(W)) + [k for k in sd if not k.startswith("model.")])

ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)
print(f"替换前的 logits：形状 {tuple(ref.shape)}，下一个 token 的 top-3：", [tok.decode(i) for i in ref[0, -1].topk(3).indices])
print(status(hf))
```

```text title="输出"
24 层，hidden 1024，词表 248320，RoPE theta 1e+07，只旋转 25% 的维度
语言模型 320 个张量，752.4M 参数（文件 bf16 -> 内存 fp32）
lm_head.weight 在文件里吗： False ；tie_word_embeddings = True
第 0 层（线性注意力）的张量： ['input_layernorm.weight', 'linear_attn.A_log', 'linear_attn.conv1d.weight', 'linear_attn.dt_bias', 'linear_attn.in_proj_a.weight', 'linear_attn.in_proj_b.weight', 'linear_attn.in_proj_qkv.weight', 'linear_attn.in_proj_z.weight', 'linear_attn.norm.weight', 'linear_attn.out_proj.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight']
第 3 层（全注意力）的张量：   ['input_layernorm.weight', 'mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'post_attention_layernorm.weight', 'self_attn.k_norm.weight', 'self_attn.k_proj.weight', 'self_attn.o_proj.weight', 'self_attn.q_norm.weight', 'self_attn.q_proj.weight', 'self_attn.v_proj.weight']
和 HF 的 state_dict 对比：320 个相同，0 个不同；HF 多出来的键： ['lm_head.weight']
替换前的 logits：形状 (1, 6, 248320)，下一个 token 的 top-3： ['散步', '玩', '吧']
已替换 0/9：embed_tokens✗  input_layernorm✗  post_attention_layernorm✗  mlp✗  rotary_emb✗  self_attn✗  linear_attn✗  norm✗  lm_head✗
```

两点确认：320 个张量逐个相等，所以后面所有误差都来自我们自己的算子，和读权重无关；HF 的 `state_dict` 里有 `lm_head.weight` 但文件里没有——那就是共用的那张嵌入表（第 2 章）。

!!! interview "怎么讲清楚"
    讲"模型文件里有什么"，答案就是一张表：名字 → 形状 → 参数量。能把 0.8B 的 752M 参数拆成"词表 254M + 18 层线性注意力各 21.5M + 6 层全注意力各 18.4M"，并说出每一块对应哪个算子，就说明真的读过权重而不是只调过 API。顺手提一句 safetensors 为什么取代了 pickle：有目录、可以零拷贝地只读一部分、没有执行任意代码的风险。

## 小结

- [x] `Config.load` 只取 `text_config` 里用得到的字段；`load_weights` 按前缀取语言模型 / 视觉 / MTP 三部分之一，并去掉前缀。
- [x] 320 个张量和 HF 加载的逐个相等；`lm_head` 不在文件里，是和嵌入层共用的。
- [x] `harness.py` 的三个适配器把 HF 的调用翻译成我们的签名；`status()` 显示换了几块。

**这一章新建的文件**：`qwen/weights.py`、`qwen/harness.py`。
