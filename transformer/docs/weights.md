# （一）权重与配置

<p class="lead">从零写模型的第一步不是写算子，是把<strong>配置</strong>和<strong>权重</strong>读进来：config.json 里的几十个超参数决定每个张量的形状，safetensors 里的 320 个张量按名字对应到模块树上。这一章只写一个文件 <code>weights.py</code>，然后按模块把参数账算清楚——后面每一章写一个模块，都要先知道它的权重叫什么、有多大、长什么样。</p>

!!! note "这一章跑什么"
    `ch01_weights.py`，约 10 秒（读权重、按模块算参数账）。从这一章起，所有脚本都从 `qwen/` 包里 import，脚本本身放在包旁边：`python ch01_weights.py`。

@@files@@

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
                    out[k[len(part):]] = st.get_tensor(k).to(dtype)   # 文件里是 bf16；读进来转成 fp32（原因见下文）
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
- **文件里是 bf16，我们转成 fp32。** 不是因为 fp32 更"对"，而是第 9 章和参考实现（HF）对 logits 时两边都用 fp32，才能把误差压到 1e-5 量级，一眼看出写错没有。GPU 上实际部署用 bf16（第 15 章）。

## 权重名就是模块树

320 个张量的名字有规律：`layers.0.mlp.gate_proj.weight` 读作"第 0 层 → 前馈 → gate 投影 → 权重"。点号前面的部分就是模块树，和 Hugging Face 实现里的模块名一样（`model.layers[0].mlp`）。后面各章写自己的模块时，属性名也取一样的（`gate_proj`、`self_attn`、`linear_attn` ……），于是 `load_state_dict` 能直接按名字把权重灌进去，不用写任何映射表。

![图：模块树：权重名的前缀就是模块名，每个模块哪一章写](assets/figures/skeleton.svg){.aig-svg}

阶段一的八章不和任何参考实现对答案：每一章写好一个零件，就用一个小实验看它在做什么——嵌入空间里谁和谁近、归一化把尺度调到哪、注意力矩阵长什么样、记忆表为什么不随长度变大。到第 9 章把零件拼成整模型，全书唯一一次和 HF 对 logits（差 1e-5 量级）；对上了，说明前面每一个零件都写对了。

## 跑一遍：参数账和权重的样子

跑一遍看四件事：配置里的数字、按模块算出来的参数账、一层里有哪些张量、几个权重的数值长什么样。

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

groups = OrderedDict()                                           # 1. 按模块算参数账：名字里的层号换成 N
for k, v in W.items():
    groups.setdefault(re.sub(r"\.\d+\.", ".N.", k).rsplit(".", 1)[0], []).append(v.numel())
print("模块                             张量数  参数量   占比")
for name, sizes in groups.items():
    print(f"  {name:<30} ×{len(sizes):<3} {sum(sizes) / 1e6:7.1f}M {sum(sizes) / total:5.1%}")

print("第 0 层（线性注意力）：", sorted(sub(W, "layers.0.")))                            # 2. 一层里有什么：名字的前缀就是模块
print("第 3 层（全注意力）：  ", sorted(sub(W, "layers.3.")))

for name in ("embed_tokens.weight", "layers.0.mlp.gate_proj.weight", "layers.3.self_attn.q_proj.weight", "norm.weight", "layers.0.linear_attn.A_log"):
    v = W[name]                                                   # 3. 权重的数值长什么样：矩阵都很小、归一化的权重不是
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

几个值得记住的数：词表占 34%；线性注意力的 `in_proj_qkv`（1024 → 6144）一项就占 15%，比全注意力的 `q_proj` 大得多——因为它有 18 层；三个前馈矩阵各 11.7%。矩阵权重的标准差都在 0.01～0.02（初始化的量级），而最后那个 `norm.weight` 均值 3.3——它不是零均值的小数，第 3 章解释它为什么存成 $w-1$、为什么把尺度放大。`A_log` 是 16 个数（每个头一个），第 7 章讲它是线性注意力的"遗忘速度"。

!!! interview "怎么讲清楚"
    讲"模型文件里有什么"，答案就是一张表：名字 → 形状 → 参数量。能把 0.8B 的 752M 参数拆成"词表 254M + 18 层线性注意力各 21.5M + 6 层全注意力各 18.4M"，并说出每一块对应哪个算子，就说明真的读过权重而不是只调过 API。顺手提一句 safetensors 为什么取代了 pickle：有目录、可以零拷贝地只读一部分、没有执行任意代码的风险。

## 小结

- [x] `Config.load` 只取 `text_config` 里用得到的字段；`load_weights` 按前缀取语言模型 / 视觉 / MTP 三部分之一，并去掉前缀。
- [x] 752M 参数的账：词表 34%、线性注意力 18 层 25%、前馈 24 层 35%、全注意力 6 层 6%；`lm_head` 不在文件里，是和嵌入层共用的。
- [x] 权重名的前缀就是模块名；我们的模块按同样的名字命名，权重直接按名字灌进去。

**这一章新建的文件**：`qwen/weights.py`。
