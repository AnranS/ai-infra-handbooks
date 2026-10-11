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
