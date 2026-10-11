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
