"""mini_llm.py —— 从零实现的 LLaMA / Qwen2 / Qwen3 结构的解码器模型（约 200 行），可以加载真实权重。

结构：Embedding → N × [RMSNorm → 注意力(GQA + QK-Norm + RoPE) → 残差 → RMSNorm → SwiGLU MLP → 残差] → RMSNorm → LM Head
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False          # Qwen2 的 q/k/v 投影带偏置，LLaMA、Qwen3 不带
    qk_norm: bool = False                 # Qwen3：q、k 在 RoPE 之前按头各做一次 RMSNorm
    tie_word_embeddings: bool = False     # 输出层是否与词嵌入共享权重
    head_dim: int | None = None

    @property
    def hd(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        c = json.loads(Path(path).read_text())
        return cls(
            vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", c["num_attention_heads"]),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6), rope_theta=c.get("rope_theta", 10000.0),
            attention_bias=c.get("attention_bias", c.get("model_type") == "qwen2"), qk_norm=c.get("model_type") == "qwen3",
            tie_word_embeddings=c.get("tie_word_embeddings", False), head_dim=c.get("head_dim"),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                              # 统计量用 FP32 计算
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """返回 [T, head_dim] 的 cos、sin 表。第 i 对维度的旋转频率为 theta^(-2i/d)。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]        # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)                       # [T, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: [B, H, T, D]；把第 i 维和第 i + D/2 维看成一个二维向量，按位置旋转。"""
    return (x * cos + rotate_half(x) * sin).to(x.dtype)


class KVCache:
    """每层保存到目前为止所有 token 的 K、V：[B, n_kv_heads, S, head_dim]。"""

    def __init__(self, num_layers: int):
        self.k = [None] * num_layers
        self.v = [None] * num_layers

    def update(self, layer: int, k, v):
        if self.k[layer] is not None:
            k = torch.cat([self.k[layer], k], dim=2)
            v = torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    @property
    def length(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.layer, self.nh, self.nkv, self.hd = layer, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hd
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        # QK-Norm：每个头的 q、k 各自做 RMSNorm（权重维度 head_dim）；没有 QK-Norm 的模型用恒等映射，调用方不必判断
        norm = (lambda: RMSNorm(self.hd, cfg.rms_norm_eps)) if cfg.qk_norm else nn.Identity
        self.q_norm, self.k_norm = norm(), norm()

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # [B, nh, T, hd]
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)   # [B, nkv, T, hd]
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)                               # 在 RoPE 之前
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(self.layer, k, v)                           # 拼上历史 token 的 K、V
        S = k.shape[2]
        rep = self.nh // self.nkv                                           # GQA：每个 KV 头服务 rep 个 query 头
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)             # [B, nh, T, S]
        # 第 i 个新 token 的绝对位置是 S - T + i，只能看到位置不超过它的 key
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = scores.float().softmax(dim=-1).to(q.dtype)
        out = (probs @ v).transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))   # SwiGLU


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config, layer: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, layer)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)   # 残差连接
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, cache: KVCache | None = None):
        """input_ids: [B, T] → logits: [B, T, vocab]。有 cache 时，只需传入新 token。"""
        start = cache.length if cache is not None else 0
        positions = torch.arange(start, start + input_ids.shape[1], device=input_ids.device)
        cos, sin = rope_cos_sin(positions, self.cfg.hd, self.cfg.rope_theta)
        x = self.embed_tokens(input_ids)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, cache)
        return self.lm_head(self.norm(x))

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype=torch.float32) -> "Transformer":
        """加载 Hugging Face 格式（config.json + *.safetensors）的 LLaMA / Qwen2 / Qwen3 权重。"""
        from safetensors.torch import load_file

        path = Path(path)
        model = cls(Config.from_json(path / "config.json"))
        state = {}
        for f in sorted(path.glob("*.safetensors")):
            state.update(load_file(f))
        state = {k.removeprefix("model."): v for k, v in state.items()}   # HF 的层名多一个 "model." 前缀
        if model.cfg.tie_word_embeddings:
            state.pop("lm_head.weight", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not (model.cfg.tie_word_embeddings and m == "lm_head.weight")]
        assert not missing and not unexpected, (missing, unexpected)
        return model.to(dtype).eval()


@torch.no_grad()
def generate(model: Transformer, input_ids, max_new_tokens: int, pick=lambda logits: logits.argmax(-1),
             eos_token_id: int | None = None):
    """自回归生成：prefill 一次处理整个提示词，之后每步只输入 1 个新 token（decode）。"""
    cache = KVCache(model.cfg.num_hidden_layers)
    logits = model(input_ids, cache)                  # prefill
    out = []
    for _ in range(max_new_tokens):
        next_id = pick(logits[:, -1, :])              # [B]
        out.append(next_id)
        if eos_token_id is not None and bool((next_id == eos_token_id).all()):
            break
        logits = model(next_id[:, None], cache)       # decode：只算新 token
    return torch.stack(out, dim=1)
