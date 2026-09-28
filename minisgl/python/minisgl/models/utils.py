from __future__ import annotations

import torch
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    silu_and_mul,
)

from .config import ModelConfig


class GatedMLP(BaseOP):
    """SwiGLU 前馈网络：gate 与 up 合并成一次矩阵乘，激活后再乘 down。"""

    def __init__(self, config: ModelConfig):
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}")
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [config.intermediate_size, config.intermediate_size], has_bias=False
        )
        self.down_proj = LinearRowParallel(config.intermediate_size, config.hidden_size,
                                           has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class MoEMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts.forward(hidden_states=x, router_logits=self.gate.forward(x))


class RopeAttn(BaseOP):
    """自注意力子层：qkv_proj → AttentionLayer（QK 归一化、RoPE、注意力后端）→ o_proj。"""

    def __init__(self, config: ModelConfig, layer_id: int, *, has_attn_bias: bool = False,
                 has_qk_norm: bool = False):
        head_dim = config.head_dim
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps) if has_qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps) if has_qk_norm else None
        self.attn = AttentionLayer(
            layer_id=layer_id,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=head_dim,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(head_dim * config.num_qo_heads, config.hidden_size,
                                  has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(self.attn.forward(self.qkv_proj.forward(x)))
