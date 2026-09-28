from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    """从 Hugging Face 的 config.json 中提取出引擎关心的字段。"""

    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    attention_bias: bool
    model_type: str
    architectures: List[str]
    # MoE 相关（稠密模型为 0）
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = False
    mlp_only_layers: List[int] = field(default_factory=list)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        num_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads
        # transformers 5 把 rope_theta 和 rope_scaling 合并进了 rope_parameters；旧版本分开存放
        rope = dict(getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {})
        base = getattr(config, "rope_theta", None) or rope.get("rope_theta", 10000.0)
        scaling = rope if rope.get("rope_type", "default") != "default" else None
        model_type = getattr(config, "model_type", "llama")
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=num_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", None) or num_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            rms_norm_eps=config.rms_norm_eps,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=float(base),
                scaling=scaling,
            ),
            hidden_act=config.hidden_act,
            tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
            # Qwen2 的 q/k/v 投影带偏置，但它的 config 里没有 attention_bias 这个字段
            attention_bias=bool(getattr(config, "attention_bias", model_type == "qwen2")),
            model_type=model_type,
            architectures=list(getattr(config, "architectures", None) or ["LlamaForCausalLM"]),
            num_experts=getattr(config, "num_experts", 0) or 0,
            num_experts_per_tok=getattr(config, "num_experts_per_tok", 0) or 0,
            moe_intermediate_size=getattr(config, "moe_intermediate_size", 0) or 0,
            norm_topk_prob=bool(getattr(config, "norm_topk_prob", False)),
            mlp_only_layers=list(getattr(config, "mlp_only_layers", None) or []),
        )
