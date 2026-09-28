from __future__ import annotations

from .base import BaseLLMModel
from .config import ModelConfig
from .decoder import LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM

_MODEL_REGISTRY = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


def create_model(config: ModelConfig) -> BaseLLMModel:
    arch = config.architectures[0]
    if arch not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {arch} not supported")
    return _MODEL_REGISTRY[arch](config)
