"""Llama、Qwen2、Qwen3、Qwen3-MoE 共用的 decoder-only 结构。

官方实现为每种模型各写一个文件（结构几乎相同）；这里把差异收成三个开关：
q/k/v 投影是否带偏置（Qwen2）、是否有 QK 归一化（Qwen3）、MLP 是否换成 MoE（Qwen3-MoE）。
"""

from __future__ import annotations

from typing import Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding

from .base import BaseLLMModel
from .config import ModelConfig
from .utils import GatedMLP, MoEMLP, RopeAttn


class DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, has_qk_norm: bool, use_moe: bool):
        self.self_attn = RopeAttn(config, layer_id, has_attn_bias=config.attention_bias,
                                  has_qk_norm=has_qk_norm)
        self.mlp = MoEMLP(config) if use_moe else GatedMLP(config)
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor,
                residual: torch.Tensor | None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class DecoderModel(BaseOP):
    def __init__(self, config: ModelConfig, *, has_qk_norm: bool, moe: bool):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([
            DecoderLayer(config, i, has_qk_norm=has_qk_norm,
                         use_moe=moe and i not in config.mlp_only_layers)
            for i in range(config.num_layers)
        ])
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class CausalLM(BaseLLMModel):
    has_qk_norm = False
    moe = False

    def __init__(self, config: ModelConfig):
        self.model = DecoderModel(config, has_qk_norm=self.has_qk_norm, moe=self.moe)
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(self) -> torch.Tensor:
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(hidden)


class LlamaForCausalLM(CausalLM):
    pass


class Qwen2ForCausalLM(CausalLM):
    pass


class Qwen3ForCausalLM(CausalLM):
    has_qk_norm = True


class Qwen3MoeForCausalLM(CausalLM):
    has_qk_norm = True
    moe = True
