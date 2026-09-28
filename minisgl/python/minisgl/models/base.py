from __future__ import annotations

from abc import abstractmethod

import torch
from minisgl.layers import BaseOP


class BaseLLMModel(BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """不接收参数：输入从 get_global_ctx().batch 读取，返回 [batch_size, vocab] 的 logits。"""
