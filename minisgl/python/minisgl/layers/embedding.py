from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.kernel import indexing
from minisgl.utils import div_ceil

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """词表并行的嵌入层：每个 rank 保存词表的一段，查不到的 token 填 0，最后 all-reduce 相加。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        tp_info = get_tp_info()
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start = self.num_embeddings_tp * tp_info.rank
        finish = min(start + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start, finish - start)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = indexing(self.weight, x, self.vocab_range if self.tp_size > 1 else None)
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """输出层。prefill 时只取每个请求最后一个位置的隐状态去算 logits——其余位置的 logits 用不到。"""

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False,
                 tie_word_embeddings: bool = False,
                 tied_embedding: VocabParallelEmbedding | None = None):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], *, prefix: str = "",
                        _internal: bool = False) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        # 共享词嵌入时 checkpoint 里可能仍然带着 lm_head.weight，丢掉即可
        state_dict.pop(f"{prefix}.weight", None)
        state_dict.pop(f"{prefix}.bias", None)

    def state_dict(self, *, prefix: str = "", result: Dict[str, torch.Tensor] | None = None):
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = get_global_ctx().batch
        bs = batch.size
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(bs)].contiguous()
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        # all_gather 沿第 0 维拼接：[tp, bs, V/tp] -> 转置成 [bs, tp * V/tp]
        gathered = self._comm.all_gather(logits).view(self.tp_size, *logits.shape)
        gathered = gathered.permute(1, 0, 2).reshape(logits.shape[0], -1)
        return gathered[:, : self.num_embeddings]
