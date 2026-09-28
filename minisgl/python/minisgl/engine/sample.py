from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import pin

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None  # None 表示整个 batch 都是贪心解码
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=pin(device)).to(device, non_blocking=True)


def sample_torch(logits: torch.Tensor, temperatures: torch.Tensor, top_k: torch.Tensor | None,
                 top_p: torch.Tensor | None) -> torch.Tensor:
    """每个请求可以有各自的温度、top-k、top-p。与 FlashInfer 一样先做 top-k、再做 top-p。"""
    probs = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
    if top_k is not None or top_p is not None:
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        if top_k is not None:
            rank = torch.arange(probs.shape[1], device=probs.device).unsqueeze(0)
            sorted_probs = sorted_probs.masked_fill(rank >= top_k.unsqueeze(1), 0.0)
        if top_p is not None:
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            before = sorted_probs.cumsum(dim=-1) - sorted_probs  # 排在它前面的概率之和
            sorted_probs = sorted_probs.masked_fill(before >= top_p.unsqueeze(1), 0.0)
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def sample_flashinfer(logits: torch.Tensor, temperatures: torch.Tensor,
                      top_k: torch.Tensor | None, top_p: torch.Tensor | None) -> torch.Tensor:
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """在调度器一侧（CPU）为整个 batch 准备采样参数张量。"""
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)
        MIN_T = MIN_P = 1e-6
        # 混合 batch 里的贪心请求：温度取极小值，softmax 退化成 one-hot，效果等于 argmax
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        top_k = top_p = None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(make_device_tensor(ts, torch.float32, self.device), top_k, top_p)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        if args.temperatures is None:
            return torch.argmax(logits, dim=-1)
        impl = sample_torch
        if logits.is_cuda:
            try:
                import flashinfer.sampling  # noqa: F401

                impl = sample_flashinfer
            except ImportError:
                pass
        return impl(logits.float(), args.temperatures, args.top_k, args.top_p)
