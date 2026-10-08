"""sampler.py —— 批量采样：一个批次里每个请求可以有不同的温度、top-k、top-p 和随机种子。

做法与 vLLM 相同：top-k/top-p 用一次排序对整批完成；采样用"指数竞赛"代替 torch.multinomial：
argmax(p_i / E_i)（E_i 独立服从指数分布）恰好以概率 p_i 选中 i，而且每个请求可以用自己的随机数生成器。
"""

import torch


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """logits: [B, V]；k: [B]（0 表示不限制）；p: [B]（1.0 表示不限制）。先 top-k，再在剩下的 token 上做 top-p。"""
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    rank = torch.arange(logits.shape[-1]).expand_as(sorted_logits)
    k = torch.where(k > 0, k, logits.shape[-1])
    mask = rank >= k[:, None]                                            # top-k：排名靠后的去掉
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    mask |= (probs.cumsum(dim=-1) - probs) >= p[:, None]                 # top-p：前面的概率已经够 p 的去掉
    return torch.empty_like(logits).scatter_(1, idx, sorted_logits.masked_fill(mask, float("-inf")))


class Sampler:
    def __call__(self, logits: torch.Tensor, reqs) -> list[int]:
        params = [r.params for r in reqs]
        out = logits.argmax(dim=-1)                                          # 温度为 0 的请求：贪心
        rows = [i for i, sp in enumerate(params) if sp.temperature > 0]
        if rows:
            temps = torch.tensor([params[i].temperature for i in rows], dtype=logits.dtype)
            k = torch.tensor([params[i].top_k for i in rows])
            p = torch.tensor([params[i].top_p for i in rows], dtype=logits.dtype)
            probs = apply_top_k_top_p(logits[rows] / temps[:, None], k, p).softmax(dim=-1)
            q = torch.empty_like(probs).exponential_()
            for j, i in enumerate(rows):
                if reqs[i].generator is not None:                            # 指定了种子的请求用自己的生成器
                    q[j].exponential_(generator=reqs[i].generator)
            out[rows] = (probs / q).argmax(dim=-1)
        return out.tolist()
