"""moe.py —— 一个 top-k 路由的 MoE 层（带共享专家），以及两种等价的前向实现。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts, top_k, n_shared=1):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, d_ff) for _ in range(n_shared))

    def route(self, x):                                   # x: [N, d]，N 个 token
        probs = self.router(x).softmax(dim=-1)            # [N, E]
        weights, idx = probs.topk(self.top_k, dim=-1)     # [N, k]
        weights = weights / weights.sum(-1, keepdim=True) # 选中的 k 个权重重新归一化
        return weights, idx, probs

    def forward_naive(self, x):
        """逐个 token、逐个专家地计算：正确但低效。"""
        weights, idx, _ = self.route(x)
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(self.top_k):
                e = idx[t, j].item()
                out[t] += weights[t, j] * self.experts[e](x[t])
        return out + sum(s(x) for s in self.shared)

    def forward_grouped(self, x):
        """按专家分组：每个专家对分给它的全部 token 做一次矩阵乘法。"""
        weights, idx, _ = self.route(x)
        flat_expert = idx.flatten()                                   # [N*k]：每个 (token, 槽位) 去哪个专家
        flat_token = torch.arange(x.shape[0]).repeat_interleave(self.top_k)
        order = flat_expert.argsort(stable=True)                      # 按专家排序，同一专家的 token 连在一起
        counts = torch.bincount(flat_expert, minlength=len(self.experts))
        out = torch.zeros_like(x)
        start = 0
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            sel = order[start:start + n]
            tok = flat_token[sel]
            y = self.experts[e](x[tok])                               # 一次处理 n 个 token
            out.index_add_(0, tok, y * weights.flatten()[sel, None])  # 按权重加回原位置
            start += n
        return out + sum(s(x) for s in self.shared)
