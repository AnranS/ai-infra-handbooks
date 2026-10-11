"""混合专家（MoE）：Qwen3.5-35B-A3B 这类模型把每层的前馈网络换成 256 个小专家，每个 token 只走其中 8 个，外加一个所有人都走的共享专家。

0.8B 没有 MoE。这个文件按 Qwen3.5-MoE 的权重命名写，用随机权重和 transformers 的实现对答案（第 16 章）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts(nn.Module):
    """所有专家的权重堆成两个大张量：gate 和 up 拼在一起 [E, 2*inter, H]，down [E, H, inter]"""

    def __init__(self, n_experts, hidden, inter):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(n_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.empty(n_experts, hidden, inter))

    def forward(self, x, top_idx, top_weight):
        """x: [N, H]；top_idx / top_weight: [N, k] 每个 token 选中的专家和权重 -> [N, H]"""
        out = torch.zeros_like(x)
        for e in top_idx.unique().tolist():                                   # 只算被选中的专家，每个专家一次处理所有选它的 token
            rows, slot = (top_idx == e).nonzero(as_tuple=True)                # 哪些 token 选了它、排在它们的第几个名额
            gate, up = F.linear(x[rows], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e])
            out.index_add_(0, rows, y * top_weight[rows, slot, None])         # 按路由权重加回各自的 token
        return out


class MoE(nn.Module):
    def __init__(self, hidden, n_experts, top_k, inter, shared_inter):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden, n_experts, bias=False)                  # 路由器：每个 token 给每个专家打分
        self.experts = Experts(n_experts, hidden, inter)
        self.shared_expert = nn.ModuleDict(dict(gate_proj=nn.Linear(hidden, shared_inter, bias=False),
                                                up_proj=nn.Linear(hidden, shared_inter, bias=False),
                                                down_proj=nn.Linear(shared_inter, hidden, bias=False)))
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)           # 共享专家也有一个标量门

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        probs = F.softmax(self.gate(flat).float(), dim=-1)                    # [N, E]
        top_weight, top_idx = probs.topk(self.top_k, dim=-1)                  # 每个 token 取分最高的 k 个专家
        top_weight = (top_weight / top_weight.sum(-1, keepdim=True)).to(x.dtype)   # 选中的 k 个权重重新归一化
        out = self.experts(flat, top_idx, top_weight)
        s = self.shared_expert
        shared = s["down_proj"](F.silu(s["gate_proj"](flat)) * s["up_proj"](flat))
        out = out + torch.sigmoid(self.shared_expert_gate(flat)) * shared
        return out.reshape(B, T, H)
