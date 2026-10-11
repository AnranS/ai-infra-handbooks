# 16. From 0.8B to 27B and MoE

<p class="lead">This book writes the 0.8B by hand, but the code holds for the whole Qwen3.5 family (and Qwen3.6 and Qwen3.8, which share the architecture): getting bigger is only a change of configuration — wider, deeper, more heads; the MoE versions replace the feed-forward network with "256 small experts, 8 of them per token". This chapter lines up the configurations of the whole family and does the parameter arithmetic, then writes an MoE layer by hand and checks it against the transformers implementation with random weights.</p>

!!! note "What this chapter runs"
    `ch16_scale.py`, a few seconds (no large-model weights needed: parameter counts come from formulas, and the MoE is checked with random weights).

@@files@@

## The whole family {#全系配置}

| Model | Layers (full attention) | hidden | FFN | attention heads / KV groups | linear-attention k / v heads | experts | tied embedding |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B | 24 (6) | 1024 | 3584 | 8 / 2 | 16 / 16 | – | yes |
| 2B | 24 (6) | 2048 | 6144 | 8 / 2 | 16 / 16 | – | yes |
| 4B | 32 (8) | 2560 | 9216 | 16 / 4 | 16 / 32 | – | yes |
| 9B | 32 (8) | 4096 | 12288 | 16 / 4 | 16 / 32 | – | no |
| 27B (same for 3.5 / 3.6 / 3.8) | 64 (16) | 5120 | 17408 | 24 / 4 | 16 / 48 | – | no |
| 35B-A3B | 40 (10) | 2048 | 512 × 256 experts | 16 / 2 | 16 / 32 | 256 choose 8 + 1 shared | no |
| 122B-A10B | 48 (12) | 3072 | 1024 × 256 experts | 32 / 2 | 16 / 64 | 256 choose 8 + 1 shared | no |
| 397B-A17B | 60 (15) | 4096 | 1024 × 512 experts | 32 / 2 | 16 / 64 | 512 choose 10 + 1 shared | no |

What stays the same: 3 linear-attention layers + 1 full-attention layer in every group of 4; head_dim 256; linear-attention head dimension 128; RoPE base 1e7 with a quarter rotated; vocabulary 248320. What changes is only width, depth and head counts — and our code reads all of them from `Config`. Two details:

- **Linear attention has more value heads than key heads** (16 / 32 from 4B up, 16 / 48 at 27B): the `repeat_interleave` line in chapter 7 exists for this; it does not fire on the 0.8B.
- **From 9B up the embedding table is not tied**: on a large model the 254M table is only a few percent, and training the two separately is freer; `lm_head.weight` then appears in the file, and chapter 9's `assert` has to be relaxed accordingly.

## MoE: experts in place of the feed-forward {#moe把前馈换成专家}

```python title="qwen/moe.py"
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
        for e in top_idx.unique().tolist():                                   # compute only the selected experts; each expert processes all tokens that chose it at once
            rows, slot = (top_idx == e).nonzero(as_tuple=True)                # which tokens chose it and in which of their slots
            gate, up = F.linear(x[rows], self.gate_up_proj[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(gate) * up, self.down_proj[e])
            out.index_add_(0, rows, y * top_weight[rows, slot, None])         # add back to each token with its routing weight
        return out


class MoE(nn.Module):
    def __init__(self, hidden, n_experts, top_k, inter, shared_inter):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden, n_experts, bias=False)                  # router: every token scores every expert
        self.experts = Experts(n_experts, hidden, inter)
        self.shared_expert = nn.ModuleDict(dict(gate_proj=nn.Linear(hidden, shared_inter, bias=False),
                                                up_proj=nn.Linear(hidden, shared_inter, bias=False),
                                                down_proj=nn.Linear(shared_inter, hidden, bias=False)))
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)           # the shared expert has a scalar gate too

    def forward(self, x):
        B, T, H = x.shape
        flat = x.reshape(-1, H)
        probs = F.softmax(self.gate(flat).float(), dim=-1)                    # [N, E]
        top_weight, top_idx = probs.topk(self.top_k, dim=-1)                  # each token takes the k highest-scoring experts
        top_weight = (top_weight / top_weight.sum(-1, keepdim=True)).to(x.dtype)   # the k selected weights are renormalized
        out = self.experts(flat, top_idx, top_weight)
        s = self.shared_expert
        shared = s["down_proj"](F.silu(s["gate_proj"](flat)) * s["up_proj"](flat))
        out = out + torch.sigmoid(self.shared_expert_gate(flat)) * shared
        return out.reshape(B, T, H)
```

The router scores every token over the 256 experts, takes the 8 highest, renormalizes their weights and sums the experts' outputs with them; in addition there is a shared expert that every token passes through, with a sigmoid scalar gate. The experts' weights are stacked into two big tensors, `[E, 2·inter, H]` and `[E, H, inter]` — `gate_proj` and `up_proj` are concatenated.

The 35B-A3B arithmetic: per layer, 256 experts × 3 matrices of 512 × 2048 ≈ 805M parameters, but each token uses only 8 of them (25M) plus the shared expert (3M); of the 32B of expert parameters across 40 layers, about 1.1B are active per token, and with attention, the embedding and the rest the total is 3B — the A3B in the name. Memory has to hold 35B; compute is spent on 3B.

## The parameter budget, and the check {#参数账--对答案}

```python title="ch16_scale.py" ci="loose"
"""第 16 章：按公式算全系的参数量；手写的 MoE 层和 transformers 的实现用随机权重对"""
import torch

from qwen.moe import MoE

CONFIGS = {                                                          # (layers, full-attention layers, hidden, FFN, attention heads, KV groups, linear k heads, linear v heads, tied embedding, experts, experts per token, expert FFN, shared-expert FFN)
    "0.8B": (24, 6, 1024, 3584, 8, 2, 16, 16, True, 0, 0, 0, 0),
    "2B": (24, 6, 2048, 6144, 8, 2, 16, 16, True, 0, 0, 0, 0),
    "4B": (32, 8, 2560, 9216, 16, 4, 16, 32, True, 0, 0, 0, 0),
    "9B": (32, 8, 4096, 12288, 16, 4, 16, 32, False, 0, 0, 0, 0),
    "27B": (64, 16, 5120, 17408, 24, 4, 16, 48, False, 0, 0, 0, 0),
    "35B-A3B": (40, 10, 2048, 0, 16, 2, 16, 32, False, 256, 8, 512, 512),
}
V, D, DK = 248320, 256, 128                                          # vocabulary, attention head_dim, linear-attention head dim


def count(L, A, H, F, nh, nkv, lk, lv, tie, E, topk, Fe, Fs):
    attn = H * nh * D * 2 + 2 * H * nkv * D + nh * D * H              # q (twice as wide), k, v, o
    lin = H * (2 * lk * DK + lv * DK) + H * lv * DK + 2 * H * lv + (2 * lk * DK + lv * DK) * 4 + lv * DK * H   # qkv, z, a/b, conv, out
    mlp = 3 * H * F if not E else 3 * H * Fe * E + 3 * H * Fs + H * E + H   # dense FFN, or experts + shared expert + router + shared gate
    active_mlp = 3 * H * F if not E else 3 * H * Fe * topk + 3 * H * Fs + H * E + H
    total = V * H * (1 if tie else 2) + A * attn + (L - A) * lin + L * mlp
    active = V * H * (1 if tie else 2) + A * attn + (L - A) * lin + L * active_mlp
    return total, active, attn, lin, mlp


print(f"{'模型':<8}{'总参数':>9}{'每 token 激活':>13}{'一层注意力':>11}{'一层线性注意力':>13}{'一层前馈':>11}")
for name, c in CONFIGS.items():
    total, active, attn, lin, mlp = count(*c)
    print(f"{name:<8}{total / 1e9:8.2f}B{active / 1e9:12.2f}B{attn / 1e6:10.1f}M{lin / 1e6:12.1f}M{mlp / 1e6:10.1f}M")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig   # check the answer: a tiny config with random weights
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

torch.manual_seed(0)
cfg = Qwen3_5MoeTextConfig(hidden_size=64, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=48)
ref = Qwen3_5MoeSparseMoeBlock(cfg)
for p in ref.parameters():
    torch.nn.init.normal_(p, std=0.2)
mine = MoE(hidden=64, n_experts=8, top_k=2, inter=32, shared_inter=48)
mine.load_state_dict(ref.state_dict())                               # the weight names must match HF's for the load to work
x = torch.randn(2, 5, 64)
print(f"\nMoE 层（8 个专家选 2 + 共享专家）和 transformers 的最大差：{(ref(x) - mine(x)).abs().max().item():.1e}")
probs = torch.softmax(mine.gate(x.reshape(-1, 64)), dim=-1)
print("每个 token 选中的专家：", probs.topk(2).indices.tolist())
```

```text title="output"
模型            总参数   每 token 激活      一层注意力      一层线性注意力       一层前馈
0.8B        0.75B        0.75B       7.3M        10.5M      11.0M
2B          1.88B        1.88B      14.7M        21.1M      37.7M
4B          4.21B        4.21B      36.7M        42.1M      70.8M
9B          8.95B        8.95B      58.7M        67.4M     151.0M
27B        26.90B       26.90B     104.9M       115.9M     267.4M
35B-A3B    34.66B        3.45B      27.3M        33.7M     809.0M

MoE 层（8 个专家选 2 + 共享专家）和 transformers 的最大差：0.0e+00
每个 token 选中的专家： [[3, 5], [3, 5], [3, 2], [2, 7], [7, 1], [6, 5], [2, 3], [3, 1], [2, 6], [7, 2]]
```

The 0.75B computed from the formula for the 0.8B agrees with what chapter 1 actually counted (the difference is a few hundred thousand normalization weights). The 27B's linear-attention layer is 0.12B and its feed-forward layer 0.27B, 10 to 25 times the 0.8B's, with not a line of code changed.

!!! interview "How to explain it"
    How a model grows: width (hidden, FFN), depth (layers) and head count are three knobs, with the architecture unchanged; MoE is the fourth — total parameters and active parameters come apart, memory is sized by the total and compute by the active ones. Working out where the "3B" of 35B-A3B comes from (8 experts + the shared expert + attention + the embedding) is more convincing than reciting a definition. Mention that the hard parts of MoE inference are expert parallelism and load balancing — that is the *Inference Systems* book.

## Summary {#小结}

- [x] The whole Qwen3.5 family shares one architecture; the differences are width, depth, head counts and whether it is MoE; the code reads these numbers from `Config`.
- [x] MoE = a top-k router + sparse experts + a shared expert; the hand-written version matches transformers element by element.
- [x] 35B-A3B: 35B parameters in total, about 3B active per token.

**File created in this chapter**: `qwen/moe.py`.
