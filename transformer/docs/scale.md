# （十六）从 0.8B 到 27B 与 MoE

<p class="lead">这本书手写的是 0.8B，但代码对 Qwen3.5 全系（以及同架构的 Qwen3.6、Qwen3.8）都成立：变大只是改配置——更宽、更深、更多头；MoE 版则把前馈网络换成"256 个小专家，每个 token 走 8 个"。这一章把全系的配置摆在一起算参数账，再手写一个 MoE 层，用随机权重和 transformers 的实现对答案。</p>

!!! note "这一章跑什么"
    `ch16_scale.py`，几秒（不需要大模型的权重：参数量按公式算，MoE 用随机权重对）。

## 全系配置

| 模型 | 层数（全注意力） | hidden | 前馈 | 注意力头 / KV 组 | 线性注意力 k / v 头 | 专家 | 共用嵌入 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B | 24（6） | 1024 | 3584 | 8 / 2 | 16 / 16 | – | 是 |
| 2B | 24（6） | 2048 | 6144 | 8 / 2 | 16 / 16 | – | 是 |
| 4B | 32（8） | 2560 | 9216 | 16 / 4 | 16 / 32 | – | 是 |
| 9B | 32（8） | 4096 | 12288 | 16 / 4 | 16 / 32 | – | 否 |
| 27B（3.5 / 3.6 / 3.8 同） | 64（16） | 5120 | 17408 | 24 / 4 | 16 / 48 | – | 否 |
| 35B-A3B | 40（10） | 2048 | 512 × 256 专家 | 16 / 2 | 16 / 32 | 256 选 8 + 1 共享 | 否 |
| 122B-A10B | 48（12） | 3072 | 1024 × 256 专家 | 32 / 2 | 16 / 64 | 256 选 8 + 1 共享 | 否 |
| 397B-A17B | 60（15） | 4096 | 1024 × 512 专家 | 32 / 2 | 16 / 64 | 512 选 10 + 1 共享 | 否 |

不变的东西：每 4 层里 3 层线性注意力 + 1 层全注意力；head_dim 256；线性注意力的头维 128；RoPE 基频 1e7、只转 1/4 维；词表 248320。变的东西只有宽度、深度和头数——我们的代码里这些都从 `Config` 读。两个细节：

- **线性注意力的 value 头比 key 头多**（4B 起 16 / 32，27B 是 16 / 48）：第 7 章 `repeat_interleave` 那一行就是为这个写的，0.8B 上它不触发。
- **9B 起不共用嵌入表**：模型大了，254M 的嵌入表只占百分之几，分开训练更自由；`load_state_dict` 时 `lm_head.weight` 会出现在文件里，第 9 章那个 `assert` 要相应放宽。

## MoE：把前馈换成专家

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
```

路由器给每个 token 在 256 个专家上打分，取最高的 8 个，权重归一化后加权求和；另有一个所有 token 都走的共享专家，带一个 sigmoid 标量门。专家的权重堆成 `[E, 2·inter, H]` 和 `[E, H, inter]` 两个大张量——`gate_proj` 和 `up_proj` 拼在了一起。

35B-A3B 的账：每层 256 个专家 × 3 个 512 × 2048 的矩阵 ≈ 805M 参数，但每个 token 只用 8 个（25M）+ 共享专家（3M）；40 层共 32B 的专家参数里，每个 token 激活约 1.1B，加上注意力、嵌入等共 3B——这就是名字里的 A3B。显存要放下 35B，算力只花 3B 的。

## 参数账 + 对答案

```python title="ch16_scale.py"
"""第 16 章：按公式算全系的参数量；手写的 MoE 层和 transformers 的实现用随机权重对"""
import torch

from qwen.moe import MoE

CONFIGS = {                                                          # (层数, 全注意力层数, hidden, 前馈, 注意力头, KV 组, 线性 k 头, 线性 v 头, 共用嵌入, 专家数, 每 token 专家数, 专家前馈, 共享专家前馈)
    "0.8B": (24, 6, 1024, 3584, 8, 2, 16, 16, True, 0, 0, 0, 0),
    "2B": (24, 6, 2048, 6144, 8, 2, 16, 16, True, 0, 0, 0, 0),
    "4B": (32, 8, 2560, 9216, 16, 4, 16, 32, True, 0, 0, 0, 0),
    "9B": (32, 8, 4096, 12288, 16, 4, 16, 32, False, 0, 0, 0, 0),
    "27B": (64, 16, 5120, 17408, 24, 4, 16, 48, False, 0, 0, 0, 0),
    "35B-A3B": (40, 10, 2048, 0, 16, 2, 16, 32, False, 256, 8, 512, 512),
}
V, D, DK = 248320, 256, 128                                          # 词表、注意力 head_dim、线性注意力头维


def count(L, A, H, F, nh, nkv, lk, lv, tie, E, topk, Fe, Fs):
    attn = H * nh * D * 2 + 2 * H * nkv * D + nh * D * H              # q（两倍宽）、k、v、o
    lin = H * (2 * lk * DK + lv * DK) + H * lv * DK + 2 * H * lv + (2 * lk * DK + lv * DK) * 4 + lv * DK * H   # qkv、z、a/b、conv、out
    mlp = 3 * H * F if not E else 3 * H * Fe * E + 3 * H * Fs + H * E + H   # 稠密前馈 或 专家 + 共享专家 + 路由器 + 共享门
    active_mlp = 3 * H * F if not E else 3 * H * Fe * topk + 3 * H * Fs + H * E + H
    total = V * H * (1 if tie else 2) + A * attn + (L - A) * lin + L * mlp
    active = V * H * (1 if tie else 2) + A * attn + (L - A) * lin + L * active_mlp
    return total, active, attn, lin, mlp


print(f"{'模型':<8}{'总参数':>9}{'每 token 激活':>13}{'一层注意力':>11}{'一层线性注意力':>13}{'一层前馈':>11}")
for name, c in CONFIGS.items():
    total, active, attn, lin, mlp = count(*c)
    print(f"{name:<8}{total / 1e9:8.2f}B{active / 1e9:12.2f}B{attn / 1e6:10.1f}M{lin / 1e6:12.1f}M{mlp / 1e6:10.1f}M")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig   # 对答案：小配置 + 随机权重
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

torch.manual_seed(0)
cfg = Qwen3_5MoeTextConfig(hidden_size=64, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=48)
ref = Qwen3_5MoeSparseMoeBlock(cfg)
for p in ref.parameters():
    torch.nn.init.normal_(p, std=0.2)
mine = MoE(hidden=64, n_experts=8, top_k=2, inter=32, shared_inter=48)
mine.load_state_dict(ref.state_dict())                               # 权重名字和 HF 一致才灌得进去
x = torch.randn(2, 5, 64)
print(f"\nMoE 层（8 个专家选 2 + 共享专家）和 transformers 的最大差：{(ref(x) - mine(x)).abs().max().item():.1e}")
probs = torch.softmax(mine.gate(x.reshape(-1, 64)), dim=-1)
print("每个 token 选中的专家：", probs.topk(2).indices.tolist())
```

```text title="输出"
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

0.8B 按公式算出来的 0.75B 和第 1 章实际数出来的一致（差的是几十万个归一化权重）。27B 的一层线性注意力 0.12B、一层前馈 0.27B，是 0.8B 的 10 到 25 倍，但代码一行不用改。

!!! interview "怎么讲清楚"
    讲模型怎么变大：宽度（hidden、前馈）、深度（层数）、头数三个旋钮，架构不变；MoE 是第四个旋钮——总参数和激活参数分开，显存按总参数算、算力按激活参数算。能把 35B-A3B 的"3B"是怎么来的算出来（8 个专家 + 共享专家 + 注意力 + 嵌入），比背定义有说服力。顺便提一句 MoE 推理的难点是专家并行和负载均衡，那是《大模型推理服务》那本书的内容。

## 小结

- [x] Qwen3.5 全系共享同一套架构，差别只在宽度、深度、头数和是否 MoE；代码从 `Config` 读这些数。
- [x] MoE = 路由器 top-k + 稀疏专家 + 共享专家；手写版和 transformers 逐位一致。
- [x] 35B-A3B：总参数 35B，每 token 激活约 3B。

**这一章新建的文件**：`qwen/moe.py`。
