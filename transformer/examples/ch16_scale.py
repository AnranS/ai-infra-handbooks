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
