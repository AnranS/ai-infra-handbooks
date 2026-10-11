"""Gated DeltaNet：线性注意力层，24 层里的 18 层是它。

它没有"所有 token 两两打分"的注意力矩阵，而是每个头维护一张固定大小的记忆表 S（128×128）。
每来一个 token 就按 delta 规则改写一次：先按 g 衰减，再把"新值 v 和表里已有预测之差"按 β 写进去；
输出是 q 去表里查询。token 再多，表也不会变大——这就是它能撑 262K 上下文的原因。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen.chunk import chunked_delta_rule
from qwen.norm import GatedRMSNorm, l2norm



def delta_rule(q, k, v, g, beta, S=None):
    """逐 token 递推。q, k: [B, T, 头, dk]，v: [B, T, 头, dv]；g（对数衰减，≤0）和 beta（写入强度，0~1）: [B, T, 头]。
    S 是开始时的记忆表（没有就从空表开始）；返回输出和算完之后的表。"""
    B, T, n_head, dk = q.shape
    dtype = v.dtype
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))   # 记忆表和中间量一律 fp32，bf16 攒不住几百步的累加
    q, k = l2norm(q), l2norm(k)                                   # q、k 先归一化到单位长度
    q = q / dk ** 0.5
    if S is None:
        S = q.new_zeros(B, n_head, dk, v.shape[-1])               # 每个头一张 128×128 的记忆表，初始为空
    out = torch.empty_like(v)
    for t in range(T):
        S = S * g[:, t].exp()[:, :, None, None]                   # 1. 衰减：整张表乘一个 0~1 的数，忘掉一点旧的
        k_t, v_t = k[:, t], v[:, t]                               # [B, 头, dk]，[B, 头, dv]
        pred = torch.einsum("bhk,bhkv->bhv", k_t, S)              # 2. 用 k 查表：表里认为 v 应该是什么
        S = S + torch.einsum("bhk,bhv->bhkv", k_t, (v_t - pred) * beta[:, t, :, None])   # 3. 只写入差值（delta）
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S)     # 4. 用 q 查表得到这个 token 的输出
    return out.to(dtype), S


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H = cfg.hidden_size
        self.n_k, self.n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.n_k * self.dk, self.n_v * self.dv            # 2048, 2048
        conv_dim = 2 * self.key_dim + self.value_dim                                       # q、k、v 拼在一起做卷积：6144
        self.in_proj_qkv = nn.Linear(H, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(H, self.value_dim, bias=False)                          # 输出门
        self.in_proj_b = nn.Linear(H, self.n_v, bias=False)                                # β：每个头一个数
        self.in_proj_a = nn.Linear(H, self.n_v, bias=False)                                # 衰减里随输入变的那部分
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, cfg.linear_conv_kernel_dim, groups=conv_dim,
                                padding=cfg.linear_conv_kernel_dim - 1, bias=False)        # 深度可分的因果短卷积：每个通道只看自己前 3 个 token
        self.dt_bias = nn.Parameter(torch.ones(self.n_v))
        self.A_log = nn.Parameter(torch.zeros(self.n_v))
        self.norm = GatedRMSNorm(self.dv, cfg.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, H, bias=False)

    def forward(self, x, state=None, mask=None):
        B, T, _ = x.shape
        if mask is not None:
            x = x * mask[..., None]                                                        # 批处理时的填充位：[B, T]，清零后既不写进记忆表也不进卷积
        qkv = self.in_proj_qkv(x).transpose(1, 2)                                          # [B, 6144, T]
        window = state.conv if state is not None else qkv.new_zeros(B, qkv.shape[1], self.conv1d.padding[0])
        qkv = torch.cat([window, qkv], dim=-1)                                             # 接上前 3 个 token（没有历史就是 0）
        if state is not None:
            state.conv = qkv[..., -self.conv1d.padding[0]:]                                # 留下最后 3 个 token 给下一次
        qkv = F.silu(F.conv1d(qkv, self.conv1d.weight, groups=qkv.shape[1])).transpose(1, 2)   # 因果卷积 + SiLU：输出长度正好是 T
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q, k, v = q.view(B, T, self.n_k, self.dk), k.view(B, T, self.n_k, self.dk), v.view(B, T, self.n_v, self.dv)
        if self.n_v > self.n_k:                                                            # 大模型里 value 头比 key 头多：q、k 复制几份
            q, k = q.repeat_interleave(self.n_v // self.n_k, 2), k.repeat_interleave(self.n_v // self.n_k, 2)
        beta = self.in_proj_b(x).sigmoid()                                                 # [B, T, 头]，0~1
        g = -self.A_log.exp() * F.softplus(self.in_proj_a(x) + self.dt_bias)               # 对数衰减，永远 ≤ 0
        S = state.S if state is not None else None
        rule = delta_rule if T <= 16 else chunked_delta_rule                               # 几个 token 用递推，一段文本用分块（分块版要补到 64 的整数倍）
        out, S = rule(q, k, v, g, beta, S)
        if state is not None:
            state.S = S
        out = self.norm(out, self.in_proj_z(x).view(B, T, self.n_v, self.dv))             # 按头做门控归一化
        return self.out_proj(out.reshape(B, T, -1))
