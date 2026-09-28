"""用 Triton 写的 fused MoE（思路与 vLLM / SGLang 的 fused_moe 相同，做了简化）。

参考实现按专家循环，每个专家一次小矩阵乘，专家一多 kernel 启动就多，GPU 也吃不满。
fused MoE 把所有 (token, 专家) 对按专家排好序，每个专家的那一段补齐到 BLOCK_M 的整数倍，
于是一个 [BLOCK_M, BLOCK_N] 的输出块只属于一个专家——整个 MoE 层只需要两次 kernel 启动：

    hidden --(按排序后的下标取行)--> GEMM(w1[专家]) --> silu*up --> GEMM(w2[专家]) * 路由权重 --> 按 token 求和

CPU 上可以用 TRITON_INTERPRET=1 在解释器里运行这个 kernel 来验证正确性。
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from minisgl.kernel import silu_and_mul

from .base import BaseMoeBackend, select_experts


@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    N, K, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr, TOP_K: tl.constexpr,
):
    """C[t, :] = A[t // TOP_K, :] @ B[专家].T（可选再乘路由权重），t 是排序后的 (token, 专家) 对编号。"""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = pid // num_pid_n, pid % num_pid_n
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    token_mask = offs_token < num_valid_tokens  # 补齐用的占位符不参与计算
    expert = tl.load(expert_ids_ptr + pid_m)  # 这个块属于哪个专家
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // TOP_K) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remain = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remain), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if MUL_ROUTED_WEIGHT:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=token_mask[:, None] & (offs_n[None, :] < N))


def moe_align_block_size(topk_ids: torch.Tensor, block_m: int,
                         num_experts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 (token, 专家) 对按专家排序，每个专家的段补齐到 block_m 的倍数。

    返回 sorted_ids（元素是 topk_ids.flatten() 中的下标，补齐位填 num_pairs）和每个块的专家号。
    """
    flat = topk_ids.flatten().long()
    num_pairs = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_m - 1) // block_m * block_m
    seg_start = torch.cumsum(padded, 0) - padded  # 每个专家的段在输出中的起点
    order = torch.argsort(flat, stable=True)  # 按专家排序后的 pair 下标
    sorted_expert = flat[order]
    rank = torch.arange(num_pairs, device=flat.device) - (torch.cumsum(counts, 0) - counts)[sorted_expert]
    sorted_ids = torch.full((int(padded.sum()),), num_pairs, dtype=torch.int32, device=flat.device)
    sorted_ids[seg_start[sorted_expert] + rank] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=flat.device),
                                         padded // block_m).to(torch.int32)
    return sorted_ids, expert_ids


def _invoke(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk_weights: torch.Tensor,
            sorted_ids: torch.Tensor, expert_ids: torch.Tensor, mul_weight: bool, top_k: int,
            num_pairs: int, block_m: int, block_n: int = 32, block_k: int = 32) -> None:
    n, k = b.shape[1], b.shape[2]
    grid = (len(expert_ids) * triton.cdiv(n, block_n),)
    fused_moe_kernel[grid](
        a, b, c, topk_weights, sorted_ids, expert_ids, n, k, num_pairs,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), c.stride(0), c.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, MUL_ROUTED_WEIGHT=mul_weight, TOP_K=top_k,
    )


class FusedMoeBackend(BaseMoeBackend):
    def __init__(self, block_m: int = 16) -> None:
        self.block_m = block_m

    def forward(self, hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                gating_output: torch.Tensor, topk: int, renormalize: bool) -> torch.Tensor:
        num_tokens, hidden = hidden_states.shape
        topk_weights, topk_ids = select_experts(gating_output, topk, renormalize)
        num_pairs = topk_ids.numel()
        sorted_ids, expert_ids = moe_align_block_size(topk_ids, self.block_m, w1.shape[0])
        weights = topk_weights.flatten().to(hidden_states.dtype).contiguous()
        cache1 = hidden_states.new_empty(num_pairs, w1.shape[1])  # [T*k, 2I]
        _invoke(hidden_states, w1, cache1, weights, sorted_ids, expert_ids, False, topk,
                num_pairs, self.block_m)
        cache2 = silu_and_mul(cache1)  # [T*k, I]
        cache3 = hidden_states.new_empty(num_pairs, hidden)  # [T*k, H]，已乘路由权重
        _invoke(cache2, w2, cache3, weights, sorted_ids, expert_ids, True, 1,
                num_pairs, self.block_m)
        return cache3.view(num_tokens, topk, hidden).sum(dim=1)
