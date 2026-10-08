"""paged.py —— 分页 KV Cache：块池、按块存放的 K/V 张量、slot mapping，以及分页注意力的参考实现。

布局与 vLLM 的 FlashAttention 后端一致：每层一个 [num_blocks, block_size, num_kv_heads, head_dim] 的 K 和 V。
"""

import math
from collections import deque

import torch


class BlockPool:
    """管理固定数量的物理块：空闲队列 + 引用计数。"""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.ref_cnt = [0] * num_blocks
        self.free_queue = deque(range(num_blocks))

    def num_free(self) -> int:
        return len(self.free_queue)

    def allocate(self, n: int) -> list[int] | None:
        """取 n 个空闲块；不够就返回 None（由调度器决定是否抢占）。"""
        if n > len(self.free_queue):
            return None
        blocks = [self.free_queue.popleft() for _ in range(n)]
        for b in blocks:
            self.ref_cnt[b] = 1
        return blocks

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            self.ref_cnt[b] -= 1
            if self.ref_cnt[b] == 0:
                self.free_queue.append(b)


class PagedKVCache:
    """所有层的 K/V 存储。一个 token 的"地址"是 slot = 块号 × block_size + 块内偏移。"""

    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float32):
        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.v = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.block_size = block_size

    def write(self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: [N, num_kv_heads, head_dim]，第 i 个 token 写到 slot_mapping[i]。"""
        self.k[layer].view(-1, *k.shape[1:])[slot_mapping] = k
        self.v[layer].view(-1, *v.shape[1:])[slot_mapping] = v

    def gather(self, layer: int, block_table: list[int], seq_len: int):
        """按块表取出一个序列的前 seq_len 个 token 的 K、V：[seq_len, num_kv_heads, head_dim]。"""
        blocks = torch.tensor(block_table[: math.ceil(seq_len / self.block_size)])
        k = self.k[layer][blocks].flatten(0, 1)[:seq_len]
        v = self.v[layer][blocks].flatten(0, 1)[:seq_len]
        return k, v


def slot_mapping_for(block_table: list[int], positions, block_size: int) -> list[int]:
    """位置 p 的 token 存在第 p // block_size 个逻辑块、块内偏移 p % block_size。"""
    return [block_table[p // block_size] * block_size + p % block_size for p in positions]


def paged_attention(q, kv: PagedKVCache, layer: int, block_tables, seq_lens, query_start_loc, scale: float):
    """变长、分页的因果注意力（参考实现，逐个请求计算）。

    q: [N, num_heads, head_dim]，N 个新 token 来自 B 个请求，第 i 个请求的新 token 是
    q[query_start_loc[i]:query_start_loc[i+1]]，它们是该请求上下文的最后几个 token（上下文总长 seq_lens[i]）。
    """
    out = torch.empty_like(q)
    num_heads = q.shape[1]
    for i, (table, seq_len) in enumerate(zip(block_tables, seq_lens)):
        start, end = query_start_loc[i], query_start_loc[i + 1]
        num_new = end - start
        k, v = kv.gather(layer, table, seq_len)                        # [S, n_kv, hd]
        rep = num_heads // k.shape[1]                                  # GQA：每个 KV 头服务 rep 个 query 头
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,shd->hts", q[start:end], k) * scale  # [H, T, S]
        # 第 j 个新 token 的位置是 seq_len - num_new + j，只能看到不超过它的位置
        mask = torch.ones(num_new, seq_len, dtype=torch.bool).tril(diagonal=seq_len - num_new)
        probs = scores.masked_fill(~mask, float("-inf")).float().softmax(-1).to(q.dtype)
        out[start:end] = torch.einsum("hts,shd->thd", probs, v)
    return out
