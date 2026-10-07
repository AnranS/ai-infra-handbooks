# Paged KV cache

<p class="lead">The first engineering problem an inference engine must solve: the KV caches of hundreds or thousands of requests have different lengths, grow at any time and are freed at any time, so how do you fit them into a fixed amount of GPU memory? vLLM's answer is PagedAttention: like an operating system managing memory, cut the KV cache into fixed-size blocks and allocate them on demand. This chapter implements a block pool, K/V tensors stored by block, slot mapping, and a reference implementation of paged attention; the mini engine of the following chapters is built on it.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. If each request reserves KV cache for its maximum length, roughly what is memory utilization? And with paging?
    2. What are the block table and the slot mapping? How is it computed where a token's K/V are written?
    3. With paging, what extra inputs does the attention kernel need?
    4. What are the pros and cons of block sizes 1, 16 and 256?

??? success "Answers (try first, then expand to compare)"
    1. It depends on the distribution of request lengths; in this chapter's example, reserving for the maximum length gives only about 46% utilization; with paging the only waste is the unused part of each request's last block, and with a block size of 16 utilization is about 99.6%.
    2. The block table maps a request's logical block numbers to physical block numbers; the slot mapping gives, for each new token in this step, where in the KV pool its K / V are written: slot = physical block number × block size + offset within the block (look up the block table with position ÷ block size; position % block size is the offset).
    3. Each request's block table (logical block → physical block), each request's context length, and `cu_seqlens` to tell apart each request's queries (plus the slot mapping for writing KV).
    4. Block size 1: no internal fragmentation and the finest prefix caching granularity, but the longest block tables and the largest overhead in metadata and indirect addressing in kernels; 16: the common compromise (vLLM's default); 256: short block tables and contiguous memory access in kernels, but lots of internal fragmentation, prefixes that only hit in large blocks, and coarse granularity for transfers and swapping.

<!-- comic ../assets/comics/paged-kv.webp is in Chinese; put it back once the English version exists -->

## Why paging {#为什么要分页}

The most naive approach reserves a contiguous region of memory for each request, as large as the maximum context length. There are three problems:

- **Reservation waste**: how long a request will actually be is unknown in advance, so the maximum must be reserved;
- **Internal fragmentation**: even with the final length known, the space at the end sits empty throughout generation;
- **External fragmentation**: as requests come and go, free memory is cut into small pieces of different sizes that cannot hold a new long request.

Estimate it with a simple workload:

```python
import math
import random

random.seed(0)
max_len = 4096
reqs = [(random.randint(100, 2000), random.randint(20, 1500)) for _ in range(200)]   # (prompt length, output length)
used = sum(p + o for p, o in reqs)
print(f"按最大长度预留：利用率 {used / (len(reqs) * max_len):.1%}")
for block_size in (1, 16, 64, 256):
    allocated = sum(math.ceil((p + o) / block_size) * block_size for p, o in reqs)
    print(f"分页，块大小 {block_size:3d}：利用率 {used / allocated:.1%}")
```

```text
按最大长度预留：利用率 46.2%
分页，块大小   1：利用率 100.0%
分页，块大小  16：利用率 99.6%
分页，块大小  64：利用率 98.3%
分页，块大小 256：利用率 93.2%
```

Drag the block size and the number of requests to see which part is actually wasted:

<div class="aig-widget" data-widget="pagedkv"></div>

Reserving for the maximum length wastes more than half the memory; with paging, the only waste is the unfilled part of each request's last block, half a block on average. The vLLM paper reports that existing systems wasted 60–80% of their KV cache memory, while PagedAttention wastes less than 4%. The memory saved holds more concurrent requests, and [throughput grows almost linearly with batch size](llm://inference/kv-cache/#批处理让多个请求分摊权重读取), so paging translates directly into throughput.

## Implementation {#实现}

The whole paged KV cache has three parts:

1. **The block pool**: manages a fixed number of physical blocks, recording which are free and how many requests reference each block;
2. **KV storage**: one pair of K and V tensors per layer, of shape `[num_blocks, block_size, num_kv_heads, head_dim]`;
3. **Block tables**: one table per request, whose entry i is the physical block holding its logical block i.

```python title="paged.py"
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
        rep = num_heads // k.shape[1]                                  # GQA: each KV head serves rep query heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,shd->hts", q[start:end], k) * scale  # [H, T, S]
        # new token j sits at position seq_len - num_new + j and can only see positions up to it
        mask = torch.ones(num_new, seq_len, dtype=torch.bool).tril(diagonal=seq_len - num_new)
        probs = scores.masked_fill(~mask, float("-inf")).float().softmax(-1).to(q.dtype)
        out[start:end] = torch.einsum("hts,shd->thd", probs, v)
    return out
```

### Block tables and slot mapping {#块表与-slot-mapping}

A token at position p in the sequence belongs to logical block `p // block_size`, at offset `p % block_size` within the block; looking up the block table gives physical block b, and its K/V are stored at **slot** `b × block_size + p % block_size`. Before each forward pass, the engine computes a slot mapping array for all the new tokens of the step, and the attention layers write the new K/V into the cache by it:

![Figure: the block table maps each request's logical blocks to physical blocks](../assets/figures/paged-kv.svg){.aig-svg}

```pycon
>>> from paged import BlockPool, slot_mapping_for
>>> pool = BlockPool(num_blocks=8)
>>> table = pool.allocate(3)            # a request needs 3 blocks
>>> table, pool.num_free()
([0, 1, 2], 5)
>>> other = pool.allocate(2)            # another request
>>> pool.free(table)                    # the first request finishes; its blocks go to the end of the free queue
>>> pool.allocate(4)                    # the new request's physical blocks are not contiguous
[5, 6, 7, 0]
>>> slot_mapping_for([5, 6, 7, 0], positions=range(14), block_size=4)
[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1]
```

14 logically contiguous tokens are physically scattered across 4 non-adjacent blocks. That is exactly the point of paging: physical blocks can be allocated anywhere, with no need for large contiguous regions of memory.

### Paged attention {#分页注意力}

When computing attention, each request must read the K/V of its **whole context**, which are scattered across several physical blocks. So a paged attention kernel takes three more kinds of input than ordinary attention:

| Input | Meaning |
| --- | --- |
| `block_tables` | each request's block table |
| `seq_lens` | each request's total context length (after this step's computation) |
| `query_start_loc` | the new tokens of all requests laid end to end in one dimension; request i's queries are `q[qsl[i]:qsl[i+1]]` (called `cu_seqlens_q` in FlashAttention) |

The `paged_attention` above is a per-request reference implementation: it gathers the K/V into contiguous tensors by the block table, then runs ordinary causal attention. A real kernel does not gather first; it addresses through the block table inside the kernel, computing as it reads (see [paged decode attention](cuda://advanced/attention/#一个分页-decode-attention-kernel支持-gqa) in the CUDA book), but the math is exactly the same.

Check it: three requests at different stages (decoding 1 token, a full prefill, the second half of a chunked prefill), with physical blocks allocated interleaved; write their K/V into the paged cache, then compare with "contiguous storage + standard attention":

```python
import torch
import torch.nn.functional as F
from paged import BlockPool, PagedKVCache, paged_attention, slot_mapping_for

torch.manual_seed(0)
num_heads, num_kv_heads, head_dim, block_size = 8, 2, 64, 4
seq_lens, query_lens = [37, 5, 20], [1, 5, 8]          # decode, a full prefill, the last 8 tokens of a chunked prefill
pool = BlockPool(num_blocks=32)
kv = PagedKVCache(num_layers=1, num_blocks=32, block_size=block_size, num_kv_heads=num_kv_heads, head_dim=head_dim)

# give the three requests one block at a time in turn, so their physical blocks interleave
tables = [[] for _ in seq_lens]
while any(len(t) * block_size < s for t, s in zip(tables, seq_lens)):
    for t, s in zip(tables, seq_lens):
        if len(t) * block_size < s:
            t += pool.allocate(1)
print("块表：", tables)

ks = [torch.randn(s, num_kv_heads, head_dim) for s in seq_lens]
vs = [torch.randn(s, num_kv_heads, head_dim) for s in seq_lens]
for t, k, v in zip(tables, ks, vs):
    kv.write(0, torch.tensor(slot_mapping_for(t, range(len(k)), block_size)), k, v)

qs = [torch.randn(n, num_heads, head_dim) for n in query_lens]
qsl = [0]
for n in query_lens:
    qsl.append(qsl[-1] + n)
out = paged_attention(torch.cat(qs), kv, 0, tables, seq_lens, qsl, scale=head_dim ** -0.5)

for i, (q, k, v, s, n) in enumerate(zip(qs, ks, vs, seq_lens, query_lens)):
    rep = num_heads // num_kv_heads
    mask = torch.ones(n, s, dtype=torch.bool).tril(diagonal=s - n)   # the new tokens are at the end of the sequence
    ref = F.scaled_dot_product_attention(q.transpose(0, 1), k.repeat_interleave(rep, 1).transpose(0, 1),
                                         v.repeat_interleave(rep, 1).transpose(0, 1), attn_mask=mask)
    err = (out[qsl[i]:qsl[i + 1]] - ref.transpose(0, 1)).abs().max().item()
    print(f"请求 {i}：上下文 {s}，新 token {n}，与连续存放的最大误差 {err:.1e}")
    assert err < 1e-5
```

```text
块表： [[0, 3, 6, 8, 10, 12, 13, 14, 15, 16], [1, 4], [2, 5, 7, 9, 11]]
请求 0：上下文 37，新 token 1，与连续存放的最大误差 2.1e-07
请求 1：上下文 5，新 token 5，与连续存放的最大误差 2.4e-07
请求 2：上下文 20，新 token 8，与连续存放的最大误差 4.2e-07
```

## Choosing the block size {#块大小怎么选}

| Block size | Pros | Cons |
| --- | --- | --- |
| Small (1–8) | almost no internal fragmentation; prefix caching down to the token | very long block tables, lots of metadata; kernels can read only a little contiguous data at a time, inefficient memory access |
| Medium (16–64) | balances both; vLLM defaults to 16 | half a block wasted per request on average |
| Large (128 and up) | contiguous reads, efficient kernels; short block tables | lots of fragmentation; coarse prefix caching granularity, short prefixes cannot be shared |

The actual block size is also constrained by kernels: FlashAttention's paged KV requires a multiple of 16, FlashMLA uses 64, and FlashInfer supports any page size (including 1). SGLang historically defaulted to a page size of 1 (managing by token), which, together with its radix tree, gives token-granular prefix sharing.

!!! source "Source code"
    - **vLLM**: the block pool is `BlockPool` in `vllm/v1/core/block_pool.py`. Free blocks live in a doubly linked list, `FreeKVCacheBlockQueue` (`kv_cache_utils.py`), so blocks hit by the prefix cache can be removed from the middle in O(1). Block 0 is reserved as `null_block` for padding. Each request's blocks are allocated by `KVCacheManager.allocate_slots`. On the worker side, `vllm/v1/worker/block_table.py` maintains the block tables and computes the slot mapping with a Triton kernel (`compute_slot_mapping`).
    - In vLLM 0.30, the FlashAttention backend's KV tensor has shape `[num_blocks, num_kv_heads, block_size, 2 * head_size]` (K and V concatenated in the last dimension). The logical layout of all backends is described uniformly as `[L, B, H, N, C]`, with the physical arrangement decided by `KVCacheLayout` (`vllm/v1/kv_cache_layout.py`). Different attention backends require different layouts, which is also why transferring KV under PD disaggregation needs layout conversion.
    - **SGLang**: a two-level mapping. `ReqToTokenPool` (`mem_cache/memory_pool.py`) records the KV slot of each request's token i, the `TokenToKVPoolAllocator` family (`mem_cache/allocator/`) allocates slots, and classes such as `MHATokenToKVPool` hold the actual K/V tensors.

!!! interview "In an interview"
    On paged KV: reserving KV for each request's maximum length leaves most memory occupied by space that "might be used", so utilization is low; paging allocates by block (16 tokens by default in vLLM) on demand, leaving only the unused part of the last block as waste. The block pool manages physical blocks and reference counts (shared prefixes, copy-on-write), the block table maps a request's logical blocks to physical blocks, and a token is written to slot = block number × block size + offset in block. The attention kernel takes three more kinds of input: block tables, each request's context length, and `cu_seqlens`, and the math is exactly the same as contiguous storage. Block size trades off fragmentation, kernel efficiency and prefix caching granularity: larger blocks mean less metadata and more contiguous access but more tail waste and coarser prefix reuse; SGLang can use 1.

## Exercises {#练习}

**1. Estimating concurrency.** Qwen2.5-7B (28 layers, 4 KV heads, head_dim 128, BF16) is deployed on an 80 GB GPU, with about 15 GB of weights and 10 GB reserved for activations and other overhead. With an average context of 3000 tokens, how many concurrent requests does paging (block size 16) fit? What if each request reserves the maximum context of 32K?

??? success "Answer"
    ```python
    kv_per_token = 2 * 28 * 4 * 128 * 2                     # 57344 bytes = 56 KB
    budget = (80 - 15 - 10) * 1024**3
    paged = budget // (math.ceil(3000 / 16) * 16 * kv_per_token)
    prealloc = budget // (32768 * kv_per_token)
    print(f"每 token {kv_per_token // 1024} KB；分页约 {paged} 个并发，按 32K 预留只有 {prealloc} 个")
    ```

    ```text
    每 token 56 KB；分页约 342 个并发，按 32K 预留只有 31 个
    ```

    Paging raises concurrency about 10-fold. It also shows why structures that reduce KV, such as GQA, matter so much for serving.

**2. Copy-on-write.** With parallel sampling (`n > 1`), the same prompt generates several answers. How can the block pool's reference counts let them share the prompt's KV cache? What should happen when one answer needs to write a new token into a shared block?

??? success "Answer"
    The prompt's blocks are computed once, every answer's block table points to these physical blocks, and the reference count equals the number of answers. During generation, if the block to be written has a reference count greater than 1 (it is shared), first allocate a new block, copy the old block's content into it, then write into the new block and decrement the old block's reference count. This is copy-on-write. Full prompt blocks are never written again, so only the last, partially filled block may need copying. vLLM V1 does something simpler: it shares only full blocks, and each answer starts writing in its own block.

## Summary {#小结}

- [x] Reserving KV cache for the maximum length gives very low memory utilization; with paging, the only waste is the unused part of the last block.
- [x] The block pool manages physical blocks and reference counts; the block table maps logical blocks to physical blocks; slot = block number × block size + offset in block.
- [x] Paged attention adds three kinds of input (block tables, context lengths and `cu_seqlens`), and the math is exactly the same as contiguous storage.
- [x] Block size trades off fragmentation, kernel efficiency and prefix caching granularity; vLLM defaults to 16, and SGLang can go down to 1.
