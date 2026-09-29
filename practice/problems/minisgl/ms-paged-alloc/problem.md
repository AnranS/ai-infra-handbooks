---
title: 按页分配 KV 并写进 page table
chapter: compute/kvcache.md
difficulty: 中等
tags: [page table, 分页, 批量分配]
---
mini-sglang 的 page table 形状是 `(max_running_reqs, max_seq_len)`，**按 token 存放**：`page_table[row, j]` 是第 `row` 行请求第 `j` 个 token 的 KV 在池中的位置。
KV 池按页管理（`page_size` 个 token 一页），空闲列表 `free_slots` 存每个空闲页**第一个 token 的位置**：`page_size = 4` 时初始是 `[0, 4, 8, ...]`。

实现 `PageAllocator(num_pages, page_size, max_rows, max_len)`（`page_table` 初始全是 -1，`free_slots` 是 Python 列表）：

1. `allocate_paged(reqs)`：`reqs` 是 `[(row, cached_len, device_len), ...]`，给每个请求本轮要算的 token 分配页并写进 page table：
   - 请求已经占有的页不重复分配：需要的页是 `[ceil(cached_len / ps), ceil(device_len / ps))`；
   - 所有请求需要的页**一次性**从 `free_slots` 头部取出（按请求顺序依次使用），不够时抛出 `MemoryError` 且不改变状态；
   - 把页展开成 token 位置（页 `X` → `X, X+1, ..., X+ps-1`），写到 `page_table[row, first_page*ps : last_page*ps]`；
2. `out_loc(reqs)`：本轮新 token 的 KV 写入位置：把每个请求的 `page_table[row, cached_len:device_len]` 按顺序拼起来（numpy 数组）；
3. `free(indices)`：释放一段 token 位置（它们总是按页对齐的完整页），把 `indices[::ps]` 追加到 `free_slots` 尾部；
4. `lazy_free()`：上下文管理器，在里面调用的 `free` 先攒起来，退出时按调用顺序一次性追加（书中用来减少 `torch.cat` 的次数）。

<!-- 题解 -->
```python
info, need = [], 0
for row, c, d in reqs:
    f, l = ceil(c / ps), ceil(d / ps)
    if l > f: info.append((row, f, l)); need += l - f
pages = free_slots[:need]; del free_slots[:need]
tokens = (np.array(pages)[:, None] + np.arange(ps)).ravel()
```

然后按 `info` 顺序切片写进 page table。`cached_len` 不是页的整数倍时（比如上一轮 prefill 结束在页中间），这一页已经分配过，所以用 `ceil`。
书中把所有 (行, 列) 下标先在 CPU 上算好、再一次索引赋值写到 GPU 上的 page table，避免很多次小拷贝。
