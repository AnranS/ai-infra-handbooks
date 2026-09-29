---
title: cache_req：把请求的 KV 交给前缀缓存
chapter: schedule/cache-manager.md
difficulty: 中等
tags: [前缀缓存, 所有权, 资源释放]
---
请求 prefill 结束（或整个请求结束）时，`CacheManager.cache_req` 把它算好的 KV 插进前缀缓存。这一步要仔细处理 KV 页的**所有权**，书中把请求的 KV 分成四段：

```text
[0, old_cached)            prefill 之前就在缓存里（本请求锁着旧句柄）
[old_cached, prefix_len)   本请求自己算的，但期间别的请求已经把同样的前缀插进了缓存：缓存里已有一份，这份重复了，释放
[prefix_len, new_cached)   新插入缓存的部分：所有权转交给缓存
[new_cached, cached_len)   不足一页、插不进缓存的尾巴：请求结束就释放，否则留给请求继续用
```

其中 `old_cached` 是请求当前锁着的句柄的长度，`prefix_len` 是插入前缓存里已有的长度，`new_cached` 是插入后新句柄的长度（按页对齐）。

模板里给出了一个简化的前缀缓存 `ToyPrefixCache`（`insert_prefix(ids, indices) -> (prefix_len, handle)`、`lock(handle)`、`unlock(handle)`），实现 `cache_req(cache, req, finished, free)`：

- `req` 有 `input_ids`、`cached_len`、`page_indices`（前 `cached_len` 个 token 的 KV 位置，列表）、`handle`（当前锁着的句柄，`handle.cached_len` 就是 `old_cached`）；
- 插入 `input_ids[:cached_len]` 和对应的 `page_indices`；
- **先解锁旧句柄**（插入完成之后），再释放重复的那段；`finished=True` 时释放尾巴，否则把 `req.handle` 换成新句柄并加锁；
- `free(indices)` 是释放函数（测试会记录调用）；空区间不要调用它。

<!-- 题解 -->
```python
ids, idx = req.input_ids[:req.cached_len], req.page_indices[:req.cached_len]
old = req.handle
prefix_len, new = cache.insert_prefix(ids, idx)
cache.unlock(old)
dup = idx[old.cached_len:prefix_len]
if dup: free(dup)
if finished:
    tail = idx[new.cached_len:]
    if tail: free(tail)
else:
    req.handle = new; cache.lock(new)
```

为什么"先插入、后解锁"：如果先解锁，旧句柄那段可能在插入之前被别的请求触发的淘汰删掉。
书中第 11 章还发现了一个 bug：prefill 在途时收到 abort，请求会被 `cache_req` 两次，第二次把缓存里的页当成"重复"释放掉，导致同一页既在空闲列表又在缓存里。
