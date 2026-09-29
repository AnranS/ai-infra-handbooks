---
title: 两级 KV 缓存：GPU 显存 + CPU 内存
chapter: distributed/kv-offload.md
difficulty: 中等
tags: [KV 卸载, LRU, 缓存层级]
---
GPU 显存放不下所有前缀缓存时，可以把被淘汰的块卸载到 CPU 内存（LMCache、SGLang HiCache 的思路）。实现两级 LRU 缓存 `TieredKVCache(gpu_blocks, cpu_blocks)`，缓存的对象是块哈希：

- `access(h)`：请求要用哈希为 `h` 的块，返回 `"gpu"`、`"cpu"` 或 `"miss"`：
  - `"gpu"`：在 GPU 上，刷新它在 GPU LRU 里的位置；
  - `"cpu"`：在 CPU 上，需要**加载回 GPU**：从 CPU 移除，放进 GPU（如果 GPU 满了，先把 GPU 上最久没用的块**卸载到 CPU**；如果 CPU 也满了，CPU 上最久没用的块被丢弃）；
  - `"miss"`：两级都没有，需要重新计算，算完放进 GPU（同样可能触发卸载和丢弃）；
- 统计：`stats` 字典，包含 `gpu_hits`、`cpu_hits`、`misses`、`offloads`（GPU → CPU 的次数）、`loads`（CPU → GPU 的次数）、`drops`（从 CPU 丢弃的次数）；
- `gpu_contents()`、`cpu_contents()`：返回两级里的哈希，按 LRU 顺序（最久没用的在前）。

`cpu_blocks` 可以为 0（没有 CPU 缓存，GPU 淘汰的块直接丢弃，此时算 `drops`，不算 `offloads`）。

<!-- 题解 -->
两个 `OrderedDict`，都按"最久没用的在前"。GPU 腾位置的函数：

```python
def _evict_gpu(self):
    h, _ = self.gpu.popitem(last=False)
    if self.cpu_cap == 0: self.stats["drops"] += 1; return
    if len(self.cpu) >= self.cpu_cap: self.cpu.popitem(last=False); self.stats["drops"] += 1
    self.cpu[h] = None; self.stats["offloads"] += 1
```

从 CPU 加载回来的块在 GPU 上是"最新"的。多轮对话场景里，用户思考的几十秒里他的 KV 被挤到 CPU，下一轮直接从 CPU 读回来（PCIe 带宽几十 GB/s），比重算便宜得多。
