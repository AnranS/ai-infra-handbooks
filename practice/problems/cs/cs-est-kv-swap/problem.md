---
title: 估算：KV 换回来还是重算
chapter: os/pinned-numa.md
difficulty: 简单
tags: [估算, KV Cache, PCIe, 卸载]
---
一个请求的 KV Cache 被卸载到了 CPU 内存，要继续生成时有两个选择：通过 PCIe 把 KV 换回 GPU，或者丢掉 KV、重新 prefill。实现：

1. `kv_bytes(tokens, layers, kv_heads, head_dim, dtype_bytes)`：这么多 token 的 KV 占多少字节（K 和 V 各一份）；
2. `swap_ms(nbytes, gbps, overhead_ms=0.0)`：以 `gbps` GB/s（1 GB = 10^9 字节）的带宽传输 `nbytes` 字节要多少毫秒，再加上固定开销 `overhead_ms`（发起传输、同步、更新块表）；
3. `recompute_ms(tokens, params, tflops)`：重算 prefill 的时间，按每个 token `2 × params` 次浮点运算、有效算力 `tflops`（10^12 次/秒）估算；
4. `choose(tokens, model, gbps, tflops, overhead_ms=0.0)`：`model` 是字典，含 `layers`、`kv_heads`、`head_dim`、`dtype_bytes`、`params`。返回 `("swap" 或 "recompute", 换回的毫秒数, 重算的毫秒数)`，时间一样时选 `"swap"`。

```python
m70 = {"layers": 80, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "params": 70e9}
kv_bytes(32768, 80, 8, 128, 2)           # 10737418240（10 GiB）
choose(32768, m70, 50, 600)[0]           # "swap"：换回约 215 ms，重算约 7.6 秒
```

<!-- 题解 -->
每个 token 每层存一份 K 和一份 V，各 `kv_heads × head_dim` 个元素，所以是 `tokens × layers × 2 × kv_heads × head_dim × dtype_bytes` 字节。70B 的 GQA 模型每个 token 约 0.31 MiB，32K token 就是 10 GiB。

有意思的是：换回和重算的时间都和 token 数成正比——每个 token 的换回时间是"每 token 的 KV 字节数 ÷ 带宽"，重算时间是"2 × 参数量 ÷ 算力"。所以在没有固定开销的模型里，哪个更快和上下文长度无关，只取决于模型（GQA 越激进、每 token 的 KV 越小，换回越划算）和硬件（PCIe 越快越划算，算力越强越倾向重算）。真实系统里换回有固定开销，短上下文时重算还能和其他请求拼进同一批，所以短的重算、长的换回；再加上 prefill 会占用本可以服务别人的算力，长上下文几乎总是换回划算。
