---
title: 估算：张量并行的通信量与耗时
chapter: distributed/tensor-parallel.md
difficulty: 中等
tags: [估算, 张量并行, all-reduce, 通信]
---
Megatron 式的张量并行里，每个 Transformer 层前向有两次 all-reduce（注意力输出、MLP 输出），每次的数据量是 `tokens × hidden × dtype_bytes`。
用 ring all-reduce 时，每张卡要发送（也要接收）$2(n-1)/n$ 倍的数据；另外每一步（共 $2(n-1)$ 步）都有一个固定延迟。

实现（`link_gbs` 是每张卡单方向的链路带宽，GB/s = $10^9$ 字节/秒；时间单位毫秒）：

1. `allreduce_ms(nbytes, n, link_gbs, latency_us=0.0)`：$\frac{2(n-1)}{n} \cdot \frac{\text{nbytes}}{\text{link}} + 2(n-1) \cdot \text{latency}$；`n == 1` 时为 0；
2. `tp_comm_ms(tokens, hidden, n_layers, tp, link_gbs, dtype_bytes=2, latency_us=0.0)`：一次前向里所有层的 all-reduce 总耗时；
3. `comm_share(comm_ms, compute_ms)`：通信不与计算重叠时，通信占总时间的比例。

```python
# Llama-3-70B（hidden 8192、80 层），TP=8，NVLink 单方向 450 GB/s
tp_comm_ms(8192, 8192, 80, 8, 450)                    # prefill 8K token：约 83 ms
tp_comm_ms(64, 8192, 80, 8, 450, latency_us=2)       # decode batch 64：带宽项只有 0.65 ms，延迟项 4.5 ms
```

<!-- 题解 -->
- prefill 时数据量大，通信时间由带宽决定：同样的 8K prefill 换成 PCIe（单方向约 25 GB/s）就是 1.5 秒，张量并行基本不可用——这就是 TP 只在 NVLink 域内做的原因；
- decode 时每次 all-reduce 只有 1 MB，带宽项很小，时间几乎全是延迟：$2(n-1)$ 步 × 每步几微秒 × 160 次。
  所以推理框架会为小消息专门写 one-shot / two-shot all-reduce（一步把数据直接写到所有卡），用 CUDA Graph 消掉启动开销，或者把 all-reduce 和下一层的计算融合。
