---
title: 估算：PD 分离的 KV 传输能不能藏进 prefill
chapter: distributed/pd-disagg.md
difficulty: 中等
tags: [估算, PD 分离, KV 传输, RDMA]
---
PD 分离后，prefill 实例算完的 KV Cache 要传给 decode 实例，这段时间会加到首 token 延迟（TTFT）上。
如果**逐层传输**——第 $i$ 层算完就开始传第 $i$ 层的 KV，同时计算第 $i+1$ 层——大部分传输可以和计算重叠。

把 prefill 平均分到 $L$ 层，每层计算 $c = P/L$、每层传输 $t = T/L$（$P$、$T$ 分别是 prefill 和传输的总时间）。逐层流水时，最后一层的 KV 传完的时刻是：

- $t \le c$（传得比算得快）：$L \cdot c + t$；
- $t > c$（传得慢）：$c + L \cdot t$。

实现（链路速率单位 Gbps = $10^9$ 比特/秒；时间单位毫秒）：

1. `kv_transfer_ms(prompt_len, kv_bytes_per_token, link_gbps, n_links=1, efficiency=0.8)`：整条 prompt 的 KV 一次性传完的时间，
   有效带宽 = `link_gbps` / 8 × `n_links` × `efficiency`（GB/s）；
2. `exposed_ms(prefill_ms, transfer_ms, n_layers)`：逐层流水时，prefill 算完之后还要等多久 KV 才传完；
3. `ttft_ms(prefill_ms, transfer_ms, n_layers, layerwise=True)`：prefill 时间加上暴露出来的传输时间（`layerwise=False` 时整个传输都暴露）。

```python
kv_transfer_ms(8192, 327680, 400, n_links=8)          # Llama-3-70B 的 8K prompt：约 8.4 ms（8 张 400G 网卡）
exposed_ms(314.7, 8.39, 80)                           # 约 0.1 ms：几乎完全藏住
exposed_ms(314.7, 1074, 80)                           # 约 763 ms：链路太慢，藏不住
```

<!-- 题解 -->
- 70B 模型 8K prompt 的 KV 约 2.7 GB，prefill 却要 300 ms 左右，只要有几张 RDMA 网卡，传输完全能藏在计算后面；
- 链路慢到"每层传输时间 > 每层计算时间"时，逐层流水也救不了，暴露时间约等于 $T - P$；
- MLA 模型每 token 的 KV 只有 GQA 模型的 1/4～1/5，对网络的要求低得多，这也是 MLA 模型做 PD 分离特别合适的原因之一；
- 实际系统还要考虑 KV 的页布局（不连续的页要打包或用 scatter-gather）、prefill 与 decode 的 TP 规模不同时的重排，以及传输完成的通知机制。
