---
title: 屋顶线模型：算力瓶颈还是带宽瓶颈
chapter: performance-math.md
difficulty: 简单
tags: [roofline, 算术强度, 估算]
---
一次矩阵乘 $C = A B$，$A$ 是 $M \times K$、$B$ 是 $K \times N$，元素大小 `bytes_per_elem` 字节。假设每个矩阵只从显存读一次、结果写一次：

- 计算量：$2MNK$ FLOPs；
- 访存量：$(MK + KN + MN) \times$ `bytes_per_elem`；
- 算术强度：计算量 / 访存量（FLOPs/Byte）。

实现：

1. `gemm_intensity(M, N, K, bytes_per_elem=2)`：返回算术强度；
2. `roofline_time(flops, bytes_, peak_tflops, bandwidth_gbs)`：屋顶线模型下的时间下限（秒），$\max(\text{flops}/\text{peak}, \text{bytes}/\text{带宽})$，
   注意单位：`peak_tflops` 是 $10^{12}$ FLOPs/s，`bandwidth_gbs` 是 $10^9$ Bytes/s；
3. `bound(M, N, K, peak_tflops, bandwidth_gbs, bytes_per_elem=2)`：返回 `"compute"` 或 `"memory"`，表示这个 GEMM 受限于算力还是带宽（算术强度等于"屋脊点" $\text{peak}/\text{带宽}$ 时算 `"compute"`）；
4. `decode_step_ms(num_params, batch, peak_tflops, bandwidth_gbs, bytes_per_param=2)`：估算大模型 decode 一步的时间下限（毫秒）：
   每步要把全部权重读一遍（$\text{num\_params} \times \text{bytes\_per\_param}$ 字节），计算量约 $2 \times \text{num\_params} \times \text{batch}$；忽略 KV cache 和激活。

```python
bound(4096, 4096, 4096, 989, 3350)    # "compute"：大方阵
bound(1, 4096, 4096, 989, 3350)       # "memory"：batch=1 的 decode 就是 GEMV
decode_step_ms(7e9, 1, 989, 3350)     # 约 4.18 ms：读 14 GB 权重
```

<!-- 题解 -->
屋脊点 $I^* = \text{peak}/\text{BW}$（H100 上约 $989 \times 10^{12} / 3.35 \times 10^{12} \approx 295$ FLOPs/Byte）。算术强度低于它就是带宽瓶颈。

GEMV（$M=1$）的算术强度约为 $2NK / (2NK) = 1$，远低于屋脊点，所以 decode 阶段的时间几乎就是"把权重读一遍"的时间；
batch 增大时权重读一次可以服务多个请求，算术强度随 batch 线性增长，这就是 decode 要攒批的根本原因。
