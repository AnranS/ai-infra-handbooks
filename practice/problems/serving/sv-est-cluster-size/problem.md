---
title: 估算：部署规模——从 QPS 到 GPU 数
chapter: career/system-design.md
difficulty: 困难
tags: [估算, 系统设计, 容量规划, Little 定律]
---
系统设计题的第一步几乎总是："每秒 N 个请求、输入 I、输出 O、TPOT 不超过 X ms，要多少张卡？"这里估算 **decode 实例池**的规模（假设 PD 分离，prefill 另算）。
一个实例由 `gpus_per_instance` 张卡组成（张量并行），约定 1 GB = $10^9$ 字节：

- decode 期间一条请求的平均上下文长度取 `input_len + output_len / 2`；
- KV 预算 = 每卡 `gpu_mem_gb × mem_util - reserve_gb` 乘卡数，再减去权重；并发数不能超过 KV 预算能放下的请求数；
- decode 一步的时间按带宽瓶颈算：(权重 + batch × 平均上下文 × 每 token KV 字节) ÷ 实例总带宽，再加 `overhead_ms`；并发数还要保证这一步不超过 `tpot_ms`；
- 每条请求要占用一个并发位 `output_len` 步，由 Little 定律，实例每秒能完成的请求数 = batch ÷（`output_len` × 单步时间）；
- 为了应对峰值，每个实例只按 `headroom` 的比例使用。

实现：

1. `max_batch_by_kv(kv_budget_bytes, avg_ctx, kv_bytes_per_token)`：KV 预算能放下的并发数（向下取整）；
2. `step_ms(batch, weight_bytes, kv_bytes_per_token, avg_ctx, bw_gbs, overhead_ms=0.0)`：`bw_gbs` 是实例总带宽；
3. `plan(qps, input_len, output_len, tpot_ms, weight_bytes, kv_bytes_per_token, gpus_per_instance, gpu_mem_gb, gpu_bw_gbs, mem_util=0.9, reserve_gb=4.0, overhead_ms=0.0, headroom=0.7)`：
   返回字典 `batch`（取 KV 上限和 TPOT 上限中较小的）、`step_ms`、`rps_per_instance`、`instances`（向上取整）、`gpus`；
   如果连 batch=1 都满足不了 TPOT 或放不下 KV，抛出 `ValueError`。

```python
# Llama-3-70B（BF16 权重 141 GB、每 token KV 327680 字节），8 × H100 一个实例，输入 2000、输出 500、TPOT ≤ 50 ms
plan(200, 2000, 500, 50, 141e9, 327680, 8, 80, 3350)
# batch 546（受 KV 限制）、单步约 20 ms、每实例约 54 req/s → 6 个实例、48 张卡
```

<!-- 题解 -->
这个例子里并发数被 **KV 容量**卡住（546），而不是被 TPOT 卡住（TPOT 允许 1600 多）。这时候最有效的手段是省 KV：FP8 KV Cache 让并发翻倍、单实例吞吐接近翻倍；
而把权重量化到 INT4 只省下约 100 GB，效果反而不如前者。反过来，如果 TPOT 要求很严（比如 10 ms），并发就被带宽卡住，这时要加卡（更大的 TP）或者换带宽更高的卡。

面试时把这几步讲清楚，再补上被忽略的部分：prefill 的算力（按 $2NI$ 估算 prefill 池的规模）、decode 的算力上限（batch 很大时 $2N \times \text{batch}$ 的 FLOPs 也要检查）、
请求长度的分布（按 P99 而不是平均值留余量）、以及故障冗余（N+1）。
