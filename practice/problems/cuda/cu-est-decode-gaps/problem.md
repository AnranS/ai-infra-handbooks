---
title: 估算：decode 一步里 kernel 边界的空隙占多少
chapter: tools/pdl-megakernel.md
difficulty: 简单
tags: [估算, CUDA Graph, PDL, decode]
---
decode 的每个 kernel 都受访存限制，时间约等于读写的字节数除以有效带宽；用了 CUDA Graph 之后，相邻 kernel 之间在 GPU 上仍有一段空隙 g。实现（有效带宽默认 `BW = 3.35e12 * 0.85` 字节/秒，时间单位都是微秒）：

1. `decode_kernels(hidden, inter, layers, heads, kv_heads, head_dim, vocab, batch=1, context=1024, nbytes=2)`：Qwen3 这类模型 decode 一步的 kernel 列表 `[(名字, 读写字节数), ...]`。每层 10 个 kernel，依次是：
    - `add_rmsnorm`：`4 × act`（`act = batch × hidden × nbytes`）；`qkv_proj`：`hidden × (q + 2kv) × nbytes`（`q = heads × head_dim`，`kv = kv_heads × head_dim`）；
    - `qk_norm_rope`：`2 × batch × (q + kv) × nbytes`；`kv_cache_write`：`2 × batch × kv × nbytes`；`attention`：`batch × context × 2kv × nbytes`；
    - `o_proj`：`q × hidden × nbytes`；`add_rmsnorm`：`4 × act`；`gate_up_proj`：`hidden × 2inter × nbytes`；`silu_mul`：`3 × batch × inter × nbytes`；`down_proj`：`inter × hidden × nbytes`；

    所有层之后再加 3 个：`final_norm`（`2 × act`）、`lm_head`（`hidden × vocab × nbytes`）、`sample`（`batch × vocab × 4`）；
2. `step_time_us(kernels, gap_us, bw=BW)`：返回 `(读写数据的时间, 一步的总时间, 空隙占总时间的比例)`，总时间 = 读写数据的时间 + kernel 数 × 空隙；
3. `eager_time_us(kernels, cpu_launch_us=5.0, gap_us=0.5, bw=BW)`：不用 CUDA Graph 时，CPU 逐个发射 kernel，一步至少要 `kernel 数 × cpu_launch_us`；返回它和 GPU 这一侧的时间（读写数据 + kernel 数 × `gap_us`）中较大的那个；
4. `fused_step_us(kernels, layers, per_layer_before, per_layer_after, gap_us, bw=BW)`：把每层的 kernel 数从 `per_layer_before` 融合到 `per_layer_after` 后一步的总时间（读写数据的时间不变，只少了空隙）。

```python
k = decode_kernels(1024, 3072, 28, 16, 8, 128, 151936)     # Qwen3-0.6B，batch 1，上下文 1024
len(k)                                                      # 283
step_time_us(k, 2.0)                                        # (460.6, 1026.6, 0.551)
```

<!-- 题解 -->
一步的时间 = 读写的数据 / 带宽 + kernel 数 × 空隙。Qwen3-0.6B 在 batch 1 时一共只读写约 1.2 GB，283 个 kernel 里一半连 1 µs 都不到，而每个边界的空隙是 1～2 µs 的量级，所以空隙能占到一半以上；Qwen3-8B 每个 kernel 平均读写十几微秒的数据，同样的空隙只占一成左右；batch 64、上下文 4096 时读 KV 的时间占了大头，空隙降到 5%。

所以减少空隙的手段（PDL 让相邻 kernel 重叠、融合减少 kernel 的个数、megakernel 去掉边界）主要服务小模型、小 batch、低延迟的场景。不用 CUDA Graph 时 CPU 每个 kernel 5 µs 的发射开销更是远超 GPU 上的工作，一步 1.4 ms，这就是 decode 离不开 CUDA Graph 的原因。
