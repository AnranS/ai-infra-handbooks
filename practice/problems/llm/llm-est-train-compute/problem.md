---
title: 估算：训练算力、GPU 时与 MFU
chapter: training/pretraining.md
difficulty: 中等
tags: [估算, 6ND, MFU, 训练]
---
训练一个稠密（或 MoE）模型要多少算力、多少 GPU 时？实测吞吐对应多高的 MFU？这是面试里最常见的估算题。约定：

- 训练总算力 $C \approx 6ND$：$N$ 是（激活）参数量，$D$ 是训练 token 数（前向 $2N$、反向 $4N$ 每 token）；
- 算上注意力的每 token 训练 FLOPs（PaLM 的写法）：$6N + 12 \cdot L \cdot d_{attn} \cdot s$，$L$ 层数，$d_{attn}$ = 头数 × 头维度，$s$ 序列长度；
- MFU = 实测 tokens/s × 每 token 训练 FLOPs ÷（GPU 数 × 单卡峰值）；
- 全量激活重计算时前向多做一遍，硬件实际执行的每 token FLOPs 是 $8N + 16 \cdot L \cdot d_{attn} \cdot s$，按它算出来的叫 HFU。

实现（`peak_tflops` 的单位是 $10^{12}$ FLOPs/s）：

1. `train_flops(n_params, n_tokens)`：$6ND$；
2. `gpu_hours(total_flops, peak_tflops, mfu)`：在给定 MFU 下需要多少 GPU 时；
3. `flops_per_token(n_params, n_layers, d_attn, seq_len)`：含注意力项的每 token 训练 FLOPs；
4. `mfu(tokens_per_s, n_gpus, peak_tflops, n_params, n_layers, d_attn, seq_len)`；
5. `hfu(tokens_per_s, n_gpus, peak_tflops, n_params, n_layers, d_attn, seq_len)`：全量重计算时的硬件利用率。

```python
train_flops(405e9, 15.6e12)             # 约 3.79e25：和 Llama 3.1 405B 公开的 3.8e25 FLOPs 一致
gpu_hours(train_flops(37e9, 14.8e12), 989, 0.35)   # 约 264 万 GPU 时（37B 激活参数的 MoE，H800 BF16 峰值）
```

先别写代码，口算：7B 模型训练 2T token 要多少 FLOPs？在 1024 张 H100（989 TFLOPS）上以 40% MFU 训练要几天？

<!-- 题解 -->
口算：$6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}$ FLOPs；$1024 \times 989\times10^{12} \times 0.4 \approx 4.05\times10^{17}$ FLOPs/s，
约 $2.07\times10^5$ 秒，也就是 2.4 天。

- MoE 用**激活参数**算：每个 token 只经过被选中的专家。第二个例子得到约 264 万 GPU 时，和公开报告里同规模 MoE 的预训练用时（约 266 万 H800 GPU 时）在同一量级。
- 注意力项 $12 L d_{attn} s$ 在短序列时可以忽略；以 8B 模型为例，8K 序列时约占 21%，32K 时超过一半。长序列训练只用 $6N$ 会严重低估 MFU。
- MFU 衡量"有用功"，重计算做的额外前向不算；HFU 把它算进去，所以 HFU ≥ MFU。汇报时要说清楚是哪个。
