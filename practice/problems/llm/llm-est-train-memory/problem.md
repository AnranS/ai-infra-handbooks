---
title: 估算：训练显存账本（优化器状态、ZeRO 与激活重计算）
chapter: training/pretraining.md
difficulty: 中等
tags: [估算, 显存, ZeRO, 重计算]
---
混合精度 + Adam 训练时，每个参数要占多少显存？ZeRO 各阶段、激活重计算分别省下多少？约定（$\Psi$ 为参数量）：

- **模型状态**：bf16 参数 2 字节 + bf16 梯度 2 字节 + fp32 主参数、一阶矩、二阶矩共 12 字节 = $16\Psi$；
- 数据并行度 $N$ 时，ZeRO-1 切分优化器状态：$4\Psi + 12\Psi/N$；ZeRO-2 再切梯度：$2\Psi + 14\Psi/N$；ZeRO-3 连参数一起切：$16\Psi/N$；
- **激活**（单卡、不做张量并行，bf16，Megatron 论文的估算）：每层 $sbh\,(34 + 5as/h)$ 字节，$s$ 序列长度、$b$ micro-batch、$h$ 隐藏维度、$a$ 注意力头数；
  用 FlashAttention 时不再保存 $s \times s$ 的注意力分数，每层变成 $34\,sbh$；全量重计算时每层只保存输入，$2\,sbh$。

实现：

1. `model_state_bytes(n_params, zero_stage=0, dp=1)`；
2. `activation_bytes(seq, batch, hidden, heads, n_layers, flash=True, recompute=False)`：`recompute=True` 时按全量重计算算（与 `flash` 无关）；
3. `per_gpu_gib(n_params, dp, zero_stage, act_bytes)`：每卡总显存（模型状态 + 激活），单位 GiB（$2^{30}$ 字节）。

```python
model_state_bytes(7e9)                  # 112e9：7B 模型光模型状态就要 112 GB，一张 80 GB 的卡放不下
model_state_bytes(7e9, 3, 8) / 2**30    # 约 13.0 GiB
```

先口算：70B 模型在 64 卡上用 ZeRO-3，每卡模型状态多少 GB？

<!-- 题解 -->
口算：$16 \times 70 / 64 = 17.5$ GB。

- 激活常常比模型状态还大：4K 序列、32 层、隐藏维度 4096 的模型，不用 FlashAttention 时激活约 97 GiB，用了约 17 GiB，全量重计算只要约 1 GiB（代价是多一次前向，约 33% 的额外计算）。
- ZeRO-3 能把模型状态压到 $16\Psi/N$，但每层前向、反向都要 all-gather 参数，通信量是普通数据并行的 1.5 倍。
- 真实训练还要算上张量并行 / 序列并行对激活的切分、通信缓冲、显存碎片，所以估算值只是下限。
