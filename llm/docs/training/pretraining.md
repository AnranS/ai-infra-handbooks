# 预训练与 Scaling Law

<p class="lead">做推理不需要会训练大模型，但需要知道模型是怎么来的：训练目标、数据规模、算力估算、为什么现在的小模型要"过度训练"。这些知识能帮你理解模型的能力边界，也能让你看懂训练和推理在显存、并行、精度上的共同点与差异。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 训练一个 N 参数的模型、使用 D 个 token，大约需要多少 FLOPs？为什么系数是 6？
    2. Chinchilla 定律说了什么？为什么 LLaMA-3-8B 要用远超"最优"的数据量训练？
    3. 用 AdamW 混合精度训练，每个参数要占多少字节显存？
    4. MFU 是什么？
    5. 常见的训练并行方式有哪些？和推理的并行有什么不同？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 约 $6ND$。前向每个 token 每个参数一次乘加（2 次运算），反向要算对激活的梯度和对权重的梯度，各一次同样大的矩阵乘（4 次），合计 6。
    2. 给定算力时，参数量和数据量应同比例增长，最优约 20 token/参数。但推理成本只和模型大小有关，小模型多训练一些能在同样的推理成本下更强，所以 LLaMA-3-8B 用了约 15T token（约 1900 token/参数）。
    3. 约 16 字节：bf16 的参数 2 + 梯度 2，fp32 的主权重 4 + Adam 的两个矩各 4。
    4. 模型算力利用率：实际完成的模型计算量（按每 token $6N$ 计，不含重计算）÷ 硬件的峰值算力；大规模训练通常在 35%～55%。
    5. 数据并行（含 ZeRO / FSDP）、张量并行、流水线并行、上下文并行、专家并行。训练还要放下梯度和优化器状态、处理反向与大量激活；推理没有这些，关心的是延迟和 KV Cache，常用 TP、EP、DP Attention 和 PD 分离。

## 预训练做什么

目标只有一个：在海量文本上做[下一个 token 预测](../basics/language-model.md#训练目标交叉熵)，最小化交叉熵。

- **数据**：网页、书籍、代码、论文、数学等，经过去重、质量过滤、敏感内容过滤，按一定比例混合。主流模型的训练数据量已经到了十几万亿（10^13）token 的级别：LLaMA-3 超过 15T，Qwen2.5 为 18T，DeepSeek-V3 为 14.8T；
- **序列打包**：把多篇文档首尾相接，切成固定长度（比如 4096 或 8192）的训练样本，避免填充浪费；
- **长度分阶段**：先在较短的上下文上训练大部分数据，最后用一个"长上下文阶段"扩展到 32K 或 128K（配合 RoPE 的调整，见[位置编码](../transformer/position.md#长上下文扩展)）；
- **优化器**：AdamW；学习率先预热（warmup）再余弦衰减，或者使用"预热-稳定-衰减"（WSD）的调度；梯度裁剪；每一步的 batch 通常是数百万个 token。

下面用 `mini_llm.py` 在一小段文本上跑几十步训练，完整走一遍流程：

```python
import math
import torch
import torch.nn.functional as F
from mini_llm import Config, Transformer

torch.manual_seed(0)
text = "北京是中国的首都。上海是中国最大的城市。" * 40
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text])

cfg = Config(vocab_size=len(chars), hidden_size=64, intermediate_size=172, num_hidden_layers=2,
             num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
model = Transformer(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
steps, warmup, seq_len = 60, 6, 32
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))

losses = []
for step in range(steps):
    starts = torch.randint(0, len(data) - seq_len - 1, (16,))
    x = torch.stack([data[s:s + seq_len] for s in starts])          # [16, 32]
    y = torch.stack([data[s + 1:s + seq_len + 1] for s in starts])  # 错开一位的目标
    loss = F.cross_entropy(model(x).view(-1, cfg.vocab_size), y.view(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # 梯度裁剪
    opt.step()
    sched.step()
    losses.append(loss.item())
print(f"loss: {losses[0]:.2f} -> {losses[-1]:.2f}")
assert losses[-1] < losses[0] / 3
```

真实的预训练就是把这个循环放大到上千张 GPU、数万亿 token，再加上数据管道、并行、容错、监控。

## 训练需要多少算力

[前面](../basics/math-torch.md#矩阵乘法线性层)讲过，前向传播每个 token 约 2N 次运算（N 为参数量）。反向传播要计算对输入和对权重的两份梯度，约为前向的两倍，即 4N。所以：

$$
C_{\text{train}} \approx 6ND
$$

```python
def train_flops(params, tokens):
    return 6 * params * tokens

C = train_flops(8.03e9, 15e12)                      # LLaMA-3-8B 规模：80 亿参数，15 万亿 token
h100_bf16 = 989e12                                  # H100 SXM 的 BF16 稠密峰值
gpu_hours = C / (0.4 * h100_bf16) / 3600             # 假设 40% 的算力利用率
print(f"{C:.2e} FLOPs，约 {gpu_hours / 1e3:.0f}k 个 H100 小时")
```

**MFU（Model FLOPs Utilization，模型算力利用率）** = 实际达到的有效算力 / 硬件峰值。大规模训练的 MFU 通常在 35%-55% 之间。推理的 prefill 阶段也用同样的指标衡量；decode 阶段是访存瓶颈，更适合用带宽利用率衡量。

## Scaling Law

**Kaplan 等（2020）** 发现：模型的损失随参数量、数据量、算力按**幂律**平滑下降，可以用小规模实验预测大规模训练的结果。

**Chinchilla（Hoffmann 等，2022）** 回答了：给定算力预算 C，参数量和数据量怎么分配损失最低？结论是两者应该同比例增长，**最优的数据量约为参数量的 20 倍**。按这个标准，70B 的模型用 1.4T token 训练就"够了"。

这个结论来自一个拟合出来的损失函数 $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$：把它画成曲面，算力预算 $C = 6ND$ 就是曲面上的一条线，沿线找最低点就是最优分配。拖动旋转，再拉一拉预算：

<div class="aig-widget" data-widget="scaling3d"></div>

但今天的模型普遍远超这个比例：LLaMA-3-8B 用了 15T token，是参数量的约 1900 倍。原因在推理：

!!! inference "推理视角"
    Chinchilla 最优只考虑了**训练**成本。而一个模型发布后要被调用成千上万亿次，推理成本远超训练。**把一个小模型训练得更久**（"过度训练"），虽然训练效率不是最优，但得到的是一个同等能力下更小、更便宜的推理模型。这就是 7B、8B 这类小模型用十几万亿 token 训练的原因，也是 MoE 流行的原因：用较少的激活参数获得较大的容量。

## 训练的显存

用 AdamW 做混合精度训练，每个参数需要：

| 项目 | 字节 |
| --- | --- |
| BF16 权重（前向、反向使用） | 2 |
| BF16 梯度 | 2 |
| FP32 主权重（优化器更新用） | 4 |
| Adam 的一阶矩 m（FP32） | 4 |
| Adam 的二阶矩 v（FP32） | 4 |
| **合计** | **16** |

一个 7B 模型仅这部分就要 112 GB，还不算激活值。而推理只需要权重本身（BF16 下 2 字节/参数）加上 KV Cache。

```python
def train_state_gb(params, bytes_per_param=16):
    return params * bytes_per_param / 2**30

for name, n in [("0.6B", 0.596e9), ("7B", 7e9), ("70B", 70e9)]:
    print(f"{name}: 训练状态约 {train_state_gb(n):.0f} GB，推理权重（BF16）约 {n * 2 / 2**30:.0f} GB")
```

所以大模型训练必须把这些状态切分到多张卡上：

| 并行方式 | 切分什么 | 说明 |
| --- | --- | --- |
| 数据并行 + ZeRO / FSDP | 优化器状态、梯度、权重按数据并行的卡切分，用时再临时收集 | 训练最常用 |
| 张量并行（TP） | 每个矩阵切到多张卡 | 与推理的 TP 相同，限于机内 |
| 流水线并行（PP） | 按层切分 | 需要把 batch 切成微批次填满流水线 |
| 上下文并行（CP） | 沿序列长度切分 | 长上下文训练 |
| 专家并行（EP） | MoE 的专家分到不同的卡 | 与推理的 EP 相同 |

此外还有**激活重计算**（前向时不保存某些中间结果，反向时重新计算），用算力换显存。

## 多 token 预测

DeepSeek-V3 在训练时增加了一个**多 token 预测（MTP）**目标：除了预测下一个 token，还用一个额外的小模块预测再下一个 token。它让训练信号更密集；更重要的是，推理时这个模块可以直接作为**投机解码的草稿模型**，一次"猜"出多个 token 再由主模型验证，见[推理服务](../inference/serving.md#投机解码)。这是训练设计直接服务于推理加速的一个好例子。

!!! interview "面试怎么答"
    训练相关的估算题：训练算力约 6ND（前向 2N、反向 4N）；Chinchilla 最优约 20 token/参数，但为了降低推理成本，小模型普遍"过度训练"（LLaMA-3-8B 用了约 15T token）；混合精度 AdamW 每个参数约 16 字节（推理只要 2 字节），所以训练离不开 ZeRO / FSDP、TP、PP；MFU 衡量算力利用率。能把训练侧的设计（MTP、GQA、MoE）和推理成本联系起来，会是加分项。

## 练习

**1. 估算。** 用 Chinchilla 最优比例（20 token/参数），1e24 FLOPs 的算力预算应该训练多大的模型、用多少数据？

??? success "参考答案"
    C = 6ND，D = 20N，所以 C = 120N²，N = √(C/120) ≈ 9.1e10（约 91B 参数），D ≈ 1.8e12（约 1.8T token）。

    ```python
    import math
    C = 1e24
    N = math.sqrt(C / 120)
    print(f"N ≈ {N / 1e9:.0f}B, D ≈ {20 * N / 1e12:.1f}T")
    assert 90e9 < N < 92e9
    ```

**2. 思考题。** 训练时的 batch 通常是几百万个 token，推理时一个 decode 批次只有几十到几百个 token。为什么训练的 GPU 利用率通常比推理高得多？

??? success "参考答案"
    训练时，每个矩阵乘法的 M 维度（batch × 序列长度）有数百万，权重读一次被复用数百万次，完全是计算瓶颈，能充分利用 Tensor Core。推理的 decode 阶段每个请求每步只有 1 个 token，batch 中所有请求加起来也只有几十到几百个 token，权重读取无法被充分摊薄，是访存瓶颈。提高 decode 效率的核心手段（更大的 batch、量化、投机解码一次验证多个 token）本质上都是在提高每次读权重时参与计算的 token 数，或者减少读取的字节数。

## 小结

- [x] 预训练 = 在十几万亿 token 上最小化下一个 token 的交叉熵；训练算力约 6ND。
- [x] Chinchilla 最优约 20 token/参数，但为了降低推理成本，小模型普遍"过度训练"。
- [x] 混合精度 AdamW 训练每个参数约 16 字节，远超推理的 2 字节；需要 ZeRO/FSDP、TP、PP 等并行。
- [x] MFU 衡量算力利用率；MTP 等训练设计可以直接用于推理加速。
