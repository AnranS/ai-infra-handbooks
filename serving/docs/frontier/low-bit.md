# 超大 MoE 的低比特推理：INT4、FP4 与量化感知训练

<p class="lead"><a href="../../perf/quantization-deploy/">量化部署实战</a>一章比较了各种低精度格式在小模型上的精度。模型到了万亿参数，低比特就不再只是"省一点显存"：FP8 的万亿模型要两台 8 卡机，4 比特只要一台，部署的最小单元、通信方式、decode 的延迟下限都随之改变。另一方面，推理模型动辄输出上万个 token，每一步一点点的量化误差会沿着自回归链条放大。这一章先算清楚 4 比特对超大 MoE 意味着什么，再用一个小实验说明为什么发布方越来越多地直接用<b>量化感知训练</b>（QAT）产出 INT4 / FP4 权重，最后看不同硬件上对应的 kernel。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个万亿参数的 MoE，BF16、FP8、4 比特分别要几张 H200？
    2. 为什么单步准确率只差 1%，长输出的质量可能差很多？
    3. 训练后量化（PTQ）和量化感知训练（QAT）的区别是什么？QAT 的梯度怎么穿过取整？
    4. Hopper 上跑 INT4 权重和 Blackwell 上跑 FP4 权重，kernel 有什么不同？
    5. W4A16、W4A8、W4A4 各适合什么场景？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 按本章"万亿总参数、激活 320 亿"的例子（留 30% 给 KV 和激活）：BF16 约 2 TB，要 21 张（3 台 8 卡机）；FP8 约 1 TB，要 11 张（2 台）；4 比特约 0.53～0.56 TB，6 张就够（1 台）。
    2. 推理模型的输出很长，一个 token 错了，后面的推理链就可能跟着走偏；单步 99% 的准确率，256 步全对的概率只有约 6%（$0.99^{256}$），小误差被长链条放大。
    3. PTQ 在训练好的模型上直接量化，不再训练；QAT 在训练（后训练）中就用伪量化的前向，让模型适应量化误差。取整的梯度几乎处处为 0，用直通估计（STE）：反向时把取整当成恒等函数，梯度直接穿过。
    4. Hopper 没有 FP4 Tensor Core：INT4 权重在寄存器里反量化成 BF16 / FP8，再用对应精度的 Tensor Core 计算（W4A16 / W4A8）；Blackwell 有 MXFP4 / NVFP4 原生的块缩放 Tensor Core，权重（和激活）直接以 FP4 参与矩阵乘。
    5. W4A16：只减少读权重的字节，适合 decode、小 batch、延迟敏感的场景；W4A8：计算也用 8 位，Hopper 上兼顾 decode 与 prefill；W4A4（FP4）：Blackwell 上计算也用 4 位，prefill 和大 batch 也能受益，但精度要求更高（通常配合 QAT）。

## 4 比特对万亿 MoE 意味着什么

以一个"万亿总参数、每 token 激活 320 亿"的 MoE 为例（示意的尺寸），算一下不同格式的权重总量、至少要几张 H200（每张 141 GB，留 30% 给 KV 和激活），以及 8 卡 decode 时每个 token 读权重的时间下限：

```python
import math

TOTAL, ACTIVE = 1.0e12, 32e9                          # 示意：万亿总参数、每个 token 激活 320 亿的 MoE
FORMATS = {                                           # 每个参数的平均字节数（含缩放因子）
    "BF16": 2.0,
    "FP8（128×128 分块）": 1 + 4 / 128**2,
    "INT4（每 32 个一个 bf16 缩放）": 0.5 + 2 / 32,
    "MXFP4（每 32 个一个 E8M0）": 0.5 + 1 / 32,
    "NVFP4（每 16 个一个 E4M3）": 0.5 + 1 / 16,
}
GPU_MEM, GPU_BW = 141e9, 4.8e12                       # H200 一张卡的显存与带宽
for name, b in FORMATS.items():
    w = TOTAL * b
    gpus = math.ceil(w / (GPU_MEM * 0.7))
    print(f"{name}：权重 {w / 1e12:.2f} TB，至少 {gpus} 张 H200（留 30% 给 KV，{math.ceil(gpus / 8)} 台 8 卡机），"
          f"8 卡 decode 每 token 读权重至少 {ACTIVE * b / (8 * GPU_BW) * 1e3:.2f} ms")
```

```text title="输出"
BF16：权重 2.00 TB，至少 21 张 H200（留 30% 给 KV，3 台 8 卡机），8 卡 decode 每 token 读权重至少 1.67 ms
FP8（128×128 分块）：权重 1.00 TB，至少 11 张 H200（留 30% 给 KV，2 台 8 卡机），8 卡 decode 每 token 读权重至少 0.83 ms
INT4（每 32 个一个 bf16 缩放）：权重 0.56 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.47 ms
MXFP4（每 32 个一个 E8M0）：权重 0.53 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.44 ms
NVFP4（每 16 个一个 E4M3）：权重 0.56 TB，至少 6 张 H200（留 30% 给 KV，1 台 8 卡机），8 卡 decode 每 token 读权重至少 0.47 ms
```

从 FP8 到 4 比特，部署单元从"两台机器"变成"一台机器"，影响远不止显存：

- **不需要跨机的专家并行**：整个模型在一个 NVLink 域里，EP 的 all-to-all 全走 NVLink（[上一部分](../comm/interconnect.md)：比网卡快 9 倍），不再需要 DeepEP 那一套跨机优化，小规模部署也能跑；
- **decode 的延迟下限减半**：小 batch 时 decode 受权重读取限制，读的字节少一半，单请求的速度上限高一倍——对输出很长的推理模型尤其重要；
- **缩放因子的开销不能忽略**：每 32 个 4 比特数配一个 bf16 缩放，额外 12.5%；NVFP4 每 16 个一个 FP8 缩放，同样 12.5%；MXFP4 的缩放只有 1 字节 / 32 个，额外约 6%。

## 为什么要量化感知训练

**训练后量化**（PTQ）：训练完直接把权重取整到低精度（可以用 GPTQ、AWQ 这类方法减小误差）。**量化感知训练**（QAT）：在训练（或微调）的前向里就使用量化后的权重，让模型"适应"取整误差；取整没有梯度，反向传播时用**直通估计**（STE）——把取整当成恒等函数，梯度原样传给背后的高精度权重。

对推理模型，PTQ 的问题在于误差会沿着输出链条放大：一步错了，后面的每一步都建立在错误的上下文上。用一个小实验模拟这个过程：一个两层的小网络学会一张"由前两个 token 决定下一个 token"的表（相当于一个完全记住规则的"语言模型"），然后分别做 INT4 的 PTQ 和 QAT，看单步准确率和 256 步贪心生成的正确率：

```python
import torch

torch.manual_seed(0)
torch.set_num_threads(4)
V, H, G = 64, 224, 32                                # 词表、隐藏维度、INT4 量化的分组大小
table = torch.randint(0, V, (V, V))                  # "老师"：下一个 token 由前两个 token 查表决定
ctx = torch.cartesian_prod(torch.arange(V), torch.arange(V))
target = table[ctx[:, 0], ctx[:, 1]]


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(V, 32)
        self.fc1 = torch.nn.Linear(64, H)
        self.fc2 = torch.nn.Linear(H, V)

    def forward(self, x, quant=False):
        h = self.emb(x).flatten(1)
        w1, w2 = (fake_int4(self.fc1.weight), fake_int4(self.fc2.weight)) if quant else (self.fc1.weight, self.fc2.weight)
        h = torch.relu(torch.nn.functional.linear(h, w1, self.fc1.bias))
        return torch.nn.functional.linear(h, w2, self.fc2.bias)


def fake_int4(w):
    """对称 INT4、每 G 个输入通道一组；前向用量化值，反向把梯度原样传给 fp32 权重（直通估计 STE）"""
    g = w.view(w.shape[0], -1, G)
    s = g.abs().amax(-1, keepdim=True) / 7
    q = (g / s).round().clamp(-8, 7) * s
    return w + (q.view_as(w) - w).detach()


def train(model, steps, quant, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(model(ctx, quant), target)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def evaluate(model, quant):
    acc = (model(ctx, quant).argmax(-1) == target).float().mean().item()
    starts = torch.randint(0, V, (200, 2), generator=torch.Generator().manual_seed(1))
    seq, ok = starts.clone(), torch.ones(200, dtype=torch.bool)
    lengths = torch.zeros(200)
    for _ in range(256):                             # 贪心自回归 256 步：一步错了，后面整条链都偏离
        nxt = model(seq[:, -2:], quant).argmax(-1)
        ok &= nxt == table[seq[:, -2], seq[:, -1]]
        lengths += ok.float()
        seq = torch.cat([seq, nxt[:, None]], 1)
    return acc, lengths.mean().item(), ok.float().mean().item()


base = Tiny()
train(base, 3000, False, 3e-3)
ptq = base                                           # 训练后直接量化（PTQ）
qat = Tiny()
qat.load_state_dict(base.state_dict())
train(qat, 2000, True, 1e-3)                         # 量化感知训练（QAT）：在伪量化的前向上继续训练
for name, model, quant in (("原模型（fp32）", base, False), ("INT4 训练后量化", ptq, True), ("INT4 量化感知训练", qat, True)):
    acc, run, full = evaluate(model, quant)
    print(f"{name}：单步准确率 {acc:.1%}，连续正确的平均步数 {run:.0f} / 256，整条链全对 {full:.0%}")
```

```text title="输出"
原模型（fp32）：单步准确率 100.0%，连续正确的平均步数 256 / 256，整条链全对 100%
INT4 训练后量化：单步准确率 98.9%，连续正确的平均步数 96 / 256，整条链全对 29%
INT4 量化感知训练：单步准确率 100.0%，连续正确的平均步数 256 / 256，整条链全对 100%
```

PTQ 的单步准确率只掉了 1.1%，但 256 步的链条里有 71% 在中途出错，平均只走了 96 步——每步 1% 的错误率，256 步全对的概率是 $0.989^{256} \approx 6\%$，出错的输入还会让后续的错误更多（错误的上下文正是模型没见过的）。QAT 在伪量化的前向上继续训练 2000 步，单步准确率回到 100%，整条链全部正确。

真实模型当然不是一张查找表，但道理相同：推理模型的输出常常上万个 token，还要保持长链条上的逻辑一致，PTQ 在短基准上"几乎无损"，在长推理任务上却可能明显掉点。所以越来越多的模型直接发布 QAT 得到的低比特权重——gpt-oss 以 MXFP4 发布 MoE 权重，也有万亿参数的推理模型在后训练阶段做 INT4 QAT、直接发布 INT4 权重。QAT 的代价是训练侧的工作量：前向要插入伪量化，而且要在 RL 等后训练阶段同样使用（否则后训练又会把权重"训"回高精度的分布）。

## 不同硬件上的 kernel

| 方案 | 权重 | 激活与计算 | 硬件 | 适合 |
| --- | --- | --- | --- | --- |
| W4A16 | INT4 / FP4 | 在寄存器里把权重反量化成 BF16，用 BF16 Tensor Core 计算 | 各代 GPU（Hopper 没有 FP4 Tensor Core） | decode：受权重读取限制，读的字节少了就快 |
| W4A8 | INT4 | 权重反量化成 FP8，激活量化成 FP8，用 FP8 Tensor Core | Hopper | 兼顾 decode 的读取和 prefill 的算力 |
| W4A4（FP4） | MXFP4 / NVFP4 | 激活也量化成 FP4，直接用 FP4 Tensor Core | Blackwell | prefill 和大 batch：算力再翻倍 |

几个细节：

- **反量化的位置**：W4A16 的 kernel（例如 Marlin）在从共享内存搬到寄存器的路上完成反量化，并且把权重在显存里预先重排成方便向量化解包的布局——这正是 CUDA 手册[量化 GEMV](cuda://advanced/quantization/)一章的做法；
- **MoE 的分组 GEMM 也要支持低比特**：每个专家一个低比特的权重矩阵，配合路由的分组；vLLM 在 `model_executor/layers/fused_moe/` 下为不同格式提供了专家计算的实现（`experts/marlin_moe.py` 等），并由 `oracle/` 下的 `int_wna16.py`、`mxfp4.py`、`fp8.py` 按格式和硬件选择 kernel；
- **decode 与 prefill 的取舍不同**：decode 时 4 比特权重让读取减半，W4A16 就够了；prefill 是计算受限的，W4A16 的计算仍然是 BF16，速度不会比 FP8 快，Blackwell 的 FP4 Tensor Core 才能让 prefill 也受益；
- **KV Cache 另算**：权重 4 比特之后，长上下文时 KV 往往成为显存的大头，通常还要配合 FP8 KV。

!!! interview "面试怎么答"
    被问到"万亿参数模型怎么部署"或"为什么用 INT4 QAT"，先算账：FP8 要两台 8 卡机、4 比特一台，跨机 EP 变成机内 EP，decode 延迟下限减半；再讲精度：推理模型的长输出让 PTQ 的小误差沿链条放大（单步 99% 的准确率，256 步全对只剩约 6%），所以要在后训练中做 QAT，用 STE 让梯度穿过取整；最后讲硬件：Hopper 上 W4A16 / W4A8（在寄存器里反量化），Blackwell 上 MXFP4 / NVFP4 原生 Tensor Core，decode 和 prefill 的收益不同。

## 练习

**1. STE 为什么能工作？** 取整函数的导数几乎处处为 0，STE 却直接把梯度原样传过去。这样训练出来的权重为什么对量化更"友好"？

??? success "参考答案"
    STE 让损失对"量化后的权重"的梯度作用到背后的高精度权重上：前向看到的是量化误差带来的真实损失，反向就会把高精度权重往"量化之后损失更小"的方向推。训练的结果是高精度权重落在量化网格上"安全"的位置（远离取整的分界点、或者让取整误差彼此抵消），决策边界也相应地调整，使模型对剩余的量化误差不敏感。STE 的梯度是有偏的近似，但在量化噪声不太大时足够好用，这也是 QAT 通常在已训练好的模型上做少量微调、而不是从头训练的原因。

**2. 缩放因子的精度。** MXFP4 的缩放因子是 E8M0（只能是 2 的幂），NVFP4 是 E4M3（有 3 位尾数）、而且每 16 个数一个。它们各自的利弊是什么？

??? success "参考答案"
    E8M0 缩放只能取 2 的幂，乘法变成指数加法、硬件简单，但缩放不精确：一组的最大值被放大到 FP4 的范围里时，最坏情况下浪费接近一倍的动态范围（相当于少用了一个档位）。NVFP4 的 E4M3 缩放能精确地把每组的最大值对准 FP4 的最大值 6，每 16 个数一组也更细，所以精度更好（[量化部署实战](../perf/quantization-deploy.md)一章实测过），代价是缩放因子多占一倍的空间（每 16 个数 1 字节），并且需要一个额外的张量级 FP32 缩放来扩大 E4M3 的表示范围。

## 小结

- [x] 万亿 MoE 从 FP8 到 4 比特：部署单元从两台机器变成一台，跨机 EP 变成机内 EP，decode 延迟下限减半；缩放因子额外占 6%～12.5%。
- [x] 长输出会放大量化误差：单步 1% 的错误，256 步的链条大部分会中途出错；QAT 用 STE 在伪量化的前向上微调，可以把误差训回去，越来越多的模型直接发布 QAT 的低比特权重。
- [x] Hopper 上用 W4A16 / W4A8（寄存器里反量化），Blackwell 上用 MXFP4 / NVFP4 的原生 Tensor Core；decode 靠少读字节受益，prefill 要靠低精度算力受益。
