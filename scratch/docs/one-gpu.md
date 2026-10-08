# 一张消费级卡能训出什么：文本小模型与图像生成小模型

<p class="lead">前面三章的"从零训练"在 CPU 上跑通了全流程，这一章换到一张真卡上：16 GB 的 RTX 5070 Ti。回答三个问题——这张卡能放下多大的模型、训完要多久、怎么配才不浪费算力；然后用它训两个东西：一个能续写中文的文本小模型，和一个能出图的图像生成小模型。图像那条线用<strong>流匹配</strong>（flow matching），训练目标只有一行，比 DDPM 少一半超参，本章的最小实现在 CPU 上几秒钟就能看到它学会一个分布。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 16 GB 显存里，一个 350M 的模型训练时各部分各占多少？哪一部分最容易失控？
    2. 整层激活重计算能省多少显存、多花多少时间？什么时候该开？
    3. 流匹配的训练目标是什么？它和 DDPM 的关系是什么？
    4. 一张消费级卡上，`torch.compile`、bf16、FlashAttention、梯度累积，哪个带来的收益最大？
    5. 训练崩了（loss 变 NaN 或者突然飞起来），你按什么顺序查？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 混合精度 AdamW 下模型状态是 $16\Psi$ 字节（bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数副本 4Ψ + Adam 的 m、v 各 4Ψ），350M 就是 5.2 GB；激活随 `层数 × 序列长度 × batch` 线性增长，是最容易失控的一项——不开重计算时 seq 1024、micro-batch 8 就要 6.4 GB。
    2. 整层重计算把每层只留输入，激活从每 token 约 $34d$ 字节降到约 $2d$，大约省 17 倍；代价是反向时重做一次前向，约 +30% 时间。显存不够、或者想把 batch 开大到能吃满算力时就该开。
    3. 训练目标是让网络预测"从噪声端到数据端的直线速度"：在 $x_t = (1-t)x_0 + t x_1$ 上回归 $x_1 - x_0$，一行 MSE。DDPM 预测的是加进去的噪声、要设计噪声调度和很多超参；流匹配等价于一种特定的概率路径，形式更简单，采样时直接用常微分方程积分，步数可以很少。
    4. 通常是 **bf16 + FlashAttention**（省显存又提速），其次 `torch.compile`（消灭 kernel 启动和访存往返，小模型上尤其明显），梯度累积只解决"显存不够但想要大 batch"，本身不提速。
    5. 先看是不是数值问题（换 fp32 跑几步还崩吗）、再看 LR 和 warmup（把 LR 砍十倍还崩吗）、再看数据（有没有异常长的样本或全是同一个 token 的片段）、最后看梯度裁剪和 loss 缩放。同时盯 `grad_norm` 曲线：突刺通常比 loss 早出现几十步。

## 先量这张卡

任何估算之前先拿到两个数：**实测显存带宽**和**实测 bf16 算力**。标称值（5070 Ti：16 GB GDDR7、256 bit、28 Gbps，理论带宽约 896 GB/s）只是上限，真正能用到的要测。

```python title="bench_gpu.py" run="no"
# 在你自己的卡上测两个数：显存带宽和 bf16 矩阵乘算力。后面所有估算都用这两个数当分母。
import time

import torch

assert torch.cuda.is_available()
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), f"| 显存 {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB",
      f"| 计算能力 sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")


def timed(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


n = 1 << 26                                     # 256 MB 的 bf16 张量，纯访存
a = torch.randn(n, dtype=torch.bfloat16, device=dev)
b = torch.empty_like(a)
dt = timed(lambda: b.copy_(a))
print(f"显存带宽（读+写）{2 * a.numel() * 2 / dt / 1e9:.0f} GB/s")

for m in (2048, 4096, 8192):                    # 方阵矩阵乘，纯算力
    x = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    y = torch.randn(m, m, dtype=torch.bfloat16, device=dev)
    dt = timed(lambda: torch.mm(x, y))
    print(f"bf16 矩阵乘 {m}×{m}：{2 * m ** 3 / dt / 1e12:.0f} TFLOPS")
```

消费级 Blackwell 的 bf16 稠密张量核心算力在 **150～180 TFLOPS 这个量级**（以你实测为准；本章后面的时间估算用的是"有效算力"，也就是实测峰值乘以 MFU，典型 40%～50%）。带宽实测通常能到标称的 80%～90%。

!!! warning "Blackwell（sm_120）的环境坑"
    RTX 50 系是 `sm_120`，比 Ada（`sm_89`）新一代。装环境时注意：

    - **PyTorch 要 2.7 以上、且是 cu128 及更新的 wheel**。老版本的 wheel 里没有 `sm_120` 的 cubin，跑起来会提示 `no kernel image is available for execution on the device`，或者悄悄退回到 PTX JIT（第一次跑每个 kernel 都要编译，慢得离谱）。
    - 用 `torch.cuda.get_arch_list()` 确认输出里有 `sm_120`。
    - **FlashAttention 的预编译 wheel 往往还没跟上**。先用 PyTorch 自带的 `F.scaled_dot_product_attention`（它会自动选 FlashAttention / mem-efficient 后端），不要一上来就装第三方轮子。
    - `torch.compile` 第一次编译会慢几分钟，属于正常；把 `TORCHINDUCTOR_CACHE_DIR` 设到一个固定目录，后面就能复用。

## 显存账本：16 GB 放得下什么

```python title="budget.py"
# 一张 16 GB 的卡：多大的模型放得下，训完要多久
GB = 1024 ** 3
VRAM = 16 * GB
RESERVED = 1.2 * GB                      # 驱动、CUDA context、cuBLAS workspace 和碎片，先扣掉


def states(params):
    """混合精度 AdamW 的常驻显存 = 16Ψ 字节：
    bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数副本 4Ψ + Adam 的 m、v 各 4Ψ"""
    return 16 * params


def activations(layers, d, seq, batch, recompute):
    """激活。不重计算时每层每 token 大约 34d 字节（bf16，含注意力里的中间量）；
    整层重计算只留每层的输入，约 2d 字节，代价是多做一次前向（约 +30% 时间）。"""
    return layers * seq * batch * (2 if recompute else 34) * d


print("模型能不能放进 16 GB（seq 1024，micro-batch 8，整层重计算）")
for name, layers, d, params in [
    ("30M", 6, 384, 30e6), ("124M / GPT-2", 12, 768, 124e6), ("350M", 24, 1024, 350e6),
    ("770M", 24, 1536, 770e6), ("1.4B", 24, 2048, 1.4e9),
]:
    st, act = states(params), activations(layers, d, 1024, 8, True)
    total = st + act + RESERVED
    print(f"  {name:<14} {layers:>2} 层 × {d:<5} 权重+优化器 {st / GB:>5.1f} GB  "
          f"激活 {act / GB:>4.1f} GB  合计 {total / GB:>5.1f} GB  "
          f"{'放得下' if total < VRAM else '放不下，要更小的 batch 或 LoRA'}")

print()
print("不重计算的话激活是多少（同样 seq 1024、micro-batch 8）")
for name, layers, d in [("124M / GPT-2", 12, 768), ("350M", 24, 1024)]:
    a1 = activations(layers, d, 1024, 8, False)
    a2 = activations(layers, d, 1024, 8, True)
    print(f"  {name:<14} 不重计算 {a1 / GB:>5.1f} GB，整层重计算 {a2 / GB:>4.1f} GB，省了 {a1 / a2:>4.0f} 倍")

print()
print("训完要多久（Chinchilla 的 20 token / 参数，6ΨN FLOP）")
for name, params, tflops in [("30M", 30e6, 60), ("124M / GPT-2", 124e6, 90), ("350M", 350e6, 110)]:
    tokens = 20 * params
    hours = 6 * params * tokens / (tflops * 1e12) / 3600
    print(f"  {name:<14} {tokens / 1e9:>5.1f}B token，按实测有效算力 {tflops} TFLOPS 约 {hours:>5.1f} 小时")
```

```text title="输出"
模型能不能放进 16 GB（seq 1024，micro-batch 8，整层重计算）
  30M             6 层 × 384   权重+优化器   0.4 GB  激活  0.0 GB  合计   1.7 GB  放得下
  124M / GPT-2   12 层 × 768   权重+优化器   1.8 GB  激活  0.1 GB  合计   3.2 GB  放得下
  350M           24 层 × 1024  权重+优化器   5.2 GB  激活  0.4 GB  合计   6.8 GB  放得下
  770M           24 层 × 1536  权重+优化器  11.5 GB  激活  0.6 GB  合计  13.2 GB  放得下
  1.4B           24 层 × 2048  权重+优化器  20.9 GB  激活  0.8 GB  合计  22.8 GB  放不下，要更小的 batch 或 LoRA

不重计算的话激活是多少（同样 seq 1024、micro-batch 8）
  124M / GPT-2   不重计算   2.4 GB，整层重计算  0.1 GB，省了   17 倍
  350M           不重计算   6.4 GB，整层重计算  0.4 GB，省了   17 倍

训完要多久（Chinchilla 的 20 token / 参数，6ΨN FLOP）
  30M              0.6B token，按实测有效算力 60 TFLOPS 约   0.5 小时
  124M / GPT-2     2.5B token，按实测有效算力 90 TFLOPS 约   5.7 小时
  350M             7.0B token，按实测有效算力 110 TFLOPS 约  37.1 小时
```

三个结论：

- **模型状态不是瓶颈，激活才是**。770M 的权重加优化器要 11.5 GB，勉强塞得下；真正让人 OOM 的是激活，而激活可以用重计算压到几乎为零。
- **一张卡上的甜点区是 100M～350M**。124M（GPT-2 规模）一晚上能训完一轮完整的 Chinchilla 配比，350M 要一天半——够写在简历上，也够跑消融实验。
- **更大就该换思路**：1.4B 以上不要想从零训，改成 **LoRA 微调**一个现成的开源模型（只训 0.1% 的参数，优化器状态从 $12\Psi$ 降到几乎为零）。

## 文本小模型：一张卡一晚上

模型结构直接用[从零训练（二）](model.md)里那个，换成 GPT-2 规模（12 层、768 维、12 头、seq 1024）。关键在训练配置：

```python title="train_text.py" run="no"
# GPT-2 规模的小模型，一张 16 GB 卡上的训练配置。模型结构见"从零训练（二）"。
import math
import time

import torch

from model import GPT, GPTConfig          # 从零训练（二）里的模型

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")          # 让 fp32 的矩阵乘也走 TF32

cfg = GPTConfig(n_layer=12, n_head=12, d_model=768, seq_len=1024, vocab_size=32000)
model = GPT(cfg).to(dev)
model = torch.compile(model)                        # 第一次编译几分钟，之后复用缓存

MICRO_BS, ACCUM = 8, 16                             # 有效 batch = 8 × 16 × 1024 ≈ 131k token
STEPS, WARMUP, LR = 20000, 400, 6e-4

# AdamW：只给二维以上的参数加 weight decay（LayerNorm 和 bias 不加）
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_at(step):                                     # 线性 warmup + 余弦退火到 10%
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * t))


def get_batch():                                     # 换成你自己的数据加载
    raise NotImplementedError


for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for micro in range(ACCUM):                       # 梯度累积：小显存也能有大 batch
        x, y = get_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)                        # GPT.forward 传了 targets 就顺便算 loss（内部用 fp32 算 softmax）
        (loss / ACCUM).backward()                        # loss 要除以累积步数
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok = MICRO_BS * ACCUM * cfg.seq_len
        print(f"step {step:>6} loss {loss.item():.4f} grad_norm {gnorm:.2f} "
              f"{tok / dt:>8.0f} tok/s 显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
```

几个配置的理由：

| 配置 | 为什么 |
| --- | --- |
| `bf16` autocast，不用 fp16 | bf16 的指数位和 fp32 一样宽，不需要 loss scaling，也就不会有"scaler 一直在调、白跑很多步"的问题（见[混合精度与 FP8 训练](train://practice/mixed-precision/)） |
| `torch.compile` | 把 LayerNorm、残差、激活函数这些访存密集的算子融掉。小模型上这些算子占比高，收益尤其明显 |
| `fused=True` 的 AdamW | 参数更新本来是几百个小 kernel，融成一个 |
| 梯度累积 16 步 | 有效 batch 约 13 万 token。batch 太小时 loss 抖得厉害、学习率提不上去 |
| `set_float32_matmul_precision("high")` | 剩下的 fp32 矩阵乘走 TF32，几乎不损精度 |
| 整层重计算（本例没开） | 显存还够就别开，它要多花约 30% 时间。真要开，用 `checkpoint_sequential` 只包住 transformer 层 |

**怎么判断训得对不对**：

- **`grad_norm` 比 loss 更早报警**。正常情况下它应该在 warmup 之后稳定在一个窄区间（比如 0.2～1.0）。突然跳到几十，几步之后 loss 就会飞。
- **loss 的参照值**：GPT-2 规模、32k 词表，训到 Chinchilla 配比时验证集 loss 大约在 3.0～3.5（换算成困惑度 20～33）。第一步的 loss 应该约等于 $\ln(\text{vocab\_size}) \approx 10.4$——如果第一步就远小于这个数，多半是标签泄漏（`y` 没有右移）。
- **吞吐的参照值**：用你实测的 bf16 算力算出 MFU = $6\Psi \times \text{tok/s} / \text{实测算力}$。消费级卡上 35%～45% 算正常；低于 20% 就去 profile，通常是数据加载没跟上或者 batch 太小。

!!! tip "显存不够时的顺序"
    1. 先减 `MICRO_BS`、同步加大 `ACCUM`（有效 batch 不变，几乎不损吞吐）；
    2. 再开整层重计算（省 17 倍激活，+30% 时间）；
    3. 再把序列长度砍半（激活线性下降，但会影响长文本能力）；
    4. 最后才考虑 8-bit 优化器（`bitsandbytes` 的 `AdamW8bit`，把 $8\Psi$ 的 m、v 压成 $2\Psi$）；
    5. 还不够就说明模型选大了——换成 LoRA 微调现成模型。

## 图像生成小模型：流匹配

扩散模型的公式看起来吓人，但**流匹配**把它化简到了一行。思路是：在"噪声"和"数据"之间连一条直线

$$x_t = (1 - t)\,x_0 + t\,x_1, \qquad x_0 \sim \mathcal{N}(0, I),\ x_1 \sim \text{数据}$$

让网络 $v_\theta(x_t, t)$ 去预测这条直线的速度，而速度恰好就是两端之差 $x_1 - x_0$。训练目标就是一个 MSE：

$$\mathcal{L} = \mathbb{E}_{t, x_0, x_1}\ \big\|\, v_\theta(x_t, t) - (x_1 - x_0) \,\big\|^2$$

采样时从噪声出发，沿着这个速度场用欧拉法积分到 $t = 1$。没有噪声调度、没有 $\bar\alpha_t$、没有方差参数化的选择——只有"插值 + 回归"。

先在 CPU 上用二维分布看它到底在学什么：

```python title="flow2d.py"
# 流匹配（flow matching）最小实现：训练目标只有一行，在 CPU 上几秒钟就能看到它学会一个分布
import math

import torch
import torch.nn as nn

torch.manual_seed(0)
K, R = 8, 2.0                                    # 目标分布：半径 2 的圆上 8 个高斯


def sample_data(n):
    idx = torch.randint(0, K, (n,))
    ang = idx.float() * (2 * math.pi / K)
    center = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
    return center + 0.08 * torch.randn(n, 2), idx


class Velocity(nn.Module):
    """v(x, t)：给定位置和时间，预测该往哪个方向走"""

    def __init__(self, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


model = Velocity()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
recent = []
for step in range(1, 4001):
    x1, _ = sample_data(256)                     # 数据端
    x0 = torch.randn(256, 2)                     # 噪声端
    t = torch.rand(256, 1)
    xt = (1 - t) * x0 + t * x1                   # 直线插值：t=0 是噪声，t=1 是数据
    loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()   # 目标速度就是两端之差，整个方法只有这一行
    opt.zero_grad()
    loss.backward()
    opt.step()
    recent.append(loss.item())
    if step in (1, 200, 1000, 4000):
        print(f"step {step:>4}  最近 {len(recent):>4} 步的平均 loss {sum(recent) / len(recent):.2f}")
        recent = []


@torch.no_grad()
def sample(n, steps=100):                        # 采样：从噪声出发，用欧拉法沿速度场积分到 t=1
    x = torch.randn(n, 2)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt)
        x = x + model(x, t) * dt
    return x


gen = sample(2000)
ang = torch.arange(K).float() * (2 * math.pi / K)
centers = torch.stack([R * ang.cos(), R * ang.sin()], dim=1)
dist = torch.cdist(gen, centers)                 # 每个样本到最近模态的距离
near = dist.min(dim=1)
hit = (near.values < 0.3).float().mean().item()
counts = torch.bincount(near.indices, minlength=K)
print(f"落在某个模态 0.3 半径内的样本：{hit:.0%}")
print(f"八个模态各分到：{counts.tolist()}（理想是每个 250）")
```

```text title="输出"
step    1  最近    1 步的平均 loss 3.20
step  200  最近  199 步的平均 loss 2.26
step 1000  最近  800 步的平均 loss 1.93
step 4000  最近 3000 步的平均 loss 1.87
落在某个模态 0.3 半径内的样本：81%
八个模态各分到：[223, 260, 306, 192, 279, 242, 277, 221]（理想是每个 250）
```

两点要理解：

- **loss 不会降到 0，这是对的**。给定 $x_t$，可能的 $(x_0, x_1)$ 配对有无穷多种，网络只能预测条件期望，剩下的方差是不可约的。所以判断训得好不好要看**采样质量**，不能只看 loss 数字。上面那个"81% 的样本落在模态附近、八个模态大致均分"才是真正的指标。
- **不同的模态被均匀覆盖**说明没有模式崩塌——这是 GAN 最头疼的问题，扩散和流匹配天然不容易犯。

### 换成图像：一个小 UNet

二维换成 $32\times32\times3$ 的图像，网络换成 UNet，其余一模一样：

```python title="unet.py" run="no"
# 给 32×32 图像用的小 UNet：时间用正弦编码注入每个残差块
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(torch.nn.functional.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(torch.nn.functional.silu(self.n2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, base=128, tdim=256):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        c1, c2, c3 = base, base * 2, base * 2
        self.inp = nn.Conv2d(3, c1, 3, padding=1)
        self.d1, self.d2 = Block(c1, c1, tdim), Block(c1, c2, tdim)        # 32 → 16
        self.down1, self.down2 = nn.Conv2d(c1, c1, 3, 2, 1), nn.Conv2d(c2, c2, 3, 2, 1)
        self.mid = Block(c2, c3, tdim)                                     # 8×8 的瓶颈
        self.u2, self.u1 = Block(c3 + c2, c2, tdim), Block(c2 + c1, c1, tdim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 3, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t * 1000, self.tdim))
        h0 = self.inp(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        m = self.mid(self.down2(h2), temb)
        u2 = self.u2(torch.cat([self.up(m), h2], dim=1), temb)
        u1 = self.u1(torch.cat([self.up(u2), h1], dim=1), temb)
        return self.out(u1)
```

```python title="train_image.py" run="no"
# CIFAR-10 上训一个流匹配图像生成模型。和 flow2d.py 的训练目标一字不差，只是换了网络和数据。
import time

import torch
import torchvision
from torch.utils.data import DataLoader

from unet import UNet

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")

tf = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5] * 3, [0.5] * 3),          # 归一化到 [-1, 1]
])
ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=8, pin_memory=True,
                drop_last=True, persistent_workers=True)

model = UNet().to(dev).to(memory_format=torch.channels_last)
model = torch.compile(model)
ema = torch.optim.swa_utils.AveragedModel(model, avg_fn=lambda a, b, _: 0.999 * a + 0.001 * b)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0, fused=True)

EPOCHS = 200
for epoch in range(EPOCHS):
    t0, total, n = time.perf_counter(), 0.0, 0
    for x1, _ in dl:
        x1 = x1.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=dev)
        xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ((model(xt, t) - (x1 - x0)) ** 2).mean()          # 和二维版本完全一样的一行
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update_parameters(model)
        total, n = total + loss.item(), n + 1
    print(f"epoch {epoch:>3} loss {total / n:.4f} {time.perf_counter() - t0:.0f}s "
          f"显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
    if epoch % 20 == 19:
        torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, f"ckpt_{epoch}.pt")
```

```python title="sample_image.py" run="no"
# 采样：从噪声出发，沿速度场积分到 t=1。步数可以少到 20 步还能看
import torch
import torchvision

from unet import UNet

dev = torch.device("cuda")
model = UNet().to(dev)
model.load_state_dict({k.replace("module.", ""): v for k, v in torch.load("ckpt_199.pt")["ema"].items()
                       if not k.startswith("n_averaged")})
model.eval()


@torch.no_grad()
def sample(n=64, steps=50):
    x = torch.randn(n, 3, 32, 32, device=dev)
    for i in range(steps):                                   # 欧拉法；换成 Heun 法同样步数质量更好
        t = torch.full((n,), i / steps, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, t).float()
        x = x + v / steps
    return (x.clamp(-1, 1) + 1) / 2


torchvision.utils.save_image(sample(), "samples.png", nrow=8)
print("已写出 samples.png")
```

这个 UNet 在 `base=128` 时是 **6.1M 参数**（上面的形状和反向都在 CPU 上验证过）。想要更好的质量就把 `base` 调大：160 是 9.3M，192 是 13.3M——显存和时间大致按参数量线性涨，16 GB 上到 192 都很宽松。

**预期的时间与显存**（估算，以你实测为准）：batch 256、32×32、bf16，显存占用在 4～6 GB 量级；一张 5070 Ti 上一个 epoch（5 万张图）大约一两分钟，200 个 epoch 几个小时能跑完。几十个 epoch 之后能看出轮廓和配色，训到两三百个 epoch、`base` 开到 160 以上，样本才会"一眼是 CIFAR"——这个规模的模型不要期待 FID 很低，它的价值在于你亲手把整条链路跑通了。

**判断训得对不对**：

- 前 5 个 epoch loss 应该明显下降，之后进入缓慢下降的长尾；
- 采样图第一眼看**颜色分布**对不对（全灰说明速度场没学到、全饱和说明没 clamp 或者归一化错了）；
- EMA 权重采出来的图比原始权重明显更干净——这是扩散/流匹配模型的常识，别忘了存 EMA。

### 往上走一步

- **提高分辨率**：不要直接在 256×256 的像素上训，显存和时间都会爆。用**潜空间扩散**：拿一个现成的 VAE 把 $256\times256\times3$ 压到 $32\times32\times4$，在潜空间里跑完全一样的流匹配，最后用 VAE 解码。一张 16 GB 的卡训潜空间模型是可行的。
- **条件生成**：在时间嵌入上加一个类别嵌入就是类别条件；再加上 **classifier-free guidance**（训练时以 10% 概率丢掉条件，采样时把有条件和无条件的速度按 $v = v_\varnothing + w\,(v_c - v_\varnothing)$ 外推），质量会有明显提升。
- **文本条件**：把一个冻结的文本编码器的输出做交叉注意力注入——这就是 Stable Diffusion 的结构。一张卡上从零训文生图不现实，但**在现成模型上做 LoRA 微调**完全可行，而且是最常见的实用场景。
- **少步采样**：流匹配的路径接近直线，所以少步采样天然好用。先试 Heun 法（每步两次网络调用但精度是二阶），再看**重流**（reflow）和蒸馏。

## 一张卡上的省显存与提速清单

| 手段 | 省显存 | 提速 | 代价 |
| --- | --- | --- | --- |
| bf16 autocast | 激活减半 | 2～4 倍 | 几乎没有（bf16 不需要 loss scaling） |
| `torch.compile` | 略省（融合减少中间张量） | 小模型上 1.2～1.8 倍 | 首次编译几分钟；动态形状会反复重编译 |
| FlashAttention（`sdpa`） | 注意力矩阵不落地，省 $O(S^2)$ | 长序列上明显 | 无 |
| 整层激活重计算 | 约 17 倍 | −30% | 多一次前向 |
| 梯度累积 | 线性省激活 | 无 | 无（有效 batch 不变） |
| `channels_last`（卷积网络） | 无 | 1.1～1.3 倍 | 无 |
| 8-bit 优化器 | 优化器状态 $8\Psi \to 2\Psi$ | 略快 | 极少数情况下影响收敛 |
| LoRA | 优化器状态几乎为零 | 更快 | 表达能力受限，适合微调不适合预训练 |
| 更小的 micro-batch | 线性 | 变慢（算力喂不饱） | 最后才用 |

顺序建议：**bf16 → compile → sdpa → 梯度累积 → 重计算 → 8-bit 优化器 → LoRA**。前三项几乎是白拿的，后面每一项都有代价。

## 排障

- **`no kernel image is available`**：PyTorch 的 wheel 里没有 `sm_120`。`torch.cuda.get_arch_list()` 确认，然后换 cu128 及更新的 wheel。
- **第一步就 OOM**：多半是 `torch.compile` 在编译时试了很大的形状，或者数据加载器 `pin_memory` + `num_workers` 太多。先关掉 compile 定位。
- **跑几百步之后才 OOM**：显存碎片。设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，或者固定序列长度（变长会让分配器反复切块）。
- **loss 变 NaN**：先确认不是 fp16（用 bf16）；再看 LR 和 warmup；再看有没有除零（注意力 mask 全为 $-\infty$ 的行）。
- **吞吐上不去、GPU 利用率忽高忽低**：数据加载跟不上。加 `num_workers`、开 `persistent_workers`、把数据预先打包成一个 `.bin` 用内存映射读（见[从零训练（一）](data.md)）。
- **每一步都在重编译**：形状在变。把最后一个不满的 batch 丢掉（`drop_last=True`），序列长度固定。

!!! interview "怎么讲清楚"
    讲"你自己训过模型吗"，一张消费级卡上的完整实践是很好的答案，关键是讲清楚**账**：16 GB 里模型状态占 $16\Psi$、激活占 `层 × 序列 × batch × 34d`，所以瓶颈在激活、重计算能省 17 倍；用实测算力算出 MFU，说明为什么是 40% 而不是 80%；以及配置里每一项（bf16 而不是 fp16、compile、fused AdamW、梯度累积）分别解决什么问题。图像那条线可以讲流匹配为什么比 DDPM 简单：训练目标是在噪声和数据的直线插值上回归两端之差，一行 MSE，采样是解 ODE，所以少步采样天然好用。

## 练习

1. 把 `budget.py` 改成支持 LoRA：只训练 rank 为 $r$ 的低秩增量，优化器状态只作用在这部分参数上。算一算 1.4B 模型、$r = 16$ 时，16 GB 够不够？

??? success "参考答案"
    LoRA 下基座权重只需要 bf16 的一份（$2\Psi$，不需要 fp32 副本、不需要梯度、不需要 m 和 v），1.4B 就是 2.6 GB。可训练参数大约是 $2 r d L \times 2$（每层两个投影各加一对 $d\times r$），$r=16$、$d=2048$、$L=24$ 时约 3M 参数，按 $16\Psi$ 算也只有 50 MB。加上激活和预留，总共 4 GB 左右——绰绰有余，甚至能把序列长度和 batch 都开大。这就是为什么"一张卡做微调"是完全现实的，而"一张卡做预训练"只能到几百 M。

2. `flow2d.py` 里把采样步数从 100 改成 10、5、2，观察落在模态附近的比例怎么变。再把欧拉法换成 Heun 法（两次网络调用的二阶方法），同样步数下质量好多少？

??? success "参考答案"
    步数降到 10 通常还能保持大部分样本落在模态上（流匹配的路径接近直线，所以对步数不敏感），降到 2～5 时会明显偏离。Heun 法每步先用欧拉预测一步、再用两端速度的平均修正，误差从一阶降到二阶，同样的网络调用预算下质量更好——这也是为什么实际系统里很少用朴素欧拉法。

3. 用本章的显存公式解释：同样是 16 GB，为什么"350M 模型 + 序列 1024"和"124M 模型 + 序列 4096"的激活占用差不多？这对选择模型规模有什么启示？

??? success "参考答案"
    激活正比于 `层数 × 维度 × 序列长度 × batch`。350M（24 层 × 1024）比 124M（12 层 × 768）的 `层 × 维` 大约 2.7 倍，而序列 4096 比 1024 大 4 倍——两者相抵，量级接近。启示是：**序列长度和模型规模在显存上是可以互换的**，做长上下文实验时应该主动把模型缩小，否则会误以为"卡不够"，其实是把预算花在了错误的维度上。

## 小结

- [x] 16 GB 上模型状态（$16\Psi$）不是瓶颈，**激活**才是；整层重计算能把激活压 17 倍，代价约 +30% 时间。
- [x] 一张消费级卡的甜点区是 **100M～350M 从零训练**；再大就该换成 LoRA 微调。
- [x] 所有估算都要建立在**实测的带宽和算力**上，不要用标称值；MFU 40% 左右算正常。
- [x] 图像生成用**流匹配**最容易上手：在噪声与数据的直线插值上回归两端之差，一行 MSE；采样是解 ODE，少步就能出图。
- [x] Blackwell（`sm_120`）要 PyTorch 2.7 以上的 cu128 wheel，先用内置的 `sdpa` 而不是第三方 FlashAttention 轮子。
