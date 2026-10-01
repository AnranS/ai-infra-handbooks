# 采样器与调度器

<p class="lead">同一个去噪网络，换一个调度器，步数可以从 50 变成 20，质量还可能更好；时间步的分布排得不对，高分辨率出图会糊成一片。调度器决定了"在哪些噪声水平上调用网络、每次怎么更新潜变量"，是推理侧最便宜也最容易被忽视的旋钮。这一章把主流调度器的差别、时间步偏移（shift）、引导强度的调法和可复现性讲清楚，所有数字都从 diffusers 的调度器对象里直接读出来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DDIM、DPM-Solver++、UniPC 和欧拉法各是什么关系？为什么同样 20 步质量不一样？
    2. SD3 / FLUX 的 "shift" 是什么？分辨率越高为什么要把时间步往噪声端挪？
    3. 一阶和二阶求解器的 NFE 怎么算？读 benchmark 时该比什么？
    4. 固定了种子，两次生成为什么还会不一样？要做到逐像素可复现需要哪些条件？
    5. 引导强度（CFG scale）过大会怎样？有哪些不改模型就能缓解的办法？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 它们都是求解去噪 ODE 的数值方法。欧拉和 DDIM 是一阶（每步只用当前的预测），DPM-Solver++ 和 UniPC 利用扩散 ODE 的半线性结构做二阶、三阶近似（复用前几步的预测，不多算网络），同样步数下轨迹误差小一个数量级，所以 20 步能顶一阶方法 50 步。
    2. 流匹配的时间步经过一个变换 $t' = \frac{s\,t}{1 + (s-1)\,t}$，$s > 1$ 把更多步分配到高噪声端。分辨率越高，同样噪声水平下图像的信噪比越高（相邻像素相关），"真正像噪声"的区间往高噪声端缩，所以要把步数往那边挪；FLUX 按 token 数动态算 $s$。
    3. NFE = 调用去噪网络的次数。一阶方法 NFE = 步数；Heun 这类每步两次预测的方法 NFE = 2 × 步数；DPM-Solver++ 多步版复用历史，NFE = 步数。比质量要在同样的 NFE 下比。
    4. 种子只固定初始噪声；去噪网络里的矩阵乘、注意力的归约顺序随 kernel、batch 组成、硬件而变，几十步后放大成可见差异。逐像素可复现要固定：硬件、框架和 kernel 版本、batch 组成（同一请求单独跑和拼在 batch 里跑结果不同）、确定性算法开关。
    5. 过饱和、对比度爆炸、细节崩坏。缓解：guidance rescale、动态或分段的引导强度（前几步大、后几步小，或只在中间步用引导）、更低的默认值配合更好的采样器。

## 时间步怎么排：调度器给出的序列

调度器的第一件事是把训练用的 1000 个时间步压缩成推理用的 N 个。不同调度器排法不同，直接读出来：

```python
import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler,
                       EulerDiscreteScheduler, FlowMatchEulerDiscreteScheduler)

common = dict(num_train_timesteps=1000, beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012)
N = 8
for name, sch in [("DDIM", DDIMScheduler(**common)), ("DPM-Solver++ (2M)", DPMSolverMultistepScheduler(**common)),
                  ("UniPC", UniPCMultistepScheduler(**common)), ("Euler", EulerDiscreteScheduler(**common))]:
    sch.set_timesteps(N)
    print(f"{name:<18} {[int(t) for t in sch.timesteps]}")
flow = FlowMatchEulerDiscreteScheduler(shift=1.0)
flow.set_timesteps(N)
print(f"{'流匹配 (shift=1)':<18} σ = {[round(s, 3) for s in flow.sigmas.tolist()]}")
```

```text title="输出"
DDIM               [875, 750, 625, 500, 375, 250, 125, 0]
DPM-Solver++ (2M)  [999, 874, 749, 624, 500, 375, 250, 125]
UniPC              [999, 874, 749, 624, 500, 375, 250, 125]
Euler              [999, 856, 713, 570, 428, 285, 142, 0]
流匹配 (shift=1)      σ = [1.0, 0.857, 0.715, 0.572, 0.429, 0.286, 0.144, 0.001, 0.0]
```

DDPM 一族（前四个）的时间步在 0～999 之间大致均匀；流匹配直接给出噪声占比 $\sigma$，从 1 到 0 均匀下降。真正的差别不在"在哪些点调用网络"，而在**每一步怎么用网络的预测更新潜变量**。

## 求解器的阶数：同样的 NFE 谁更准

把[第一章](diffusion-inference.md)的玩具 ODE 拿回来，再加一个 DPM-Solver 风格的多步法：它不额外调用网络，而是用上一步的预测做线性外推（类似 Adams-Bashforth）。三种方法在同样的 NFE 下比误差：

```python
import math

def v(x, sigma):                                   # 速度场 v = -2σx，从 σ=1 到 0 的精确解是 x(0) = x(1) · e
    return -2.0 * sigma * x

def euler(nfe):
    x, s, d = 1.0, 1.0, 1.0 / nfe
    for _ in range(nfe):
        x, s = x - d * v(x, s), s - d
    return x

def heun(nfe):                                     # 每步两次调用，所以只能走 nfe/2 步
    steps = nfe // 2
    x, s, d = 1.0, 1.0, 1.0 / steps
    for _ in range(steps):
        v1 = v(x, s); v2 = v(x - d * v1, s - d)
        x, s = x - d * (v1 + v2) / 2, s - d
    return x

def multistep2(nfe):                               # 二阶多步：复用上一步的预测，每步仍只调用一次
    x, s, d = 1.0, 1.0, 1.0 / nfe
    prev = None
    for _ in range(nfe):
        cur = v(x, s)
        slope = cur if prev is None else 1.5 * cur - 0.5 * prev   # 线性外推到区间中点
        x, s, prev = x - d * slope, s - d, cur
    return x

print(f"{'NFE':>4} {'欧拉 (一阶)':>12} {'Heun (二阶,每步2次)':>18} {'多步二阶 (每步1次)':>18}")
for nfe in (8, 16, 32, 64):
    print(f"{nfe:>4} {abs(euler(nfe) - math.e):>12.5f} {abs(heun(nfe) - math.e):>18.5f} {abs(multistep2(nfe) - math.e):>18.5f}")
```

```text title="输出"
 NFE      欧拉 (一阶)     Heun (二阶,每步2次)        多步二阶 (每步1次)
   8      0.10188            0.05740            0.02761
  16      0.05374            0.01443            0.00698
  32      0.02759            0.00359            0.00176
  64      0.01397            0.00089            0.00044
```

**多步二阶法用一阶的代价拿到二阶的精度**——这就是 DPM-Solver++ 2M、UniPC 成为默认选择的原因：每步仍只调用一次网络，却把前一两步的预测利用起来。实践中的经验：

| 调度器 | 阶数 | NFE / 步 | 推荐步数 | 适合 |
| --- | --- | --- | --- | --- |
| DDIM | 1 | 1 | 50 | 老模型兼容、需要确定性反演（图像编辑）时 |
| Euler / Euler a | 1 | 1 | 25～30 | SDXL 默认之一；"a"（ancestral）每步加噪声，不确定性但更多样 |
| DPM-Solver++ 2M | 2 | 1 | 20～25 | SD 1.5 / SDXL 的性价比之选 |
| DPM-Solver++ SDE | 2 | 1 | 25～30 | 加随机性，细节更丰富，步数少时不如 2M |
| UniPC | 2～3 | 1 | 15～20 | 步数最少的通用选择 |
| 流匹配 Euler | 1 | 1 | 20～50 | SD3 / FLUX / Wan 的默认；轨迹接近直线所以一阶够用 |
| 蒸馏模型专用（LCM、Lightning） | — | 1 | 1～8 | 只能配对应模型用，见少步生成 |

## 时间步偏移：高分辨率为什么要往噪声端挪

流匹配模型的调度器多了一个参数 `shift`。它把均匀的时间步做一个单调变换：

$$
t' = \frac{s \cdot t}{1 + (s - 1)\, t}
$$

$s > 1$ 时，更多的步被分配到高噪声端（$\sigma$ 接近 1 的那一段）。看它对 8 步的分配有什么影响：

```python
for shift in (1.0, 3.0, 6.0):
    sch = FlowMatchEulerDiscreteScheduler(shift=shift)
    sch.set_timesteps(8)
    sig = sch.sigmas.tolist()[:-1]
    high = sum(1 for s in sig if s > 0.5)
    print(f"shift={shift:<4} σ = {[round(s, 2) for s in sig]}   σ>0.5 的步数 {high}/8")
```

```text title="输出"
shift=1.0  σ = [1.0, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14, 0.0]   σ>0.5 的步数 4/8
shift=3.0  σ = [1.0, 0.95, 0.88, 0.8, 0.69, 0.55, 0.34, 0.01]   σ>0.5 的步数 6/8
shift=6.0  σ = [1.0, 0.97, 0.94, 0.89, 0.82, 0.71, 0.51, 0.03]   σ>0.5 的步数 7/8
```

为什么分辨率越高越需要它：同样的噪声占比 $\sigma$，高分辨率图像的相邻像素高度相关，在潜空间里"信号"的能量更集中，等价的信噪比比低分辨率高——于是在高分辨率下 $\sigma = 0.5$ 已经"相当干净"，真正需要仔细走的是 $\sigma$ 接近 1 的那段。SD3 的论文给出了经验：$s = \sqrt{m / n}$，$m$ 是目标 token 数、$n$ 是基准（256² 的 token 数）。FLUX 更进一步按 token 数线性插值出 $\mu = \text{shift}$，再 $s = e^{\mu}$：

```python
def flux_shift(image_seq_len, base_len=256, max_len=4096, base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_len - base_len)
    mu = m * image_seq_len + (base_shift - m * base_len)
    return math.exp(mu)

for res in (512, 1024, 2048):
    tokens = (res // 8 // 2) ** 2
    print(f"FLUX {res}²：{tokens:>5} 个 token → shift = {flux_shift(tokens):.2f}")
```

```text title="输出"
FLUX 512²： 1024 个 token → shift = 1.88
FLUX 1024²： 4096 个 token → shift = 3.16
FLUX 2048²：16384 个 token → shift = 25.28
```

实践里"FLUX 出 2K 图发糊"的常见原因之一，就是调度器没按分辨率更新 shift——token 数变了四倍，时间步分配却还是 1K 的。

## 引导强度：数值与调法

CFG scale 是用户最常碰的参数，也最容易用过头。它对推理系统有三层影响：

1. **成本**：$w \ne 1$ 就要算两路前向（见[第一章](diffusion-inference.md)）；
2. **质量**：过大过饱和、细节崩坏，过小不听话；每个模型有自己的甜点（SD 1.5 约 7、SDXL 约 5、SD3 约 4、FLUX dev 的 `guidance` 约 3.5）；
3. **可以按步变化**：早期步决定构图、需要强引导，后期步只在填细节、引导可以减弱甚至关掉——关掉的那些步只算一路前向，直接省算力。

```python
def guidance_schedule(steps, w_max=7.5, w_min=1.0, mode="constant"):
    if mode == "constant":
        return [w_max] * steps
    if mode == "linear_decay":                      # 前期强、后期弱
        return [w_max - (w_max - w_min) * i / (steps - 1) for i in range(steps)]
    if mode == "interval":                          # 只在中间 60% 的步用引导（两端 w=1：只算一路）
        lo, hi = int(steps * 0.1), int(steps * 0.7)
        return [w_max if lo <= i < hi else 1.0 for i in range(steps)]

steps = 20
for mode in ("constant", "linear_decay", "interval"):
    ws = guidance_schedule(steps, mode=mode)
    nfe = sum(2 if w != 1.0 else 1 for w in ws)
    print(f"{mode:<13} 平均 w {sum(ws) / steps:>4.1f}   去噪网络前向次数 {nfe:>3}（最多 {2 * steps}）")
```

```text title="输出"
constant      平均 w  7.5   去噪网络前向次数  40（最多 40）
linear_decay  平均 w  4.2   去噪网络前向次数  39（最多 40）
interval      平均 w  4.9   去噪网络前向次数  32（最多 40）
```

"interval" 这种只在一段时间步用引导的做法，在不改模型的前提下省了 40% 的前向——它的理论依据是引导在中间噪声水平最有用（太早时图像还没有结构可引导，太晚时只剩纹理）。guidance rescale 则是另一个方向：引导后把预测的标准差拉回有条件预测的水平，专治过饱和。

## 可复现：种子之外还要固定什么

"固定种子就能复现"在扩散模型里只对初始噪声成立。影响最终像素的还有：

| 因素 | 为什么会变 | 怎么固定 |
| --- | --- | --- |
| 初始噪声 | 由 `Generator` 和种子决定；不同设备上同一种子的随机序列也不同 | 在 CPU 上生成噪声再搬到 GPU（diffusers 的常见做法） |
| batch 组成 | 单独跑和拼进 batch 跑，矩阵乘的切分方式不同，浮点归约顺序不同 | 服务里对"需要可复现"的请求单独跑或固定 batch 形状 |
| 注意力 kernel | FlashAttention / SDPA 的不同后端、不同分块，归约顺序不同 | 固定后端；需要时用确定性实现 |
| 框架与硬件 | cuBLAS 的算法选择、GPU 架构 | 记录版本；跨硬件不承诺逐像素一致 |
| 调度器状态 | 多步调度器（DPM-Solver++ 2M）复用历史预测，不同的历史给不同结果 | 每个请求用独立的调度器实例，别在请求间共享 |

最后一条是服务层最容易踩的坑：**多步调度器有状态**。用一个全局的调度器对象给并发请求用，历史预测会串，结果既不可复现也不正确。推理引擎里每个请求要有自己的调度器实例（或者把状态显式放进请求对象，见生成服务的调度）。

```python
sch = DPMSolverMultistepScheduler(**common)
sch.set_timesteps(4)
print(f"DPM-Solver++ 2M 的内部状态：model_outputs 缓存 {sch.config.solver_order} 项，lower_order_nums 计数器——这些都随请求走")
print(f"调度器实例是有状态的：{hasattr(sch, 'model_outputs')}；把它在并发请求间共享，预测历史就会串")
```

```text title="输出"
DPM-Solver++ 2M 的内部状态：model_outputs 缓存 2 项，lower_order_nums 计数器——这些都随请求走
调度器实例是有状态的：True；把它在并发请求间共享，预测历史就会串
```

!!! interview "面试怎么答"
    被问"采样器怎么选、步数怎么定"，先说框架：它们都是在解同一个 ODE，区别是求解器的阶数和每步调用网络的次数，所以比较要按 NFE；多步高阶法（DPM-Solver++ 2M、UniPC）用一阶的代价拿到二阶精度，是 SD 一族的默认；流匹配模型轨迹接近直线，一阶欧拉就够，但要按分辨率调 shift。再说系统视角：CFG 决定每步是一路还是两路前向，分段引导能在不改模型的前提下省 40%；多步调度器有状态，服务里不能跨请求共享；可复现要固定的不只是种子，还有 batch 组成和 kernel。

## 练习

1. 把 `multistep2` 改成三阶（用前两步的预测做二次外推），同样的 NFE 下误差再降多少？步数很少（NFE=4）时它还占优吗？

??? success "参考答案"
    三阶外推的系数是 $\frac{23}{12}c_0 - \frac{16}{12}c_1 + \frac{5}{12}c_2$（Adams-Bashforth 3），NFE=32 时误差再降一个量级。但 NFE 很少时高阶外推用的历史点间隔太大，外推反而不准，可能比二阶差——这正是 DPM-Solver++ 在步数极少时退回低阶、UniPC 在最后几步降阶的原因。

2. SD3 给出的经验 $s = \sqrt{m/n}$，以 256² 为基准：算 512²、1024²、2048² 的 shift，和 FLUX 的公式差多少？

??? success "参考答案"
    256² 的 token 数 n = 256（f8、patch 2），1024² 是 4096，$s = \sqrt{16} = 4$；512² 是 $s = 2$，2048² 是 $s = 8$。FLUX 的公式在 1024² 给 $e^{1.15} \approx 3.16$，和 SD3 同一量级；但它是对 $\mu$ 做**线性外推**，2048² 的 16384 个 token 会推到 $\mu \approx 3.2$、$s \approx 25$——明显过头了（上面代码的输出就是这个数）。所以用 FLUX 出 2K 以上的图时，实践中常把 $\mu$ 封顶在 `max_shift` 附近，或者改用 SD3 式的平方根规则。两家的共识只有一条：分辨率翻四倍，shift 大致翻一倍。

3. 服务里有一个请求要求"和昨天的结果逐像素一致"。列出你需要固定的全部条件，并说明哪一条最容易被忽略。

??? success "参考答案"
    种子与噪声生成的设备、模型权重版本、调度器类型与步数、CFG 与引导日程、分辨率、框架和 kernel 版本、硬件型号、batch 组成（单独跑还是和别的请求拼在一起）、注意力后端、确定性算法开关。最容易漏的是 batch 组成——同一请求在不同的并发环境下会被拼进不同形状的 batch，结果就不一样；需要可复现的请求要单独跑。

## 小结

- [x] 调度器决定在哪些噪声水平调用网络、每步怎么更新；DDPM 族的时间步均匀分布，流匹配直接给噪声占比 $\sigma$。
- [x] 比较采样器要按 NFE：多步高阶法（DPM-Solver++ 2M、UniPC）每步一次调用却有二阶精度，SD 族 20 步够用；流匹配模型轨迹近直线，欧拉 20～50 步。
- [x] 高分辨率要把时间步往噪声端挪（shift）：SD3 按 $\sqrt{m/n}$，FLUX 按 token 数算 $\mu$；忘了更新 shift 是高分辨率发糊的常见原因。
- [x] CFG 决定每步一路还是两路前向，分段引导能省四成算力；多步调度器有状态，不能跨请求共享；可复现要固定种子之外的 batch 组成、kernel 和硬件。
