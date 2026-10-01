# 少步生成：蒸馏把 50 步变成 4 步

<p class="lead">成本公式里最大的一项是步数。采样器把 DDPM 的 1000 步压到了 20～50 步，再往下就不是求解器的事了——是模型本身要换：用蒸馏让一个学生模型学会"一步走完老师几十步的路"。这一章把主流的少步方法按原理分成几类，用一个二维玩具在 CPU 上把"一步蒸馏"亲手做一遍，看它为什么能行、代价是什么；再把 SDXL-Turbo、Lightning、LCM、Hyper-SD、FLUX schnell 这些模型放回成本公式里，算清它们各省了多少、换走了什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么采样器只能把步数压到 20 左右，再往下要靠蒸馏？
    2. 渐进蒸馏、一致性模型、对抗蒸馏、分布匹配蒸馏各在学什么？
    3. 引导蒸馏（guidance distillation）省的是哪一项？FLUX.1-dev 的 `guidance` 输入是怎么来的？
    4. 少步模型的代价是什么？什么场景不该用？
    5. LCM-LoRA 为什么能"插"到任何同底座的模型上？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 求解器的精度受速度场的弯曲程度限制，真实模型的轨迹在高噪声端弯得厉害，步长大了误差就超过可接受范围。蒸馏改变的是速度场本身：让学生在少数几个点上直接预测"终点"或者"更直的轨迹"，不再依赖求解器沿曲线一步步走。
    2. 渐进蒸馏：学生一步模仿老师两步，反复减半；一致性模型：学会把轨迹上任意一点直接映射到终点（任意 $t$ 一步到 $x_0$），多步只是精修；对抗蒸馏：用判别器逼学生的一步输出"像真图"，不再追求和老师逐点一致；分布匹配蒸馏（DMD）：让学生的输出分布和老师的分布在分数函数意义下一致，用老师当"评分器"。
    3. 省的是 CFG 的那一倍前向：训练时把引导后的预测 $v_\varnothing + w(v_c - v_\varnothing)$ 当目标，并把 $w$ 作为额外输入教给学生，推理时一次前向就等价于两路外推。FLUX.1-dev 就是这样蒸馏出来的，所以它接受一个标量 `guidance`。
    4. 多样性下降（尤其对抗蒸馏，输出向少数"好看"的模式塌缩）、质量上限略低于老师、几乎不能换采样器和步数（LCM 必须配 LCM 调度器，Turbo 4 步以上反而变差）、对 LoRA / ControlNet 等插件的兼容性要重新验证。追求最高质量、需要大量多样候选、要做可控编辑时不该用。
    5. 它学的是"把老师的多步轨迹压成少步"这件事对权重的**增量**，而这个增量在同一底座的不同微调版本之间大体通用；把增量当 LoRA 加到任何 SD 1.5 / SDXL 微调模型上，就能 4 步出图。

## 为什么求解器到此为止

[采样器一章](../basics/schedulers.md)的玩具里，误差随步数单调下降，但真实模型的速度场在高噪声端弯得厉害，4 步欧拉的误差已经不可接受。蒸馏不去改进求解器，而是**改模型**：在少数几个采样点上，让学生直接给出"终点在哪"。

先把一个二维的流匹配老师训出来，再蒸馏一个一步学生，看它们各自在不同步数下的表现。这里的"质量"用一个可计算的指标：生成样本落在某个模态附近的比例。

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)
K, R = 8, 2.0                                                    # 目标分布：圆上 8 个高斯

def sample_data(n):
    idx = torch.randint(0, K, (n,))
    ang = idx.float() * (2 * math.pi / K)
    return torch.stack([R * ang.cos(), R * ang.sin()], dim=1) + 0.08 * torch.randn(n, 2)

class Velocity(nn.Module):
    def __init__(self, h=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))

teacher = Velocity()
opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
for step in range(3000):                                         # 老师：标准的流匹配训练
    x1, x0 = sample_data(256), torch.randn(256, 2)
    t = torch.rand(256, 1)
    loss = ((teacher((1 - t) * x0 + t * x1, t) - (x1 - x0)) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def ode(model, x, steps):                                        # 从噪声 (t=0) 走到数据 (t=1)，欧拉法
    for i in range(steps):
        t = torch.full((x.shape[0], 1), i / steps)
        x = x + model(x, t) / steps
    return x

centers = torch.stack([R * torch.cos(torch.arange(K) * 2 * math.pi / K), R * torch.sin(torch.arange(K) * 2 * math.pi / K)], 1)
def hit_rate(x):
    return (torch.cdist(x, centers).min(1).values < 0.3).float().mean().item()

torch.manual_seed(1)
noise = torch.randn(4000, 2)
print("老师在不同步数下的命中率（样本落在模态 0.3 半径内的比例）：")
for steps in (1, 2, 4, 8, 50):
    print(f"  {steps:>2} 步  {hit_rate(ode(teacher, noise, steps)):.0%}")
```

```text title="输出"
老师在不同步数下的命中率（样本落在模态 0.3 半径内的比例）：
   1 步  0%
   2 步  11%
   4 步  56%
   8 步  69%
  50 步  79%
```

老师 1 步、2 步几乎什么都画不出来，要几十步才像样——这就是求解器的极限。

## 一步蒸馏：学生学老师的终点

最朴素的蒸馏：对每个噪声 $x_0$，老师用 50 步算出终点 $x_1^{\text{teacher}}$，学生从同一个 $x_0$ 出发只走一步，要求落在同一个地方。学生的网络结构和老师一样，从老师的权重开始训：

```python
import copy

student = copy.deepcopy(teacher)
opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
t0 = torch.zeros(256, 1)
for step in range(1500):
    x0 = torch.randn(256, 2)
    target = ode(teacher, x0, 50)                                # 老师走 50 步的终点
    pred = x0 + student(x0, t0)                                  # 学生一步：x0 + v·1
    loss = ((pred - target) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

print("蒸馏后：")
for name, model, steps in [("老师 1 步", teacher, 1), ("老师 4 步", teacher, 4), ("老师 50 步", teacher, 50), ("学生 1 步", student, 1)]:
    print(f"  {name:<9} 命中率 {hit_rate(ode(model, noise, steps)):.0%}")
mode_counts = torch.bincount(torch.cdist(ode(student, noise, 1), centers).argmin(1), minlength=K)
print(f"学生 1 步生成落在各模态的样本数：{mode_counts.tolist()}（理想是每个 500）")
```

```text title="输出"
蒸馏后：
  老师 1 步    命中率 0%
  老师 4 步    命中率 56%
  老师 50 步   命中率 79%
  学生 1 步    命中率 63%
学生 1 步生成落在各模态的样本数：[482, 498, 504, 547, 502, 477, 513, 477]（理想是每个 500）
```

学生用**一次前向**从老师 1 步的 0% 跳到了老师 8 步左右的水平，NFE 从 50 降到 1；和老师 50 步还有一段差距，这段差距就是后面各种方法要补的。两个要注意的细节正是真实蒸馏方法的分野：

- 这个学生学的是"从噪声直接到终点"的映射，所以它**只会走一步**，多走几步反而没有定义——一致性模型（Consistency Models）解决的就是"任意 $t$ 都能一步到终点、多步可以精修"；
- 目标是老师的逐点输出，所以学生的多样性不会超过老师，而且回归均值的倾向会让输出偏"平"——对抗蒸馏和分布匹配蒸馏用判别器 / 分数函数代替逐点回归，就是为了解决这个。

## 主流方法的谱系

| 方法 | 学什么 | 代表模型 | 步数 | 特点 |
| --- | --- | --- | --- | --- |
| 渐进蒸馏（Progressive Distillation） | 学生一步 = 老师两步，反复减半 | 早期 SD 蒸馏版 | 4～8 | 每轮重训，流程长 |
| 一致性模型 / LCM | 轨迹上任意点 → 终点的映射（自一致性） | LCM、LCM-LoRA | 4～8（1 步也能看） | 可插拔 LoRA；需要专用调度器 |
| 对抗蒸馏（ADD / LADD） | 判别器逼一步输出像真图，老师只做分数引导 | SDXL-Turbo、SD3-Turbo、FLUX.1-schnell | 1～4 | 质量高、多样性降；步数多了反而差 |
| 渐进 + 对抗 | 逐级减半，每级加对抗损失 | SDXL-Lightning | 1 / 2 / 4 / 8 | 每个步数一套权重 |
| 分布匹配蒸馏（DMD / DMD2） | 学生输出分布与老师分布的分数差 | DMD2、各家 "4-step" 版本 | 1～4 | 不需要成对数据；质量接近老师 |
| 轨迹分段一致性 | 把轨迹切段，段内一致性 + 对抗 | Hyper-SD | 1～8 | 同一套权重多步可用 |
| 重流（Reflow） | 用老师生成的 (噪声, 图) 对重训，拉直轨迹 | InstaFlow、PeRFlow | 1～4 | 流匹配天然适合 |
| 引导蒸馏 | 把 CFG 的外推烘进模型，引导强度当输入 | FLUX.1-dev、各家 "guidance-distilled" | 不变 | 省掉 CFG 那一倍前向 |

**引导蒸馏**单独说一句：它不减步数，减的是每步的两路前向——FLUX.1-dev 接受 `guidance` 标量就是因为它。对视频模型这一项同样值 2 倍，所以 Wan 2.2 这类新模型开始自带引导蒸馏版本。

## 放回成本公式

把蒸馏后的模型放回"步数 × CFG × 单步"：

```python
CASES = [
    # 名字,                  步数, CFG 倍数, 单步相对老师, 老师步数, 老师 CFG
    ("SDXL base",               30, 2, 1.0, 30, 2),
    ("SDXL + LCM-LoRA",          6, 1, 1.0, 30, 2),
    ("SDXL-Turbo",               4, 1, 1.0, 30, 2),
    ("SDXL-Lightning 4 步",      4, 1, 1.0, 30, 2),
    ("FLUX.1-dev（引导已蒸馏）",  28, 1, 1.0, 28, 2),      # 老师按"未蒸馏 + CFG"算
    ("FLUX.1-schnell",            4, 1, 1.0, 28, 2),
    ("Wan 2.1 14B",              50, 2, 1.0, 50, 2),
    ("Wan 14B + 蒸馏 4 步",       4, 1, 1.0, 50, 2),
]
print(f"{'模型':<24} {'前向次数':>8} {'相对老师':>8}  说明")
for name, steps, cfg, rel, t_steps, t_cfg in CASES:
    nfe, t_nfe = steps * cfg * rel, t_steps * t_cfg
    note = "" if nfe < t_nfe else "基准"
    print(f"{name:<24} {nfe:>8.0f} {t_nfe / nfe:>7.1f}×  {note}")
```

```text title="输出"
模型                           前向次数     相对老师  说明
SDXL base                      60     1.0×  基准
SDXL + LCM-LoRA                 6    10.0×  
SDXL-Turbo                      4    15.0×  
SDXL-Lightning 4 步              4    15.0×  
FLUX.1-dev（引导已蒸馏）              28     2.0×  
FLUX.1-schnell                  4    14.0×  
Wan 2.1 14B                   100     1.0×  基准
Wan 14B + 蒸馏 4 步                4    25.0×  
```

**15～25 倍**——没有任何 kernel 级优化能给出这个数量级。所以交互式产品（实时绘画、预览、批量草稿）几乎都用蒸馏模型；代价是多样性和上限质量，所以"出成品"的流程往往是蒸馏模型快速出草稿、挑中的再用完整模型精修。

## 代价与不该用的场景

| 代价 | 表现 | 影响 |
| --- | --- | --- |
| 多样性下降 | 同一提示词不同种子的图很像 | 需要大量候选时不划算 |
| 质量上限 | 细节、文字、手部略差于老师 | 成品流程里只当草稿 |
| 不能自由换采样器 / 步数 | LCM 必须配 LCM 调度器；Turbo 超过 4 步变差 | 服务里按模型固定配置 |
| 插件兼容 | LoRA、ControlNet、IP-Adapter 要重新验证 | 多了一轮测试 |
| 引导不可调 | 引导蒸馏的模型 CFG scale 效果弱甚至失效 | 用户习惯的参数失灵 |
| 可控编辑受限 | 反演（inversion）、局部重绘依赖多步轨迹 | 编辑类功能用完整模型 |

还有一个和推理系统直接相关的点：**少步模型让 CPU 开销和固定成本的占比变大**。4 步 SDXL-Turbo 一张图去噪只要 100 多毫秒，文本编码、VAE 解码、Python 调度、数据搬运这些"零头"就成了主角——这时 CUDA Graph、文本编码缓存、分块 VAE 的收益反而最大（见[算子加速](kernels.md)和生成服务的调度）。

!!! interview "面试怎么答"
    被问"扩散模型怎么做到 4 步出图"，先说边界：求解器到 20 步是极限，再少要改模型。再按原理分类：渐进蒸馏是一步学两步；一致性模型学"任意点到终点"的映射；对抗蒸馏用判别器换多样性要质量；DMD 做分布匹配；引导蒸馏单独把 CFG 那一倍省掉。给数字：SDXL 60 次前向到 Turbo 的 4 次，15 倍。最后说代价和系统含义：多样性和上限下降、调度器和步数被锁死，而少步之后固定开销占比变大，CUDA Graph 和编码缓存这些手段的收益反而最大。

## 练习

1. 把一步学生改成"两步学生"：训练时让学生从 $x_0$ 走到 $t=0.5$ 处老师的轨迹点，再从那里走到终点（两个目标）。两步学生的命中率比一步高吗？这对应谱系里的哪一类方法？

??? success "参考答案"
    通常会高一些——第二步在一个更"干净"的起点上修正第一步的误差。它对应一致性 / 轨迹分段的思路：把轨迹切成段，每段学一个"段终点"的映射，多步就是逐段精修（Hyper-SD 的做法）。

2. 一步学生的损失是逐点 MSE。把 4000 个噪声样本经学生生成后的点和老师 50 步的结果比较：样本到最近模态的平均距离、落在各模态的分布，哪个指标更能暴露"回归均值"的问题？

??? success "参考答案"
    平均距离——MSE 训练会让学生在"不确定该去哪个模态"的噪声点上输出几个模态的平均位置，这些点离任何模态都远；模态分布可能仍然均匀。真实图像里这就是"少步模型细节发糊"的来源，对抗蒸馏和 DMD 用分布层面的损失替代逐点 MSE 正是为了消除它。

3. 某服务用 SDXL-Lightning 4 步模型，用户反馈"同一个提示词出 8 张图都差不多"。不换模型的话有什么办法？换模型的话怎么权衡？

??? success "参考答案"
    不换模型：加大初始噪声的差异没用（多样性是模型塌缩的，不是种子的问题），可以在提示词上做扰动（同义改写、随机加风格词）、或者前 1～2 步用完整模型走、后面交给少步模型（混合流程）。换模型：LCM 或 DMD2 的多样性通常比对抗蒸馏的好，代价是多走几步或质量略降；或者"少步出 8 张草稿 + 完整模型精修用户选中的那张"。

## 小结

- [x] 求解器到 20 步是极限；再少要蒸馏——改模型本身，让它在少数几个点上直接给出终点。
- [x] 一步蒸馏在玩具上就能看到：学生一次前向接近老师 8 步的水平，但只会走一步、多样性不超过老师——一致性模型、对抗蒸馏、DMD 分别解决这两个问题。
- [x] 引导蒸馏单独省掉 CFG 那一倍前向；蒸馏整体把前向次数降 15～25 倍，是所有手段里最大的一项。
- [x] 代价：多样性与上限下降、调度器和步数被锁死、插件要重验；少步之后固定开销占比变大，CUDA Graph 和编码缓存的收益反而最大。
