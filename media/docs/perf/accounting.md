# 推理的算账：FLOP、带宽与一步要多久

<p class="lead">优化之前先算账。LLM 推理的账本（见[性能与服务中的数学](math://performance-math/)）在这里要重算一遍：扩散模型没有 KV Cache、每一步都是完整的前向，batch 的含义也不一样。这一章给出 DiT 一步的 FLOP 公式，算出注意力什么时候压过线性层、一步是算力受限还是带宽受限、一张卡上一步要多久，并把这些数字和主流模型对上。有了这张账，后面每种加速手段省在哪一项、能省多少，都能在动手前先估出来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. DiT 一层的 FLOP 由哪两项组成？各自和 token 数 $N$、隐藏维 $d$ 是什么关系？
    2. token 数到多少时注意力的计算量超过线性层？FLUX 在 1024² 和视频模型分别在哪一侧？
    3. 扩散模型的一步是算力受限还是带宽受限？和 LLM 的 decode 为什么不同？
    4. 怎么从 FLOP 估一步的时间？MFU 大概多少算正常？
    5. 把 batch 从 1 加到 4，吞吐能涨几倍？延迟呢？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 线性层（QKV、输出投影、MLP）约 $24 N d^2$，和 $N$ 成正比；注意力的 $QK^\top$ 和 $PV$ 约 $4 N^2 d$，和 $N^2$ 成正比。
    2. 令两者相等得 $N = 6d$。FLUX 的 $d = 3072$，临界点约 18k token：1024² 的 4608 个 token 在线性层主导的一侧，2048² 接近临界，视频模型（几万到十几万 token）注意力占到七八成以上。
    3. 算力受限。一步要对全部 token 做完整前向，每读一次权重做的 FLOP 是 $2N$ 量级（$N$ 几千到几万），算术强度远在屋顶线的拐点之上；LLM 的 decode 每读一次权重只算一个 token，是带宽受限的。
    4. $t = \text{FLOP} / (\text{MFU} \times \text{峰值算力})$。扩散模型的 MFU 通常 30%～50%（注意力和线性层占比高，但归一化、激活函数、调制这些访存算子不少）；UNet 因为卷积和各种小算子更低一些。
    5. 吞吐几乎不涨——一步本来就是算力受限的，batch 只是把更多 token 一起算，总 FLOP 线性增加；延迟接近线性变长。这和 LLM decode 里"batch 越大越划算"相反，是扩散服务里调度策略不同的根源。

## 一步的 FLOP

DiT 一层做的事和 LLM 的 Transformer 层一模一样：QKV 投影、注意力、输出投影、MLP。设 token 数 $N$、隐藏维 $d$、MLP 放大 4 倍，一层的 FLOP（乘加算 2）是

$$
\text{FLOP}_\text{layer} \approx \underbrace{24\, N d^2}_{\text{线性层}} + \underbrace{4\, N^2 d}_{\text{注意力}}
$$

两项相等的临界点是 $N = 6d$。写成函数，把主流模型代进去：

```python
MODELS = {
    # 名字: (层数, 隐藏维, 图像 token 数, 文本 token 数, 每步 CFG 倍数)
    "SD3-medium 1024²":        (24, 1536, 4096, 333, 2),
    "FLUX.1-dev 1024²":        (57, 3072, 4096, 512, 1),      # 19 双流 + 38 单流，这里按同一宽度估
    "FLUX.1-dev 2048²":        (57, 3072, 16384, 512, 1),
    "CogVideoX-5B 480p 49 帧": (42, 3072, 17550, 226, 2),
    "Wan 2.1-14B 720p 81 帧":  (40, 5120, 75600, 512, 2),
    "HunyuanVideo 720p 129 帧":(60, 3072, 118800, 256, 1),
}

def step_flops(layers, d, n_img, n_txt):
    n = n_img + n_txt
    linear = 24 * n * d * d
    attn = 4 * n * n * d
    return layers * linear, layers * attn

print(f"{'模型':<26} {'token':>8} {'临界 N=6d':>9} {'线性 TFLOP':>10} {'注意力 TFLOP':>12} {'注意力占比':>9} {'一步合计':>9}")
for name, (L, d, ni, nt, cfg) in MODELS.items():
    lin, att = step_flops(L, d, ni, nt)
    print(f"{name:<26} {ni + nt:>8,} {6 * d:>9,} {lin / 1e12:>10.1f} {att / 1e12:>12.1f} {att / (lin + att):>9.0%} {(lin + att) * cfg / 1e12:>9.1f}")
```

```text title="输出"
模型                            token   临界 N=6d   线性 TFLOP    注意力 TFLOP     注意力占比      一步合计
SD3-medium 1024²              4,429     9,216        6.0          2.9       32%      17.8
FLUX.1-dev 1024²              4,608    18,432       59.5         14.9       20%      74.4
FLUX.1-dev 2048²             16,896    18,432      218.1        200.0       48%     418.1
CogVideoX-5B 480p 49 帧       17,776    18,432      169.1        163.1       49%     664.4
Wan 2.1-14B 720p 81 帧        76,112    30,720     1915.4       4745.7       71%   13322.2
HunyuanVideo 720p 129 帧     119,056    18,432     1617.9      10450.5       87%   12068.4
```

三个结论：

- **图像模型在线性层主导的一侧**：FLUX 1024² 的注意力只占两成多。这时候量化矩阵乘、融合线性层比优化注意力 kernel 更有效。
- **视频模型在注意力主导的一侧**：Wan 720p 的注意力占七成以上，HunyuanVideo 超过八成。这时候一切围绕注意力：FlashAttention / SageAttention、稀疏注意力、序列并行。
- **分辨率翻倍，注意力翻 16 倍**：FLUX 从 1024² 到 2048²，线性层涨 4 倍、注意力涨 16 倍，占比从两成跳到五成。同一个模型在不同分辨率下，瓶颈会换位置。

这些数字是估算（FLUX 的双流块比单流宽、Wan 还有交叉注意力），但量级和占比可靠，足够决定优先级。

把公式做成计算器：换模型、改 token 数和步数，看线性层和注意力的占比怎么换位、一次生成在不同卡上要多久：

<div class="aig-widget" data-widget="diffusion-flops"></div>

## 算力受限还是带宽受限

LLM 推理最重要的区分是 prefill 算力受限、decode 带宽受限（见 [KV Cache 与两阶段推理](llm://inference/kv-cache/)）。扩散模型的一步相当于一次"全 token 的 prefill"：每读一次权重，要对 $N$ 个 token 做计算。算术强度（每字节权重做多少 FLOP）就是：

```python
H100 = dict(tflops=989, bw=3.35)          # bf16 稠密 TFLOPS，显存带宽 TB/s
RTX4090 = dict(tflops=165, bw=1.0)
ridge = {k: v["tflops"] * 1e12 / (v["bw"] * 1e12) for k, v in [("H100", H100), ("RTX 4090", RTX4090)]}
print("屋顶线拐点（FLOP / 字节）：" + "，".join(f"{k} {v:.0f}" for k, v in ridge.items()))
print()
print(f"{'模型':<26} {'权重 GB(bf16)':>13} {'一步 TFLOP':>10} {'算术强度':>9}  结论")
PARAMS = {"SD3-medium 1024²": 2.0, "FLUX.1-dev 1024²": 11.9, "FLUX.1-dev 2048²": 11.9,
          "CogVideoX-5B 480p 49 帧": 5.0, "Wan 2.1-14B 720p 81 帧": 14.0, "HunyuanVideo 720p 129 帧": 12.7}
for name, (L, d, ni, nt, cfg) in MODELS.items():
    lin, att = step_flops(L, d, ni, nt)
    weight_bytes = PARAMS[name] * 1e9 * 2
    ai = (lin + att) / weight_bytes
    print(f"{name:<26} {weight_bytes / 1e9:>13.1f} {(lin + att) / 1e12:>10.1f} {ai:>9,.0f}  {'算力受限' if ai > ridge['H100'] else '带宽受限'}")
print()
print("对比 LLM decode：7B 模型 batch=1 每读 14 GB 权重只算 2×7G FLOP，算术强度 1——深陷带宽受限")
```

```text title="输出"
屋顶线拐点（FLOP / 字节）：H100 295，RTX 4090 165

模型                           权重 GB(bf16)   一步 TFLOP      算术强度  结论
SD3-medium 1024²                     4.0        8.9     2,228  算力受限
FLUX.1-dev 1024²                    23.8       74.4     3,124  算力受限
FLUX.1-dev 2048²                    23.8      418.1    17,566  算力受限
CogVideoX-5B 480p 49 帧              10.0      332.2    33,218  算力受限
Wan 2.1-14B 720p 81 帧               28.0     6661.1   237,896  算力受限
HunyuanVideo 720p 129 帧             25.4    12068.4   475,133  算力受限

对比 LLM decode：7B 模型 batch=1 每读 14 GB 权重只算 2×7G FLOP，算术强度 1——深陷带宽受限
```

**扩散模型的每一步都远在拐点右侧**：batch=1 时就已经算力受限。这解释了几件事：

- 批处理对扩散模型吞吐帮助很小：LLM 的 decode 靠 batch 把算术强度从 1 拉到几百，扩散模型的一步本来就在几千；
- 权重量化（W8、W4）对扩散模型的收益主要是**省显存**，不像 LLM decode 那样直接换来速度；要提速得量化激活（W8A8、FP8 甚至 W4A4），让矩阵乘本身变快（见量化一章）；
- 真正带宽受限的是那些**不做矩阵乘的算子**：归一化、AdaLN 调制、激活函数、残差相加、VAE 的卷积——它们 FLOP 少但要把整个激活读写一遍。它们在时间线上的占比常常和矩阵乘相当，是 torch.compile 和算子融合的用武之地（见算子加速一章）。

## 一步要多久

有了 FLOP，一步的时间就是 $t = \text{FLOP} / (\text{MFU} \times \text{峰值})$。MFU（model FLOPs utilization）是实测吞吐和峰值的比值，扩散模型的典型值：

| 情形 | 典型 MFU | 为什么 |
| --- | --- | --- |
| DiT，bf16，FlashAttention，torch.compile | 40%～55% | 几乎全是大矩阵乘和注意力 |
| DiT，不编译 | 25%～40% | 调制、归一化、激活函数的小 kernel 占了时间 |
| UNet（SDXL） | 15%～30% | 卷积、GroupNorm、不同形状的层 |
| 任何模型，batch 很小、分辨率很低 | 更低 | kernel 太小填不满 GPU |

```python
GPUS = {"H100 SXM": 989, "RTX 4090": 165, "RTX 5070 Ti（估）": 170, "A100": 312}   # bf16 稠密 TFLOPS
def one_step(name):                                   # 用上面的公式算一次前向的 TFLOP
    L, d, ni, nt, cfg = MODELS[name]
    lin, att = step_flops(L, d, ni, nt)
    return (lin + att) / 1e12
CASES = [("SDXL 1024² · 30 步 · CFG", 12.0, 60, 0.25),                               # UNet：按实测时延反推的一次前向
         ("FLUX.1-dev 1024² · 28 步", one_step("FLUX.1-dev 1024²"), 28, 0.45),
         ("Wan 2.1-14B 720p 81 帧 · 50 步 · CFG", one_step("Wan 2.1-14B 720p 81 帧"), 100, 0.45)]
print(f"{'配置':<36} " + " ".join(f"{g:>14}" for g in GPUS))
for name, tflop, nfe, mfu in CASES:
    row = []
    for g, peak in GPUS.items():
        secs = tflop * nfe / (mfu * peak)
        row.append(f"{secs:>12.1f} s" if secs < 600 else f"{secs / 60:>10.1f} min")
    print(f"{name:<36} " + " ".join(f"{r:>14}" for r in row))
print("（一次前向的 TFLOP 来自本章的公式，SDXL 按实测反推；MFU 按上表取值；没算文本编码和 VAE）")
```

```text title="输出"
配置                                         H100 SXM       RTX 4090 RTX 5070 Ti（估）           A100
SDXL 1024² · 30 步 · CFG                       2.9 s         17.5 s         16.9 s          9.2 s
FLUX.1-dev 1024² · 28 步                       4.7 s         28.0 s         27.2 s         14.8 s
Wan 2.1-14B 720p 81 帧 · 50 步 · CFG         24.9 min      149.5 min      145.1 min       79.1 min
（一次前向的 TFLOP 来自本章的公式，SDXL 按实测反推；MFU 按上表取值；没算文本编码和 VAE）
```

这张表的用法不是精确预测，而是**定数量级、找不合理**：如果你在 4090 上跑 FLUX 一张图用了 40 秒，表上说 8 秒，那差距要么在 MFU（没开编译、注意力后端退化了），要么在显存（offload 到 CPU 了），要么在数据（VAE 解码没分块、在等显存）。真实的时延还要加上文本编码（几十毫秒）和 VAE 解码（1024² 约 0.1～0.3 秒，视频几秒到几十秒）。

视频那一行最值得看：**单卡 H100 上一条 5 秒 720p 视频要二十多分钟**（公开的实测也在这个量级）。这就是视频推理必须上多卡并行、必须缓存、必须蒸馏的原因——不是锦上添花，而是没有它们就没法当服务用。

## batch 在这里意味着什么

LLM 服务的核心是把很多请求拼成一个 batch 分摊权重读取；扩散模型一步已经算力受限，拼 batch 分摊不了什么。看一下吞吐和延迟怎么随 batch 变：

```python
def throughput(batch, tflop_per_image, peak, mfu_batch1=0.40, mfu_gain=0.08):
    """batch 变大时 MFU 略有提升（kernel 更满），但一步的 FLOP 线性增长。"""
    mfu = min(0.6, mfu_batch1 + mfu_gain * (batch - 1) ** 0.5)
    step = batch * tflop_per_image / (mfu * peak)          # 这一批一步的时间（秒）
    return mfu, step, batch / step                          # MFU、一步时间、每秒多少张图的"一步"

print(f"{'batch':>5} {'MFU':>5} {'一步时间':>8} {'吞吐(相对 batch=1)':>18} {'单张延迟(相对)':>14}")
_, s1, t1 = throughput(1, 18.0, 989)
for b in (1, 2, 4, 8):
    mfu, s, t = throughput(b, 18.0, 989)
    print(f"{b:>5} {mfu:>5.0%} {s * 1000:>6.0f} ms {t / t1:>18.2f}× {s / s1:>13.2f}×")
```

```text title="输出"
batch   MFU     一步时间     吞吐(相对 batch=1)       单张延迟(相对)
    1   40%     46 ms               1.00×          1.00×
    2   48%     76 ms               1.20×          1.67×
    4   54%    135 ms               1.35×          2.97×
    8   60%    243 ms               1.50×          5.33×
```

batch 从 1 到 8，吞吐只涨三成（来自 MFU 略好），延迟却涨六倍。所以生成服务的调度逻辑和 LLM 几乎相反：**不追求大 batch，追求把每张卡填满、把请求排开**；batch 真正有用的场景是"一个提示词出 4 张候选图"这种天然同形状的请求，或者小模型 + 低分辨率时 kernel 填不满 GPU 的情形（见[生成服务的调度](../serving/scheduling.md)一章）。

!!! interview "怎么讲清楚"
    讲"扩散模型推理的瓶颈在哪"，先给公式：一层 $24Nd^2 + 4N^2d$，临界 $N = 6d$；图像在线性层一侧（量化矩阵乘有效），视频在注意力一侧（一切围绕注意力）。再给定位：每一步算术强度几千，远在屋顶线拐点右侧，是算力受限的——和 LLM decode 相反，所以 batch 对吞吐帮助很小、权重量化只省显存不提速；真正带宽受限的是归一化、调制这些小算子，要靠编译融合。最后给数字：单卡 H100 一条 5 秒 720p 视频要二十多分钟，所以视频必须并行 + 缓存 + 蒸馏。

## 练习

1. 用本章的公式算 SD3-medium 从 1024² 升到 2048²，一步的 FLOP 变成几倍、注意力占比变成多少。再和 FLUX 比：同样的分辨率跳变，哪个模型的注意力占比变化更大？为什么？

??? success "参考答案"
    token 数从 4429 到 16717，线性层约 ×3.8，注意力约 ×14，合计约 ×6；注意力占比从约 25% 升到约 55%。SD3 的 $d = 1536$ 比 FLUX 小一半，临界点 $6d$ 只有 9216，所以同样的 token 数下 SD3 更早进入注意力主导——隐藏维越小的模型，分辨率上去之后越依赖注意力优化。

2. 一张 RTX 4090（bf16 165 TFLOPS，带宽 1 TB/s）跑 Wan 2.1-1.3B（$d = 1536$、30 层）480p 81 帧，估算一步的 FLOP、算术强度和时间（MFU 取 40%）。它是算力受限的吗？

??? success "参考答案"
    潜变量 21×60×104，patch 2 后约 32,760 个 token 加 512 文本 token；线性层约 $30 \times 24 \times 33k \times 1536^2 \approx 56$ TFLOP，注意力约 $30 \times 4 \times 33k^2 \times 1536 \approx 200$ TFLOP，一步约 260 TFLOP；权重 2.6 GB，算术强度约 10 万，远远算力受限；一步约 $260 / (0.4 \times 165) \approx 4$ 秒，50 步 CFG 要 400 秒。1.3B 的"小"模型在视频上也不小——因为成本在 $N^2$ 而不在参数量。

3. 某服务把 FLUX 的请求攒到 batch=4 再跑，声称"吞吐翻了 4 倍"。用本章的模型指出这个说法哪里不对，什么情况下它可能是对的。

??? success "参考答案"
    一步算力受限时 batch=4 的一步时间约是 batch=1 的 4 倍，吞吐不变、延迟 ×4；只有当 batch=1 时 GPU 没填满（低分辨率、小模型、kernel 太小），batch 才能提高 MFU，吞吐才会涨——涨幅是 MFU 的提升比例，不可能到 4 倍。更可能的真实原因是 batch=1 时的实现有别的问题（没编译、CPU 开销大）。

## 小结

- [x] DiT 一步的 FLOP 是 $L\,(24Nd^2 + 4N^2d)$，临界点 $N = 6d$：图像模型在线性层一侧，视频模型在注意力一侧，分辨率翻倍注意力翻 16 倍。
- [x] 每一步的算术强度几千，batch=1 就算力受限；batch 对吞吐帮助很小，权重量化主要省显存，真正带宽受限的是归一化、调制这些小算子。
- [x] 一步时间 ≈ FLOP / (MFU × 峰值)，DiT 的 MFU 40%～55%、UNet 15%～30%；单卡 H100 一条 5 秒 720p 视频要二十多分钟。
- [x] 这张账决定了后面每种手段的位置：少步和缓存减前向次数，量化和编译降单步成本，并行把单步摊开，而 batch 不是答案。
