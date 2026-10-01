# 一张图是怎么生成的：pipeline 解剖

<p class="lead">`pipe("a cat")` 一行就能出图，但做推理优化必须知道这一行里到底跑了哪几个模型、各跑了几次、哪一个吃掉了大部分时间和显存。这一章不用 diffusers 的 pipeline 类，而是把文本编码器、去噪网络、VAE 三个部件用最小的配置在 CPU 上搭起来，亲手把去噪循环串一遍，数清每个部件的参数量、调用次数和计算量占比；然后把同样的账算到 SD 1.5、SDXL、SD3、FLUX 和视频模型上。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 文生图的 pipeline 由哪三类模型组成？各自被调用几次？
    2. 为什么说去噪网络几乎吃掉了全部计算量，而 VAE 决定了显存峰值？
    3. 文本编码器为什么可以提前算好、甚至缓存？
    4. 潜空间的形状由什么决定？1024×1024 的图在 SDXL 和 FLUX 里各对应多少个 token？
    5. 把 pipeline 拆成三个阶段之后，服务层能做哪些单一 pipeline 做不了的事？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 文本编码器（CLIP / T5）、去噪网络（UNet 或 DiT）、VAE。文本编码器每个提示词调用 1 次（CFG 时再编码一次空提示词），去噪网络调用 步数 × CFG 倍数 次，VAE 解码器调用 1 次（图生图时编码器也调用 1 次）。
    2. 去噪网络被调用几十次，每次都是完整的前向；VAE 只跑一次，但它在像素分辨率上做卷积，激活是潜空间的 64 倍（f8 压缩两个方向各 8 倍），1024×1024 解码时的中间激活比去噪网络一步还大。
    3. 它的输入只有提示词，和噪声、步数无关，输出的向量在整个去噪过程中不变。同一个提示词的编码可以缓存；服务层还可以把文本编码器放到另一张卡或 CPU 上，和去噪流水化。
    4. 潜空间 = 像素分辨率 ÷ VAE 压缩比，通道数由 VAE 决定。SDXL：1024/8 = 128，128×128×4，UNet 里不切 patch，注意力在 32×32 到 128×128 的特征图上做；FLUX：128×128×16，再按 2×2 打包成 64×64 = 4096 个 token，每个 token 64 维。
    5. 分阶段批处理（文本编码可以大 batch）、按阶段放到不同设备、缓存文本编码和中间结果、预览（去噪中途用小解码器出低清图）、把 VAE 解码和下一个请求的去噪重叠。

## 三个部件

```
提示词 ──► 文本编码器 ──► 条件向量 ─┐
                                    ▼
随机噪声 ──► [去噪网络 × N 步（CFG 则每步两次）] ──► 干净潜变量 ──► VAE 解码器 ──► 图像
```

用 diffusers 的类、但只给最小的配置，在 CPU 上把它们搭出来。权重随机初始化（这一章只关心形状和计算量，不关心画得好不好），所以不需要下载任何东西：

```python
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextConfig, CLIPTextModel

torch.manual_seed(0)
text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=1000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                            num_attention_heads=4, max_position_embeddings=16))
unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
                            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                            cross_attention_dim=32, attention_head_dim=8, norm_num_groups=8)
vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(32, 64), latent_channels=4, norm_num_groups=8,
                    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"), sample_size=32)
scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear", clip_sample=False)

def count(m):
    return sum(p.numel() for p in m.parameters())

for name, m in [("文本编码器", text_encoder), ("去噪网络 UNet", unet), ("VAE", vae)]:
    print(f"{name:<10} {count(m) / 1e6:6.2f}M 参数")
print(f"这个玩具 VAE 的压缩比是 1/{2 ** (len(vae.config.block_out_channels) - 1)}（真实的 SD VAE 是 1/8），潜空间通道 {vae.config.latent_channels}")
```

```text title="输出"
文本编码器        0.05M 参数
去噪网络 UNet    0.99M 参数
VAE          0.66M 参数
这个玩具 VAE 的压缩比是 1/2（真实的 SD VAE 是 1/8），潜空间通道 4
```

## 手动跑一遍去噪循环

pipeline 类做的事就是下面这几十行。关键是数清楚每个部件被调用了几次：

```python
calls = {"text": 0, "unet": 0, "vae": 0}

@torch.no_grad()
def generate(token_ids, steps=10, guidance=7.5, seed=0):
    # 1. 文本编码：有条件 + 无条件（空提示词）各一次
    cond = text_encoder(token_ids).last_hidden_state
    uncond = text_encoder(torch.zeros_like(token_ids)).last_hidden_state
    calls["text"] += 2
    context = torch.cat([uncond, cond])                           # CFG 的两路拼成 batch=2

    # 2. 从纯噪声出发，按调度器给的时间步一步步去噪
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size, generator=g)
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(torch.cat([latents, latents]), t)
        eps = unet(x, t, encoder_hidden_states=context).sample     # 一次前向，batch=2
        calls["unet"] += 1
        eps_u, eps_c = eps.chunk(2)
        eps = eps_u + guidance * (eps_c - eps_u)                   # CFG 外推
        latents = scheduler.step(eps, t, latents).prev_sample

    # 3. VAE 解码：除以 scaling_factor 回到 VAE 的潜空间尺度
    calls["vae"] += 1
    image = vae.decode(latents / vae.config.scaling_factor).sample
    return image

ids = torch.randint(1, 1000, (1, 16))
img = generate(ids, steps=10)
print(f"输出图像 {tuple(img.shape)}，像素范围约 [{img.min():.1f}, {img.max():.1f}]（随机权重，不是一张真的图）")
print(f"调用次数：文本编码器 {calls['text']}，去噪网络 {calls['unet']}（10 步，每步 batch=2 的一次前向），VAE 解码 {calls['vae']}")
```

```text title="输出"
输出图像 (1, 3, 32, 32)，像素范围约 [-1.2, 1.2]（随机权重，不是一张真的图）
调用次数：文本编码器 2，去噪网络 10（10 步，每步 batch=2 的一次前向），VAE 解码 1
```

三个部件的调用次数分别是 2、N、1。这个结构决定了优化的优先级：**去噪网络的一次前向乘以 N 是全部**，文本编码器和 VAE 各只有一次，但它们有自己的问题——下面算账。

## 算账：每个部件各占多少计算量

用 PyTorch 的 FLOP 计数器把三个部件各跑一次，再按调用次数加权：

```python
from torch.utils.flop_counter import FlopCounterMode

def flops(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()

x2 = torch.randn(2, 4, 16, 16)
ctx = torch.randn(2, 16, 32)
z = torch.randn(1, 4, 16, 16)
f_text = flops(lambda: text_encoder(ids))
f_unet = flops(lambda: unet(x2, torch.tensor([500]), encoder_hidden_states=ctx))     # batch=2：CFG 的一步
f_vae = flops(lambda: vae.decode(z))

steps = 10
total = 2 * f_text + steps * f_unet + f_vae
print(f"{'部件':<12} {'一次前向 MFLOP':>14} {'调用次数':>8} {'占比':>7}")
for name, f, n in [("文本编码器", f_text, 2), ("去噪网络（CFG）", f_unet, steps), ("VAE 解码", f_vae, 1)]:
    print(f"{name:<12} {f / 1e6:>14.1f} {n:>8} {n * f / total:>7.1%}")
```

```text title="输出"
部件               一次前向 MFLOP     调用次数      占比
文本编码器                   0.5        2    0.0%
去噪网络（CFG）             325.9       10   90.6%
VAE 解码                336.4        1    9.4%
```

这个玩具里 VAE 的一次解码就和去噪网络的一步同量级——因为它在像素分辨率上做卷积。真实模型里这个比例更夸张：SDXL 的 VAE 解码 1024×1024 一张图要约 2.5 TFLOP，和 UNet 一步（约 3 TFLOP，CFG 后翻倍）相当，而它的**激活显存峰值**是去噪网络一步的好几倍，这是分块解码（见 [VAE 与潜空间](vae-latent.md)）存在的原因。

## 真实模型的账

把同样的三个部件换成真实配置，按公开的参数量和每一步的计算量估算（数字是估算，各家实现、分辨率和步数不同会有出入）：

```python
# 去噪网络一次前向的 FLOP（估算）：DiT 按 2 × 参数量 × token 数（线性层）加每层 4N²d 的注意力累加，
# UNet 按实测时延反推；文本编码器和 VAE 各只跑一次，这里不计
MODELS = [
    # 名字,                     文本编码器,                        去噪网络,     一次前向 TFLOP, 步数, CFG
    ("SD 1.5 (512²)",           "CLIP-L 0.12B",                   "UNet 0.86B",      1.2,  25, 2),
    ("SDXL",                    "CLIP-L 0.12B + CLIP-G 0.69B",    "UNet 2.6B",      12.0,  30, 2),
    ("SD3-medium",              "CLIP-L + CLIP-G + T5-XXL 4.7B",  "MMDiT 2B",       20.0,  28, 2),
    ("FLUX.1-dev",              "CLIP-L + T5-XXL 4.7B",           "MMDiT 12B",      90.0,  28, 1),
    ("Wan 2.1-14B (720p 81 帧)", "umT5-XXL 5.7B",                 "DiT 14B",      6900.0,  50, 2),
]
print(f"{'模型':<26} {'去噪网络':<12} {'一次前向 TFLOP':>13} {'前向次数':>8} {'去噪总量 PFLOP':>14}")
for name, te, dn, tf, steps, cfg in MODELS:
    print(f"{name:<26} {dn:<12} {tf:>13,.1f} {steps * cfg:>8} {tf * steps * cfg / 1000:>14,.2f}")
```

```text title="输出"
模型                         去噪网络            一次前向 TFLOP     前向次数     去噪总量 PFLOP
SD 1.5 (512²)              UNet 0.86B             1.2       50           0.06
SDXL                       UNet 2.6B             12.0       60           0.72
SD3-medium                 MMDiT 2B              20.0       56           1.12
FLUX.1-dev                 MMDiT 12B             90.0       28           2.52
Wan 2.1-14B (720p 81 帧)    DiT 14B            6,900.0      100         690.00
```

几个读数：

- **SDXL 一张图约 0.7 PFLOP**，一张 H100 的 bf16 稠密算力约 1 PFLOP/s，理论上不到一秒；实际 2～4 秒，因为 UNet 里大量的卷积、归一化、小算子是访存受限的，MFU 只有两三成（见[推理的算账](../perf/accounting.md)）。
- **FLUX 一次前向是 SDXL 的 7 倍多**，省掉 CFG 之后总量仍是 SDXL 的 3 倍多。参数量从 2.6B 到 12B 带来的单步成本，不是靠省掉 CFG 能抵消的。
- **视频差了三个数量级**。Wan 14B 生成 5 秒 720p 视频要约 700 PFLOP，是 SDXL 的近 1000 倍——token 数（帧 × 高 × 宽）既进了线性层的 $N$，又进了注意力的 $N^2$（见[视频推理的瓶颈](../video/bottleneck.md)）。单卡要二十多分钟，多卡并行才是视频推理的常态。
- 文本编码器看起来很大（T5-XXL 4.7B），但只跑一次、输入只有几十到几百个 token，计算量可以忽略；它真正的成本是**显存**：bf16 下 9.5 GB，和 FLUX 的 24 GB 权重加在一起，16 GB 的卡放不下，于是有了"编码完卸载到 CPU"和"用 fp8 / NF4 存 T5"这些手段（见[显存与 offload](../perf/memory.md)）。

## 拆开之后能做什么

把 pipeline 看成三个阶段而不是一个黑盒，服务层立刻多出几种手段：

| 手段 | 做法 | 收益 |
| --- | --- | --- |
| 文本编码缓存 | 按提示词哈希缓存编码结果 | 重复提示词（批量出图、换种子）直接跳过文本编码器 |
| 分阶段批处理 | 文本编码器和 VAE 攒大 batch，去噪按形状分组 | 三个阶段的最优 batch 不一样 |
| 阶段流水 | 请求 A 在做 VAE 解码时，请求 B 开始去噪 | GPU 不空转，尤其 VAE 解码慢的高分辨率 |
| 预览 | 中途用 TAESD 这类几 MB 的小解码器出低清图 | 用户几步后就能看到大概，可以提前取消 |
| 异构放置 | T5 放到另一张卡或 CPU，VAE 放到另一张卡 | 大卡只留给去噪网络 |

这些在生成服务的调度里展开。它们的前提都是本章这张"谁被调用几次、谁吃算力、谁吃显存"的账。

!!! interview "面试怎么答"
    被问"文生图的 pipeline 里时间花在哪"，先给结构：文本编码 1 次、去噪 N 次、VAE 1 次；再给比例：去噪网络占 90% 以上的计算量，是优化的主战场；然后指出两个容易漏的点——VAE 解码的激活显存峰值可能是去噪一步的好几倍（所以要分块），T5 这种大文本编码器算力可忽略但显存不能忽略（所以要卸载或量化）。最后说清楚视频为什么不一样：token 数进了注意力的平方项，总计算量比图像高两三个数量级。

## 练习

1. 把 `generate` 里的 `guidance` 设成 1.0（相当于不用 CFG），改写成只跑有条件那一路。调用次数和每次的 batch 各变成多少？FLOP 省了多少？

??? success "参考答案"
    文本编码器 1 次（不用再编码空提示词），去噪网络还是 N 次但 batch 从 2 变 1，VAE 不变。去噪网络的 FLOP 减半，总量接近减半。FLUX.1-dev 把引导烘进了模型，所以它天然就是这个形态。

2. 用 `FlopCounterMode` 分别测 `vae.encode` 和 `vae.decode`，哪个更贵？为什么图生图（img2img）的额外成本通常不大？

??? success "参考答案"
    解码器比编码器贵（解码器的上采样部分通道数多、分辨率高，SD 的 VAE 解码器约 50M 参数而编码器约 34M）。图生图只是多一次编码，再从中间时间步开始去噪（跳过前面一部分步数），总成本往往比文生图还低。

3. SDXL 生成一张 2048×2048 的图，潜空间多大？UNet 最低分辨率层的注意力 token 数是 1024² 时的几倍？注意力的计算量是几倍？

??? success "参考答案"
    潜空间 256×256×4。UNet 的注意力在下采样 2 倍和 4 倍的特征图上做，最低分辨率层 64×64 = 4096 个 token，是 1024² 时（32×32 = 1024）的 4 倍；注意力计算量随 token 数平方增长，是 16 倍。这就是为什么高分辨率推理要靠分块（tiled）或先低清再超分，而不是直接把分辨率翻倍。

## 小结

- [x] 文生图 = 文本编码 1 次 + 去噪网络 N 次（CFG 则每次 batch 翻倍）+ VAE 解码 1 次；这个结构决定了优化的优先级。
- [x] 去噪网络占 90% 以上的计算量；VAE 决定激活显存峰值；大文本编码器算力可忽略但显存不能忽略。
- [x] SDXL 一张图约 0.7 PFLOP，FLUX 约 2.5 PFLOP，5 秒 720p 视频约 700 PFLOP——视频比图像高三个数量级，根源是 token 数进了线性层的 $N$ 和注意力的 $N^2$。
- [x] 把 pipeline 拆成三个阶段，服务层才能做缓存、分阶段批处理、流水和预览。
