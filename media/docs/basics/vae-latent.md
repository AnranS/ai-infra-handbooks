# VAE 与潜空间

<p class="lead">去噪网络从来不直接碰像素：它工作在 VAE 压缩出来的潜空间里，1024×1024 的图在那里只是 128×128（SD）或者打包后的 64×64 个 token（FLUX）。VAE 只跑一次，计算量占比不高，但它决定了三件推理上很要紧的事：潜空间多大（去噪网络的 token 数）、解码时的显存峰值（常常比去噪一步还高），以及视频模型里时间维怎么压缩。这一章把 VAE 的账算清楚，再把"分块解码"亲手实现一遍。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. SD 的 VAE 把 512×512×3 的图压成什么形状？压缩比是多少？为什么 16 通道的 VAE 画质更好？
    2. `scaling_factor` 是什么？去噪前后为什么要乘、除它？
    3. 1024×1024 的图解码时，VAE 的激活显存大概多大？为什么比去噪网络一步还大？
    4. 分块（tiled）解码为什么能省显存？拼接处为什么要重叠和混合？
    5. 视频 VAE 的 "4×8×8" 是什么意思？为什么它要做成因果的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 压成 64×64×4：空间两个方向各压 8 倍、通道从 3 变 4，总压缩比 48 倍（按元素数算）。16 通道的 VAE（SD3、FLUX）给每个潜像素更多信息，重建误差小一半以上，细小文字、人脸这类高频细节才保得住，代价是去噪网络的输入通道从 4 变 16（对计算量影响很小）。
    2. 一个标量（SD 1.5 是 0.18215），把 VAE 编码出的潜变量缩放到标准差接近 1，和去噪网络训练时的噪声尺度匹配。编码后乘、解码前除；忘了它，解码出来的图会严重过曝或发灰。
    3. 解码器最后几层在像素分辨率上有 128～256 个通道：1024×1024×256 个 fp16 元素就是 512 MB，再乘上残差块里的几份中间结果，峰值轻松过 4 GB；去噪网络一步在 128×128 的潜空间上，激活小得多。
    4. 把潜变量切成小块分别解码，每次只有一块的激活在显存里；相邻块重叠一段、在重叠区按权重线性混合，否则块边界会有明显的接缝。
    5. 时间维每 4 帧压成 1 帧、空间每 8×8 压成 1，所以 81 帧 720p 的视频变成 21×90×160 的潜变量。因果意味着每个潜帧只依赖它之前的帧——首帧可以单独编码（图生视频就是靠它），长视频可以分段流式解码，不用把整条视频的激活同时放进显存。

## 压缩了多少

用最小配置的 VAE 看编码、解码的形状和 `scaling_factor` 的作用：

```python
import torch
from diffusers import AutoencoderKL

torch.manual_seed(0)
vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(32, 64, 64), latent_channels=4, norm_num_groups=8,
                    down_block_types=("DownEncoderBlock2D",) * 3, up_block_types=("UpDecoderBlock2D",) * 3, sample_size=64)
f = 2 ** (len(vae.config.block_out_channels) - 1)
img = torch.randn(1, 3, 96, 96)
with torch.no_grad():
    dist = vae.encode(img).latent_dist               # 编码器给的是一个高斯分布（均值、方差）
    z = dist.sample() * vae.config.scaling_factor    # 采样一个潜变量，再乘 scaling_factor
    rec = vae.decode(z / vae.config.scaling_factor).sample
print(f"图像 {tuple(img.shape)} → 潜变量 {tuple(z.shape)}：空间压缩 1/{f}，元素数压缩 {img.numel() / z.numel():.0f} 倍")
print(f"scaling_factor={vae.config.scaling_factor}：真实 SD VAE 编码出的潜变量标准差约 5.5，乘上它才接近 1，和去噪网络训练时的噪声尺度一致"
      f"（这个随机初始化的玩具 VAE 没有这个性质，编码出的标准差只有 {dist.sample().std():.2f}）")
print(f"解码回 {tuple(rec.shape)}")
```

```text title="输出"
图像 (1, 3, 96, 96) → 潜变量 (1, 4, 24, 24)：空间压缩 1/4，元素数压缩 12 倍
scaling_factor=0.18215：真实 SD VAE 编码出的潜变量标准差约 5.5，乘上它才接近 1，和去噪网络训练时的噪声尺度一致（这个随机初始化的玩具 VAE 没有这个性质，编码出的标准差只有 1.19）
解码回 (1, 3, 96, 96)
```

把主流 VAE 的配置列出来：

| VAE | 空间压缩 | 时间压缩 | 潜通道 | 谁在用 | 备注 |
| --- | --- | --- | --- | --- | --- |
| SD VAE (kl-f8) | 8 | — | 4 | SD 1.5、SDXL | 1024² → 128×128×4 |
| SD3 / FLUX VAE | 8 | — | 16 | SD3、FLUX | 同样空间尺寸，4 倍信息量，细节明显更好 |
| DC-AE | 32 | — | 32～128 | SANA | 极高压缩，token 数少 16 倍，换来 DiT 的速度 |
| CogVideoX 3D VAE | 8 | 4 | 16 | CogVideoX | 因果 3D 卷积 |
| HunyuanVideo VAE | 8 | 4 | 16 | HunyuanVideo | 3D 因果 |
| Wan-VAE | 8 | 4 | 16 | Wan 2.1 / 2.2 | 因果，支持流式分段解码 |
| LTX-Video VAE | 32 | 8 | 128 | LTX-Video | 压缩最狠，速度最快，解码器自带一部分去噪 |

**空间压缩比直接决定去噪网络的 token 数**（见[去噪网络](unet-to-dit.md)的公式）。DC-AE 把 f8 换成 f32，同样分辨率下 token 数少 16 倍、注意力少 256 倍——SANA 能在笔记本上出 4K 图靠的就是这个。代价转移到了 VAE 身上：压缩越狠，VAE 本身越大、越难训、解码越贵。

## 解码的显存峰值

VAE 解码器在像素分辨率上有很宽的通道，激活比去噪网络一步大得多。用一个 hook 记下解码过程中最大的中间张量：

```python
peak = {"numel": 0, "shape": None, "where": ""}
def watch(name):
    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        if torch.is_tensor(out) and out.numel() > peak["numel"]:
            peak.update(numel=out.numel(), shape=tuple(out.shape), where=name)
    return hook
for name, m in vae.decoder.named_modules():
    if isinstance(m, torch.nn.Conv2d):
        m.register_forward_hook(watch(name))
with torch.no_grad():
    vae.decode(z / vae.config.scaling_factor)
print(f"解码时最大的中间张量：{peak['shape']}，在 {peak['where']}")
print(f"它是潜变量元素数的 {peak['numel'] / z.numel():.0f} 倍")

# 推到真实尺寸：SD VAE 解码 1024×1024，最宽的像素级特征是 128 通道
H = W = 1024
act = H * W * 128 * 2                                  # fp16，一份
print(f"SD VAE 解码 1024²：一份像素级特征就有 {act / 2 ** 20:.0f} MB；残差块里同时存着好几份，GroupNorm 还要 fp32 副本，实测峰值 2～4 GB")
print(f"对比去噪一步：SDXL UNet 在 128×128 潜空间上，最大的激活约 128×128×320×2 字节 = {128 * 128 * 320 * 2 / 2 ** 20:.0f} MB")
```

```text title="输出"
解码时最大的中间张量：(1, 64, 96, 96)，在 up_blocks.1.upsamplers.0.conv
它是潜变量元素数的 256 倍
SD VAE 解码 1024²：一份像素级特征就有 256 MB；残差块里同时存着好几份，GroupNorm 还要 fp32 副本，实测峰值 2～4 GB
对比去噪一步：SDXL UNet 在 128×128 潜空间上，最大的激活约 128×128×320×2 字节 = 10 MB
```

这就是"去噪跑得好好的，解码时 OOM"的来源：**去噪网络的激活在潜空间，VAE 的激活在像素空间，差了 64 倍的面积**。视频更夸张——81 帧 720p 一次解码的像素级激活是单张图的 81 倍，没有分块根本放不下。

## 分块解码

既然卷积是局部的，就可以把潜变量切成块分别解码，每次只有一块的激活在显存里。但块边界处的感受野被截断，直接拼会有接缝；diffusers 的做法是**块之间重叠一段，重叠区按线性权重混合**。自己实现一遍最小版本，和整图解码比较：

```python
def tiled_decode(vae, z, tile=16, overlap=4):
    """潜空间按 tile 切块（相邻块重叠 overlap），分别解码后在重叠区线性混合。"""
    f = 2 ** (len(vae.config.block_out_channels) - 1)
    _, _, H, W = z.shape
    stride = tile - overlap
    out = torch.zeros(1, 3, H * f, W * f)
    weight = torch.zeros(1, 1, H * f, W * f)
    ramp = torch.linspace(0, 1, overlap * f)                # 重叠区的混合权重：0 → 1
    tiles = 0
    for y in range(0, H - overlap, stride):
        for x in range(0, W - overlap, stride):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            with torch.no_grad():
                piece = vae.decode(z[:, :, y:y1, x:x1] / vae.config.scaling_factor).sample
            w = torch.ones(1, 1, piece.shape[2], piece.shape[3])
            if y > 0: w[:, :, :overlap * f, :] *= ramp[:, None]
            if x > 0: w[:, :, :, :overlap * f] *= ramp[None, :]
            out[:, :, y * f:y1 * f, x * f:x1 * f] += piece * w
            weight[:, :, y * f:y1 * f, x * f:x1 * f] += w
            tiles += 1
    return out / weight, tiles

full = rec
tiled, n_tiles = tiled_decode(vae, z, tile=12, overlap=4)
err = (tiled - full).abs()
print(f"切成 {n_tiles} 块解码，和整图解码的平均误差 {err.mean():.4f}，最大误差 {err.max():.4f}（像素范围约 ±3）")
print(f"每块只解码 {12 * f}×{12 * f} 像素，像素级激活是整图的 1/{(full.shape[2] * full.shape[3]) / (12 * f) ** 2:.0f}，显存峰值按同样比例下降")
```

```text title="输出"
切成 9 块解码，和整图解码的平均误差 0.1134，最大误差 1.2141（像素范围约 ±3）
每块只解码 48×48 像素，像素级激活是整图的 1/4，显存峰值按同样比例下降
```

误差不是零——边界处的感受野不完整，混合只是把它藏起来。重叠越大越接近整图解码，但重复计算也越多（重叠 4、块 12 时每个块有一半以上的面积被算了两遍）。真实部署里的经验值：SD VAE 用 512 像素的块、64 像素的重叠，肉眼看不出接缝，显存从几 GB 降到几百 MB。

同样的思路还有两个变体：

- **编码也分块**（图生图、视频首帧条件时用到），原理相同；
- **时间分块**：视频 VAE 按潜帧切段解码，因果结构保证前面的段不依赖后面的——Wan 的 VAE 把这做成了流式接口，一边解码一边输出帧。

## 预览解码器

去噪到一半想看看大概长什么样（用户预览、提前取消），不值得跑一次完整的 VAE。TAESD 这类**蒸馏出来的小解码器**只有几 MB、几层卷积，直接从潜变量出一张糊但能看的图：

```python
import torch.nn as nn

class TinyDecoder(nn.Module):                        # 极简示意：一个上采样卷积塔，TAESD 的结构大致如此
    def __init__(self, latent_channels, f):
        super().__init__()
        layers, c = [], latent_channels
        for _ in range(int(f).bit_length() - 1):     # 每级 ×2，直到像素分辨率
            layers += [nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(), nn.Upsample(scale_factor=2)]
            c = 16
        layers += [nn.Conv2d(c, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

tiny = TinyDecoder(4, f)
with torch.no_grad():
    preview = tiny(z)
print(f"完整 VAE 解码器 {sum(p.numel() for p in vae.decoder.parameters()) / 1e3:.0f}K 参数，"
      f"预览解码器 {sum(p.numel() for p in tiny.parameters()) / 1e3:.1f}K 参数，输出同样是 {tuple(preview.shape)}")
```

```text title="输出"
完整 VAE 解码器 587K 参数，预览解码器 3.3K 参数，输出同样是 (1, 3, 96, 96)
```

真实的 TAESD 相对 SD VAE 解码器约 50M 参数只有 1.2M，快一个数量级，WebUI 里的"实时预览"用的就是它。服务层可以把它当作一种"渐进式输出"：每隔几步给用户一张预览，用户不满意就取消——省下的是后面所有步的算力（见[生成服务的调度](../serving/scheduling.md)）。

## 视频 VAE：时间维也要压

视频模型的 VAE 把时间维也压缩了：CogVideoX、HunyuanVideo、Wan 都是 **4×8×8**——时间每 4 帧压成 1 个潜帧，空间 8×8 压成 1。算一下它对去噪网络意味着什么：

```python
def latent_shape(frames, h, w, ft=4, f=8):
    return 1 + (frames - 1) // ft, h // f, w // f         # 首帧单独编码，之后每 ft 帧压一帧

for name, frames, h, w in [("CogVideoX 480p 49 帧", 49, 480, 720), ("Wan 2.1 480p 81 帧", 81, 480, 832),
                           ("Wan 2.1 720p 81 帧", 81, 720, 1280), ("HunyuanVideo 720p 129 帧", 129, 720, 1280)]:
    T, H, W = latent_shape(frames, h, w)
    pixels = frames * h * w * 3
    latent = T * H * W * 16
    print(f"{name:<24} 像素 {pixels / 1e6:>6.1f}M 个值 → 潜变量 {T}×{H}×{W}×16 = {latent / 1e6:>5.2f}M 个值，压缩 {pixels / latent:>4.0f} 倍")
```

```text title="输出"
CogVideoX 480p 49 帧      像素   50.8M 个值 → 潜变量 13×60×90×16 =  1.12M 个值，压缩   45 倍
Wan 2.1 480p 81 帧        像素   97.0M 个值 → 潜变量 21×60×104×16 =  2.10M 个值，压缩   46 倍
Wan 2.1 720p 81 帧        像素  223.9M 个值 → 潜变量 21×90×160×16 =  4.84M 个值，压缩   46 倍
HunyuanVideo 720p 129 帧  像素  356.7M 个值 → 潜变量 33×90×160×16 =  7.60M 个值，压缩   47 倍
```

压缩 48 倍左右，和图像 VAE 相当——但起点是几亿个像素，所以潜变量本身就有几百万个值、去噪网络的 token 数到了十万。还有两个视频特有的设计：

- **因果**：每个潜帧只看它之前的帧。首帧可以单独编码成一个潜帧（所以帧数都是 4k+1：49、81、129），图生视频把首帧的潜变量直接放进去当条件；解码时也能按时间分段流式进行，不用把整条视频的激活放进显存。
- **解码是大头**：81 帧 720p 的像素级激活是单张图的 81 倍，不分块（时间上、空间上都要）根本放不下；LTX-Video 干脆把 VAE 压到 32×32×8，让解码器承担一部分"去噪"工作，用解码质量换速度。

!!! interview "面试怎么答"
    被问"VAE 在推理里为什么重要"，三点：（1）它的空间压缩比决定了去噪网络的 token 数，f8 换 f32 就是 16 倍的 token 差距、256 倍的注意力差距；（2）它的解码在像素分辨率上有很宽的通道，激活显存峰值常常超过去噪一步，所以要分块解码，块之间重叠混合消接缝；（3）视频 VAE 多了时间压缩（4×8×8）和因果结构，因果让首帧条件和流式分段解码成为可能。再补一句 TAESD：几 MB 的蒸馏解码器用来做预览，是服务层做渐进输出和提前取消的基础。

## 练习

1. 把 `tiled_decode` 的 `overlap` 改成 0 和 8，分别看误差和解码的块数。重叠多少合适？

??? success "参考答案"
    overlap=0 时块边界处误差明显（接缝），overlap=8 时误差接近 0 但块数更多、重复计算更多。合适的重叠大约是解码器感受野的量级：SD VAE 的解码器感受野在像素空间约几十像素，所以 64 像素（潜空间 8）的重叠已经看不出接缝。

2. SDXL 的 VAE 用 fp16 解码会出 NaN，官方给了一个 `madebyollin/sdxl-vae-fp16-fix`。猜一猜问题出在哪，怎么修？

??? success "参考答案"
    解码器某些层的激活超出了 fp16 的范围（65504）溢出成 inf，再经过归一化变成 NaN。修复版把权重重新缩放，让激活落回 fp16 的范围；工程上也可以只给 VAE 用 bf16 或 fp32（它只跑一次，精度开销可以接受），或者用 `force_upcast`。这是"混合精度不能一刀切"的典型例子（见[混合精度与 FP8 训练](train://practice/mixed-precision/)）。

3. 一个 81 帧 720p 的视频，用 Wan-VAE 解码，像素级特征按 128 通道、fp16 估算，一次性解码的峰值激活多大？按时间分成几段才能压到 8 GB 以内（只算一份特征）？

??? success "参考答案"
    一份像素级特征 = 81 × 720 × 1280 × 128 × 2 字节 ≈ 19 GB；残差块里同时有几份，峰值远超 40 GB。按时间切成 4 段（每段约 20 帧）一份特征约 4.8 GB，再加上空间分块就能压到几 GB。因果 VAE 让时间分段天然可行——这也是 Wan 的解码器做成流式的原因。

## 小结

- [x] VAE 把像素压到潜空间（SD 是 1/8 空间、4 或 16 通道），空间压缩比直接决定去噪网络的 token 数；`scaling_factor` 把潜变量对齐到去噪网络的噪声尺度。
- [x] 解码器工作在像素分辨率，激活显存峰值常常超过去噪一步；分块解码 + 重叠混合把峰值压下来，代价是边界处的少量重复计算。
- [x] TAESD 这类蒸馏小解码器用来做预览，是渐进输出和提前取消的基础。
- [x] 视频 VAE 是 4×8×8 的因果压缩：首帧单独编码（帧数 4k+1）、能按时间流式分段解码；解码的激活是视频推理里最容易 OOM 的地方。
