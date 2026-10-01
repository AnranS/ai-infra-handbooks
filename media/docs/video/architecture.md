# 视频生成模型的结构：时空注意力与 3D VAE

<p class="lead">视频模型不是"图像模型多了一个维度"那么简单：潜变量多了时间轴，token 数乘上几十，注意力要在空间和时间两个方向上同时看；VAE 要把时间也压缩，还得是因果的；条件除了文本还有首帧、末帧、相机轨迹和音频。这一章把 CogVideoX、HunyuanVideo、Wan、LTX-Video 这些模型的结构拆开，算清"分解的时空注意力"和"3D 全注意力"各要多少计算，用 diffusers 的小配置把 3D VAE 和视频 DiT 搭起来跑一遍，为下一章的瓶颈分析打底。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 视频潜变量的形状由什么决定？81 帧 720p 在 Wan 里是多少个 token？
    2. "分解的时空注意力"和"3D 全注意力"分别怎么算？计算量差多少？为什么新模型都选后者？
    3. 3D VAE 为什么要做成因果的？"首帧单独编码"对图生视频意味着什么？
    4. 视频模型的位置编码怎么处理三个维度？
    5. 图生视频、首尾帧控制、相机控制各是怎么把条件喂进去的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 时间轴按 VAE 的时间压缩比（通常 4）压缩、首帧单独算一帧，空间按 8 压缩，再按 patch（通常 1×2×2）打包：$T' = 1 + (T-1)/4$，$H' = H/8/2$，$W' = W/8/2$。Wan 720p 81 帧：21 × 45 × 80 = 75,600 个 token。
    2. 分解：先在每一帧内做空间注意力（$T'$ 个长度 $H'W'$ 的序列），再在每个空间位置上做时间注意力（$H'W'$ 个长度 $T'$ 的序列），计算量 $\propto T'(H'W')^2 + H'W'T'^2$；全注意力：所有 token 一起做，$\propto (T'H'W')^2$。全注意力多 $T'$ 倍左右（Wan 720p 约 20 倍），但每个 token 能直接看到任何时刻任何位置，运动和一致性好得多——算力变便宜之后新模型全部选了它。
    3. 因果：每个潜帧只依赖它之前的帧。好处一是首帧可以单独编码成一帧（所以帧数都是 $4k+1$），图生视频把首帧的潜变量直接当条件；二是解码可以按时间分段流式进行，不用把整条视频的激活放进显存；三是同一个 VAE 既能编图像也能编视频。
    4. 3D RoPE：每个 token 的位置是 $(t, h, w)$ 三元组，把头维分成三段分别按三个坐标旋转（FLUX 的 2D RoPE 加一个时间维）。它天然支持不同的分辨率和帧数，序列并行时各卡只需知道自己那段 token 的坐标。
    5. 图生视频：首帧经 VAE 编码后，和噪声潜变量在通道维拼接（加一个掩码通道标记哪些帧是给定的），或者作为额外的 token 送进注意力；首尾帧控制同理，掩码标出首尾；相机控制把相机外参编码成每帧的嵌入加到时间步嵌入上，或者编成 Plücker 射线图和潜变量拼接；音频驱动把音频特征通过交叉注意力注入。

## 潜变量：多出来的时间轴

视频 VAE 把 $(T, H, W)$ 的像素压成 $(T', H', W')$ 的潜变量，去噪网络再按 patch 打包成 token：

```python
def video_tokens(frames, h, w, ft=4, fs=8, pt=1, ps=2, latent_ch=16):
    T = 1 + (frames - 1) // ft                       # 首帧单独一帧，之后每 ft 帧压一帧
    H, W = h // fs, w // fs
    tokens = (T // pt) * (H // ps) * (W // ps)
    return (T, H, W), tokens, latent_ch * pt * ps * ps   # 每个 token 的维度 = 通道 × patch 体积

print(f"{'模型 / 设置':<30} {'潜变量 T×H×W':>14} {'token 数':>9} {'token 维度':>8} {'比 1024² 图像(4096)':>16}")
for name, frames, h, w in [("CogVideoX 480p 49 帧", 49, 480, 720), ("Wan 2.1 480p 81 帧", 81, 480, 832),
                           ("Wan 2.1 720p 81 帧", 81, 720, 1280), ("HunyuanVideo 720p 129 帧", 129, 720, 1280),
                           ("Wan 2.2 1080p 121 帧（假设）", 121, 1080, 1920)]:
    (T, H, W), n, dim = video_tokens(frames, h, w)
    print(f"{name:<30} {f'{T}×{H}×{W}':>14} {n:>9,} {dim:>8} {n / 4096:>15.0f}×")
```

```text title="输出"
模型 / 设置                             潜变量 T×H×W   token 数 token 维度 比 1024² 图像(4096)
CogVideoX 480p 49 帧                  13×60×90    17,550       64               4×
Wan 2.1 480p 81 帧                   21×60×104    32,760       64               8×
Wan 2.1 720p 81 帧                   21×90×160    75,600       64              18×
HunyuanVideo 720p 129 帧             33×90×160   118,800       64              29×
Wan 2.2 1080p 121 帧（假设）            31×135×240   249,240       64              61×
```

token 数是图像的 20～60 倍。[算账一章](../perf/accounting.md)的公式里线性层 $\propto N$、注意力 $\propto N^2$，所以视频一步的计算量是图像的几十到上千倍，而且**注意力占了大头**——这就是视频模型结构设计的全部出发点。

## 注意力怎么看时间：分解还是全部

早期视频模型（AnimateDiff、SVD、CogVideo 一代）把注意力**分解**成空间和时间两步：先在每一帧内做空间注意力，再在每个空间位置上沿时间做注意力。新一代模型（CogVideoX、HunyuanVideo、Wan、Mochi）全部换成了 **3D 全注意力**：所有 token 一起做。两者的计算量：

```python
def attn_flops(T, H, W, d, layers, mode):
    n_sp = (H // 2) * (W // 2)                       # 每帧的 token 数（patch 2）
    if mode == "分解（空间 + 时间）":
        per_layer = T * 4 * n_sp ** 2 * d + n_sp * 4 * T ** 2 * d
    else:                                            # 3D 全注意力
        per_layer = 4 * (T * n_sp) ** 2 * d
    return layers * per_layer

for name, frames, h, w, d, L in [("CogVideoX-5B 480p 49 帧", 49, 480, 720, 3072, 42), ("Wan 2.1-14B 720p 81 帧", 81, 720, 1280, 5120, 40)]:
    (T, H, W), n, _ = video_tokens(frames, h, w)
    f1, f2 = attn_flops(T, H, W, d, L, "分解（空间 + 时间）"), attn_flops(T, H, W, d, L, "3D 全注意力")
    linear = L * 24 * n * d * d
    print(f"{name:<24} 分解注意力 {f1 / 1e12:>7.0f} TFLOP，3D 全注意力 {f2 / 1e12:>7.0f} TFLOP（{f2 / f1:>4.0f}×），线性层 {linear / 1e12:>6.0f} TFLOP")
```

```text title="输出"
CogVideoX-5B 480p 49 帧   分解注意力      12 TFLOP，3D 全注意力     159 TFLOP（  13×），线性层    167 TFLOP
Wan 2.1-14B 720p 81 帧    分解注意力     224 TFLOP，3D 全注意力    4682 TFLOP（  21×），线性层   1903 TFLOP
```

3D 全注意力比分解的贵一个数量级以上，一步的注意力就是几千 TFLOP——那为什么新模型都选它？因为分解注意力里**一个 token 看不到"另一时刻的另一位置"**，运动一致性、物体穿越画面时的连贯性都要靠多层堆叠间接实现，效果明显差；算力变得便宜之后，大家选择了结构上最干净的方案，把成本留给推理系统去解决。这就是[下一章](bottleneck.md)存在的理由。

把 token 排成一个 T×H×W 的立方阵，换几种注意力模式看一个 query 能看到哪些 token：

<div class="aig-widget" data-widget="video3d"></div>

用 diffusers 的视频 DiT 小配置看一下输入输出形状——它和图像 DiT 的差别只是多了一个时间维：

```python
import torch
from diffusers import CogVideoXTransformer3DModel

torch.manual_seed(0)
dit = CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=16, in_channels=4, out_channels=4,
                                  num_layers=1, sample_width=8, sample_height=8, sample_frames=9,
                                  patch_size=2, temporal_compression_ratio=4, max_text_seq_length=4,
                                  text_embed_dim=16, time_embed_dim=32, use_rotary_positional_embeddings=False).eval()
latent = torch.randn(1, 3, 4, 8, 8)                 # [B, T', C, H', W']：3 个潜帧（1 + 8/4）
text = torch.randn(1, 4, 16)
with torch.no_grad():
    out = dit(hidden_states=latent, encoder_hidden_states=text, timestep=torch.tensor([500])).sample
n_img = 3 * (8 // 2) * (8 // 2)
print(f"输入潜变量 {tuple(latent.shape)} → 视频 token {n_img} 个 + 文本 token 4 个 = 一起做 3D 全注意力，序列长度 {n_img + 4}")
print(f"输出 {tuple(out.shape)}：形状和输入一样，和图像 DiT 没有区别——区别只在 token 数")
```

```text title="输出"
输入潜变量 (1, 3, 4, 8, 8) → 视频 token 48 个 + 文本 token 4 个 = 一起做 3D 全注意力，序列长度 52
输出 (1, 3, 4, 8, 8)：形状和输入一样，和图像 DiT 没有区别——区别只在 token 数
```

## 3D VAE：时间也要压，而且得是因果的

视频 VAE 在图像 VAE 的 2D 卷积基础上换成 3D 卷积，时间维每 4 帧压 1 帧，空间仍是 8。先用 diffusers 的 CogVideoX VAE 小配置看形状：

```python
from diffusers import AutoencoderKLCogVideoX

torch.manual_seed(0)
vae = AutoencoderKLCogVideoX(in_channels=3, out_channels=3, block_out_channels=(8, 8, 8, 8), latent_channels=4,
                             layers_per_block=1, norm_num_groups=4, temporal_compression_ratio=4,
                             down_block_types=("CogVideoXDownBlock3D",) * 4, up_block_types=("CogVideoXUpBlock3D",) * 4).eval()
video = torch.randn(1, 3, 9, 32, 32)                # [B, C, T, H, W]：9 帧 = 1 + 2×4
with torch.no_grad():
    z = vae.encode(video).latent_dist.mode()
print(f"9 帧 {tuple(video.shape[2:])} → 潜变量 {tuple(z.shape[2:])}（T 从 9 到 {z.shape[2]}：首帧 + 每 4 帧一帧；空间 32 → {z.shape[3]}）")
```

```text title="输出"
9 帧 (9, 32, 32) → 潜变量 (3, 4, 4)（T 从 9 到 3：首帧 + 每 4 帧一帧；空间 32 → 4）
```

关键设计是**因果**：时间维的卷积只向过去看——padding 全补在前面，每个输出帧只依赖它和它之前的输入帧。用一个三层的因果 3D 卷积验证：改动后面的帧，前面的输出纹丝不动。

```python
import torch.nn as nn

class CausalConv3d(nn.Module):
    """时间维只在前面补零，所以第 t 帧的输出只看得到 ≤ t 的输入"""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=(k, 3, 3), padding=(0, 1, 1))
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, self.k - 1, 0))   # (W 左右, H 上下, T 前后)：只补 T 的前面
        return self.conv(x)

torch.manual_seed(0)
enc = nn.Sequential(CausalConv3d(3, 8), nn.SiLU(), CausalConv3d(8, 8), nn.SiLU(), CausalConv3d(8, 4)).eval()
x = torch.randn(1, 3, 9, 8, 8)
x2 = x.clone(); x2[:, :, 5:] += 1.0                 # 只改第 6 帧以后
with torch.no_grad():
    d = (enc(x) - enc(x2)).abs().amax(dim=(0, 1, 3, 4))
print("改动第 6 帧以后，各帧输出的变化：", [f"{v:.2f}" for v in d.tolist()])
print("前 5 帧的输出完全不变——这就是因果；真实的视频 VAE 还要把归一化层也做成逐帧的，否则统计量会把未来泄露回去")
```

```text title="输出"
改动第 6 帧以后，各帧输出的变化： ['0.00', '0.00', '0.00', '0.00', '0.00', '0.03', '0.06', '0.08', '0.13']
前 5 帧的输出完全不变——这就是因果；真实的视频 VAE 还要把归一化层也做成逐帧的，否则统计量会把未来泄露回去
```

因果带来三件推理上要紧的事：

1. **首帧可以单独编码**。它只依赖自己，所以一张图经同一个 VAE 编码就是第一个潜帧——图生视频（I2V）把它直接当条件；这也是帧数总是 $4k + 1$（49、81、129）的原因。
2. **解码可以流式分段**。前面的潜帧不依赖后面的，解码器可以按时间段逐段输出，显存只放一段（见 [VAE 与潜空间](../basics/vae-latent.md)里的分块）。Wan 的 VAE 把这做成了一边解码一边吐帧的接口。
3. **图像和视频共用一个 VAE**。$T = 1$ 就是图像；Wan、HunyuanVideo 都用同一个 VAE 做图生视频和文生视频。

主流 3D VAE 的参数：

| VAE | 时间 × 空间压缩 | 潜通道 | 因果 | 备注 |
| --- | --- | --- | --- | --- |
| CogVideoX | 4 × 8 × 8 | 16 | 是 | 3D 因果卷积 |
| HunyuanVideo | 4 × 8 × 8 | 16 | 是 | 带时间分块解码 |
| Wan 2.1 / 2.2 | 4 × 8 × 8（2.2 的 TI2V 版 4 × 16 × 16） | 16（2.2 版 48） | 是 | 流式解码接口 |
| Mochi 1 | 6 × 8 × 8 | 12 | 是 | 时间压缩更狠 |
| LTX-Video | 8 × 32 × 32 | 128 | 是 | 压缩 1:192，解码器自带去噪 |

压缩越狠，token 越少、去噪越快，但 VAE 要承担更多"生成"的工作。LTX-Video 的 1:192 让它能实时出 24 fps 的视频，代价是细节由解码器"脑补"。这是一个明显的设计权衡：**把成本从去噪网络挪到 VAE**。

## 位置：三维的 RoPE

视频 token 的位置是 $(t, h, w)$ 三元组。3D RoPE 把每个头的维度分成三段，分别按 $t$、$h$、$w$ 旋转（FLUX 的 2D RoPE 加一个时间维）。看一下它怎么分配维度，以及为什么它对序列并行友好：

```python
def rope_3d_dims(head_dim=128, axes=(16, 56, 56)):
    """每个头 128 维：16 维给时间、56 维给高、56 维给宽（Wan / HunyuanVideo 的分法）"""
    assert sum(axes) == head_dim
    return {"时间 t": axes[0], "高 h": axes[1], "宽 w": axes[2]}

print("头维 128 的分配：", rope_3d_dims())
# 序列并行时，每张卡只持有一段 token，但它们的 (t, h, w) 坐标是绝对的——各卡独立算自己的旋转即可，不需要通信
tokens = [(t, h, w) for t in range(2) for h in range(2) for w in range(3)]
P = 3
shards = [tokens[r * len(tokens) // P:(r + 1) * len(tokens) // P] for r in range(P)]
for r, s in enumerate(shards):
    print(f"卡 {r} 持有 token 的坐标：{s}")
```

```text title="输出"
头维 128 的分配： {'时间 t': 16, '高 h': 56, '宽 w': 56}
卡 0 持有 token 的坐标：[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)]
卡 1 持有 token 的坐标：[(0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1)]
卡 2 持有 token 的坐标：[(1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)]
```

时间维分到的维度最少（16 / 128）——帧数比空间尺寸小得多，不需要那么高的频率分辨率。工程上还有一个好处：坐标是绝对的，所以**换帧数、换分辨率不用改模型**，Wan 同一套权重跑 480p 和 720p、49 帧和 81 帧都靠它（质量仍受训练分布限制）。

## 条件：文本之外

| 任务 | 条件怎么进去 | 推理上的含义 |
| --- | --- | --- |
| 文生视频 | 文本 token 交叉注意力（Wan）或联合注意力（HunyuanVideo、CogVideoX） | 和图像相同 |
| 图生视频（I2V） | 首帧经 VAE 编码，与噪声潜变量在通道维拼接 + 一个掩码通道；或作为额外 token | 多一次 VAE 编码（因果 VAE 编一帧很快）；去噪网络输入通道变多 |
| 首尾帧 / 关键帧 | 同上，掩码标出哪些帧是给定的 | 同上 |
| 视频续写 / 延长 | 已有视频编码后放在时间轴前面当条件 | 条件帧也要过去噪网络，token 数增加 |
| 相机控制 | 相机外参编成每帧嵌入加到时间步嵌入，或 Plücker 射线图拼进输入 | 几乎无额外成本 |
| 音频驱动（数字人） | 音频特征交叉注意力注入 | 多一个编码器，多一段交叉注意力 |
| 参考图 / 主体一致 | 参考图的 token 拼进序列（联合注意力） | token 数增加，注意力成本上升 |

条件进入的方式决定成本：**拼在通道维的几乎免费，拼进 token 序列的要付注意力的平方**。服务里"同一个模型，有人传了 3 张参考图"这种请求，token 数和成本和纯文生视频差很多，调度时要当作不同形状对待（见[生成服务的调度](../serving/scheduling.md)一章）。

## 主流模型一览

| 模型 | 参数 | 去噪网络 | 注意力 | 文本编码器 | 典型输出 | 步数 |
| --- | --- | --- | --- | --- | --- | --- |
| CogVideoX-5B | 5B | DiT（专家 AdaLN） | 3D 全 | T5-XXL | 480p 49 帧 | 50 |
| HunyuanVideo | 13B | MMDiT（双流 + 单流） | 3D 全 | LLaVA-LLaMA 8B + CLIP | 720p 129 帧 | 50 |
| Wan 2.1 | 1.3B / 14B | DiT + 交叉注意力 | 3D 全 | umT5-XXL | 480p / 720p 81 帧 | 50 |
| Wan 2.2 | 5B（TI2V）/ 27B MoE（14B 激活） | 高噪 / 低噪两个专家 | 3D 全 | umT5-XXL | 720p 121 帧 | 50 |
| Mochi 1 | 10B | AsymmDiT | 3D 全 | T5-XXL | 480p 163 帧 | 64 |
| LTX-Video | 2B / 13B | DiT | 3D 全 | T5-XXL | 768×512 121 帧（实时） | 20～40 |

两个值得注意的新设计：Wan 2.2 的 **MoE 按噪声水平分专家**——高噪声阶段（定构图）和低噪声阶段（填细节）用不同的 14B 专家，每一步只激活一个，参数翻倍而计算不变，这和 LLM 的 MoE 按 token 路由完全不同，对推理系统反而更友好（每步只加载一个专家，切换点固定）；LTX-Video 的**极致压缩 VAE**把生成速度推到实时。

!!! interview "面试怎么答"
    被问"视频生成模型和图像模型的推理有什么不同"，从三个结构差异说起：潜变量多了时间轴（4×8×8 压缩、首帧单独编码），token 数是图像的几十倍；注意力是 3D 全注意力（新模型全部如此），计算量比分解式贵一个数量级以上但一致性好得多，所以注意力占一步的八成以上；3D VAE 是因果的，带来首帧条件、流式分段解码和图像视频共用。再补条件方式对成本的影响：拼在通道维免费，拼进 token 序列要付平方。最后点一句 Wan 2.2 按噪声分专家的 MoE——每步只激活一个、切换点固定，比 LLM 的 MoE 好调度。

## 练习

1. 用 `video_tokens` 和 `attn_flops` 算：Wan 2.1-14B 从 480p 81 帧到 720p 81 帧，3D 全注意力的计算量变成几倍？线性层呢？

??? success "参考答案"
    token 数从 32,760 到 75,600（2.3 倍），注意力 $\propto N^2$ 变 5.3 倍，线性层 $\propto N$ 变 2.3 倍。分辨率提高时注意力占比继续上升——720p 下已经超过七成。

2. 把 3D VAE 实验里的改动放到第 1 帧（`video2[:, :, 0] += 1.0`），哪些潜帧会变？这说明首帧条件的"影响范围"是什么？

??? success "参考答案"
    所有潜帧都变——因果意味着后面的帧依赖前面的。首帧作为条件时，它的信息会通过 VAE 的因果卷积和去噪网络的注意力影响整条视频，这正是图生视频"首帧决定一切"的来源；也说明首帧的 VAE 编码质量很重要。

3. 一个 I2V 请求带 1 张首帧，另一个请求带 3 张参考图（作为 token 拼进序列）。假设基础 token 数 75,600、每张参考图 4,096 个 token，两个请求的注意力成本差多少？服务该怎么对待它们？

??? success "参考答案"
    首帧拼在通道维，token 数不变；3 张参考图把序列变成 87,888，注意力成本 $\propto N^2$ 增加 35%。调度上它们是不同形状的请求：不能拼进同一个 batch（形状不同），耗时和显存也要分别估计；带参考图的请求应当有自己的队列和超时。

## 小结

- [x] 视频潜变量多了时间轴：$T' = 1 + (T-1)/4$，token 数是图像的几十倍，注意力占了一步的大头。
- [x] 新模型全部用 3D 全注意力：比分解式贵一个数量级以上，但 token 能看到任何时刻任何位置，一致性决定了这个选择。
- [x] 3D VAE 是因果的：首帧单独编码（帧数 $4k+1$）、流式分段解码、图像视频共用；压缩比是"去噪网络 vs VAE"之间的成本权衡（LTX 1:192 换实时）。
- [x] 3D RoPE 用绝对坐标，换帧数分辨率不改模型、序列并行不需通信；条件拼通道维免费、拼 token 序列付平方；Wan 2.2 按噪声分专家的 MoE 每步只激活一个。
