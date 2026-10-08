# A video model's architecture: spatiotemporal attention and the 3D VAE

<p class="lead">A video model is not simply an image model with one more dimension: the latents gain a time axis, the token count multiplies by tens, and attention has to look along space and time at once; the VAE has to compress time too and be causal while doing it; and besides text, the conditions include the first frame, the last frame, a camera trajectory and audio. This chapter takes CogVideoX, HunyuanVideo, Wan and LTX-Video apart, works out what factorised spatiotemporal attention and full 3D attention each cost, builds and runs a 3D VAE and a video DiT from diffusers' minimal configurations, and lays the ground for the next chapter's bottleneck analysis.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What decides a video latent's shape? How many tokens is 81 frames of 720p in Wan?
    2. How do factorised spatiotemporal attention and full 3D attention each work? How far apart are they in computation? Why do new models all choose the latter?
    3. Why does a 3D VAE have to be causal? What does encoding the first frame separately mean for image-to-video?
    4. How does a video model's positional encoding handle three dimensions?
    5. How do image-to-video, first- and last-frame control, and camera control each feed their conditions in?

??? success "Answers for the self-test (answer first, then open this)"
    1. The time axis is compressed by the VAE's temporal ratio (usually 4) with the first frame computed on its own, the space by 8, and then packed by patch (usually 1x2x2): $T' = 1 + (T-1)/4$, $H' = H/8/2$, $W' = W/8/2$. Wan at 720p over 81 frames: 21 x 45 x 80 = 75,600 tokens.
    2. Factorised: spatial attention within each frame first ($T'$ sequences of length $H'W'$), then temporal attention at each spatial position ($H'W'$ sequences of length $T'$), costing $\propto T'(H'W')^2 + H'W'T'^2$. Full attention: every token together, $\propto (T'H'W')^2$. Full attention costs about $T'$ times more (about 20 times for Wan at 720p), but each token can see any position at any moment directly, which makes the motion and the consistency far better. Once the compute got cheap enough, every new model chose it.
    3. Causal: each latent frame depends only on the frames before it. The benefits: the first frame can be encoded into one latent frame on its own (hence frame counts of $4k+1$), and image-to-video uses that latent directly as the condition; decoding can proceed in streaming temporal segments without holding the whole video's activations in memory; and one VAE encodes both images and video.
    4. 3D RoPE: each token's position is a $(t, h, w)$ triple, and the head dimension is split into three ranges rotated by the three coordinates (FLUX's 2D RoPE plus a time dimension). It supports different resolutions and frame counts naturally, and under sequence parallelism each card only has to know its own tokens' coordinates.
    5. Image-to-video: the first frame is VAE-encoded and concatenated with the noisy latents along the channel dimension (with a mask channel marking which frames are given), or sent into attention as extra tokens; first- and last-frame control is the same with the mask marking both ends; camera control encodes the extrinsics into per-frame embeddings added to the timestep embedding, or into Plücker ray maps concatenated with the latents; audio-driven models inject audio features through cross-attention.

## The latents: the extra time axis {#潜变量多出来的时间轴}

A video VAE compresses $(T, H, W)$ pixels into $(T', H', W')$ latents, and the denoising network packs them into tokens by patch:

```python
def video_tokens(frames, h, w, ft=4, fs=8, pt=1, ps=2, latent_ch=16):
    T = 1 + (frames - 1) // ft                       # the first frame is its own, and after that every ft frames become one
    H, W = h // fs, w // fs
    tokens = (T // pt) * (H // ps) * (W // ps)
    return (T, H, W), tokens, latent_ch * pt * ps * ps   # each token's dimension = channels x the patch's volume

print(f"{'模型 / 设置':<30} {'潜变量 T×H×W':>14} {'token 数':>9} {'token 维度':>8} {'比 1024² 图像(4096)':>16}")
for name, frames, h, w in [("CogVideoX 480p 49 帧", 49, 480, 720), ("Wan 2.1 480p 81 帧", 81, 480, 832),
                           ("Wan 2.1 720p 81 帧", 81, 720, 1280), ("HunyuanVideo 720p 129 帧", 129, 720, 1280),
                           ("Wan 2.2 1080p 121 帧（假设）", 121, 1080, 1920)]:
    (T, H, W), n, dim = video_tokens(frames, h, w)
    print(f"{name:<30} {f'{T}×{H}×{W}':>14} {n:>9,} {dim:>8} {n / 4096:>15.0f}×")
```

```text title="output"
模型 / 设置                             潜变量 T×H×W   token 数 token 维度 比 1024² 图像(4096)
CogVideoX 480p 49 帧                  13×60×90    17,550       64               4×
Wan 2.1 480p 81 帧                   21×60×104    32,760       64               8×
Wan 2.1 720p 81 帧                   21×90×160    75,600       64              18×
HunyuanVideo 720p 129 帧             33×90×160   118,800       64              29×
Wan 2.2 1080p 121 帧（假设）            31×135×240   249,240       64              61×
```

The token count is 20 to 60 times an image's. In [the accounting chapter](../perf/accounting.md)'s formula the linear layers are $\propto N$ and attention is $\propto N^2$, so one video step's computation is tens to thousands of times an image's and **attention is the bulk of it**. That is the entire starting point for a video model's architecture.

## How attention sees time: factorised or full {#注意力怎么看时间分解还是全部}

Early video models (AnimateDiff, SVD, the first CogVideo) **factorised** attention into a spatial and a temporal step: spatial attention within each frame, then attention along time at each spatial position. The new generation (CogVideoX, HunyuanVideo, Wan, Mochi) all switched to **full 3D attention**: every token together. Their computation:

```python
def attn_flops(T, H, W, d, layers, mode):
    n_sp = (H // 2) * (W // 2)                       # the tokens per frame (patch 2)
    if mode == "分解（空间 + 时间）":
        per_layer = T * 4 * n_sp ** 2 * d + n_sp * 4 * T ** 2 * d
    else:                                            # full 3D attention
        per_layer = 4 * (T * n_sp) ** 2 * d
    return layers * per_layer

for name, frames, h, w, d, L in [("CogVideoX-5B 480p 49 帧", 49, 480, 720, 3072, 42), ("Wan 2.1-14B 720p 81 帧", 81, 720, 1280, 5120, 40)]:
    (T, H, W), n, _ = video_tokens(frames, h, w)
    f1, f2 = attn_flops(T, H, W, d, L, "分解（空间 + 时间）"), attn_flops(T, H, W, d, L, "3D 全注意力")
    linear = L * 24 * n * d * d
    print(f"{name:<24} 分解注意力 {f1 / 1e12:>7.0f} TFLOP，3D 全注意力 {f2 / 1e12:>7.0f} TFLOP（{f2 / f1:>4.0f}×），线性层 {linear / 1e12:>6.0f} TFLOP")
```

```text title="output"
CogVideoX-5B 480p 49 帧   分解注意力      12 TFLOP，3D 全注意力     159 TFLOP（  13×），线性层    167 TFLOP
Wan 2.1-14B 720p 81 帧    分解注意力     224 TFLOP，3D 全注意力    4682 TFLOP（  21×），线性层   1903 TFLOP
```

Full 3D attention is over an order of magnitude dearer than factorised, with thousands of TFLOP of attention in one step. So why do new models all choose it? Because with factorised attention **a token cannot see another position at another moment**, so motion consistency and an object's coherence as it crosses the frame have to emerge indirectly through stacked layers, and the result is clearly worse. Once compute got cheap, everyone chose the structurally cleanest option and left the cost to the inference system. That is the reason [the next chapter](bottleneck.md) exists.

Arrange the tokens into a T x H x W cube and switch between attention patterns to see which tokens one query can reach:

<div class="aig-widget" data-widget="video3d"></div>

The input and output shapes of diffusers' minimal video DiT, which differs from an image DiT only by the extra time dimension:

```python
import torch
from diffusers import CogVideoXTransformer3DModel

torch.manual_seed(0)
dit = CogVideoXTransformer3DModel(num_attention_heads=2, attention_head_dim=16, in_channels=4, out_channels=4,
                                  num_layers=1, sample_width=8, sample_height=8, sample_frames=9,
                                  patch_size=2, temporal_compression_ratio=4, max_text_seq_length=4,
                                  text_embed_dim=16, time_embed_dim=32, use_rotary_positional_embeddings=False).eval()
latent = torch.randn(1, 3, 4, 8, 8)                 # [B, T', C, H', W']: 3 latent frames (1 + 8/4)
text = torch.randn(1, 4, 16)
with torch.no_grad():
    out = dit(hidden_states=latent, encoder_hidden_states=text, timestep=torch.tensor([500])).sample
n_img = 3 * (8 // 2) * (8 // 2)
print(f"输入潜变量 {tuple(latent.shape)} → 视频 token {n_img} 个 + 文本 token 4 个 = 一起做 3D 全注意力，序列长度 {n_img + 4}")
print(f"输出 {tuple(out.shape)}：形状和输入一样，和图像 DiT 没有区别——区别只在 token 数")
```

```text title="output"
输入潜变量 (1, 3, 4, 8, 8) → 视频 token 48 个 + 文本 token 4 个 = 一起做 3D 全注意力，序列长度 52
输出 (1, 3, 4, 8, 8)：形状和输入一样，和图像 DiT 没有区别——区别只在 token 数
```

## The 3D VAE: compressing time too, and causally {#3d-vae时间也要压而且得是因果的}

A video VAE replaces an image VAE's 2D convolutions with 3D ones, compressing 4 frames into 1 along time and still 8 in space. The shapes from diffusers' minimal CogVideoX VAE:

```python
from diffusers import AutoencoderKLCogVideoX

torch.manual_seed(0)
vae = AutoencoderKLCogVideoX(in_channels=3, out_channels=3, block_out_channels=(8, 8, 8, 8), latent_channels=4,
                             layers_per_block=1, norm_num_groups=4, temporal_compression_ratio=4,
                             down_block_types=("CogVideoXDownBlock3D",) * 4, up_block_types=("CogVideoXUpBlock3D",) * 4).eval()
video = torch.randn(1, 3, 9, 32, 32)                # [B, C, T, H, W]: 9 frames = 1 + 2x4
with torch.no_grad():
    z = vae.encode(video).latent_dist.mode()
print(f"9 帧 {tuple(video.shape[2:])} → 潜变量 {tuple(z.shape[2:])}（T 从 9 到 {z.shape[2]}：首帧 + 每 4 帧一帧；空间 32 → {z.shape[3]}）")
```

```text title="output"
9 帧 (9, 32, 32) → 潜变量 (3, 4, 4)（T 从 9 到 3：首帧 + 每 4 帧一帧；空间 32 → 4）
```

The key design is **causality**: the convolutions along time look only backwards, with all of the padding at the front, so each output frame depends only on itself and the input frames before it. A three-layer causal 3D convolution verifies it: change the later frames and the earlier outputs do not move.

```python
import torch.nn as nn

class CausalConv3d(nn.Module):
    """时间维只在前面补零，所以第 t 帧的输出只看得到 ≤ t 的输入"""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(c_in, c_out, kernel_size=(k, 3, 3), padding=(0, 1, 1))
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, self.k - 1, 0))   # (W left and right, H top and bottom, T front and back): pad only the front of T
        return self.conv(x)

torch.manual_seed(0)
enc = nn.Sequential(CausalConv3d(3, 8), nn.SiLU(), CausalConv3d(8, 8), nn.SiLU(), CausalConv3d(8, 4)).eval()
x = torch.randn(1, 3, 9, 8, 8)
x2 = x.clone(); x2[:, :, 5:] += 1.0                 # change only frame 6 onwards
with torch.no_grad():
    d = (enc(x) - enc(x2)).abs().amax(dim=(0, 1, 3, 4))
print("改动第 6 帧以后，各帧输出的变化：", [f"{v:.2f}" for v in d.tolist()])
print("前 5 帧的输出完全不变——这就是因果；真实的视频 VAE 还要把归一化层也做成逐帧的，否则统计量会把未来泄露回去")
```

```text title="output"
改动第 6 帧以后，各帧输出的变化： ['0.00', '0.00', '0.00', '0.00', '0.00', '0.03', '0.06', '0.08', '0.13']
前 5 帧的输出完全不变——这就是因果；真实的视频 VAE 还要把归一化层也做成逐帧的，否则统计量会把未来泄露回去
```

Causality brings three things that matter for inference:

1. **The first frame can be encoded on its own.** It depends only on itself, so one image through the same VAE is the first latent frame, which image-to-video uses directly as the condition; this is also why the frame count is always $4k + 1$ (49, 81, 129).
2. **Decoding can stream in segments.** The earlier latent frames do not depend on the later ones, so the decoder can emit segment by segment with only one segment's memory (see the tiling in [The VAE and the latent space](../basics/vae-latent.md)). Wan's VAE turns this into an interface that emits frames as it decodes.
3. **One VAE for images and video.** $T = 1$ is an image; Wan and HunyuanVideo both use one VAE for image-to-video and text-to-video.

The mainstream 3D VAEs' parameters:

| VAE | Temporal x spatial compression | Latent channels | Causal | Notes |
| --- | --- | --- | --- | --- |
| CogVideoX | 4 x 8 x 8 | 16 | yes | causal 3D convolutions |
| HunyuanVideo | 4 x 8 x 8 | 16 | yes | with temporal tiled decoding |
| Wan 2.1 / 2.2 | 4 x 8 x 8 (4 x 16 x 16 for 2.2's TI2V) | 16 (48 in 2.2) | yes | a streaming decode interface |
| Mochi 1 | 6 x 8 x 8 | 12 | yes | more aggressive temporal compression |
| LTX-Video | 8 x 32 x 32 | 128 | yes | 1:192 compression, with the decoder denoising too |

The more aggressive the compression, the fewer the tokens and the faster the denoising, but the more generation the VAE has to take on. LTX-Video's 1:192 lets it produce 24 fps video in real time, at the price of the decoder imagining the detail. That is a clear design trade-off: **moving the cost from the denoising network to the VAE**.

## Position: RoPE in three dimensions {#位置三维的-rope}

A video token's position is a $(t, h, w)$ triple. 3D RoPE splits each head's dimensions into three ranges rotated by $t$, $h$ and $w$ respectively (FLUX's 2D RoPE plus a time dimension). Here is how the dimensions are allocated and why it suits sequence parallelism:

```python
def rope_3d_dims(head_dim=128, axes=(16, 56, 56)):
    """每个头 128 维：16 维给时间、56 维给高、56 维给宽（Wan / HunyuanVideo 的分法）"""
    assert sum(axes) == head_dim
    return {"时间 t": axes[0], "高 h": axes[1], "宽 w": axes[2]}

print("头维 128 的分配：", rope_3d_dims())
# under sequence parallelism each card holds only a segment of tokens, but their (t, h, w) coordinates are absolute, so each card computes its own rotations independently with no communication
tokens = [(t, h, w) for t in range(2) for h in range(2) for w in range(3)]
P = 3
shards = [tokens[r * len(tokens) // P:(r + 1) * len(tokens) // P] for r in range(P)]
for r, s in enumerate(shards):
    print(f"卡 {r} 持有 token 的坐标：{s}")
```

```text title="output"
头维 128 的分配： {'时间 t': 16, '高 h': 56, '宽 w': 56}
卡 0 持有 token 的坐标：[(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)]
卡 1 持有 token 的坐标：[(0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1)]
卡 2 持有 token 的坐标：[(1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)]
```

Time gets the fewest dimensions (16 of 128): the frame count is far smaller than the spatial size and needs less frequency resolution. There is an engineering benefit too: the coordinates are absolute, so **changing the frame count or the resolution does not change the model**, which is how one set of Wan weights runs 480p and 720p, 49 frames and 81 (the quality is still limited by the training distribution).

## Conditions beyond text {#条件文本之外}

| Task | How the condition enters | What it means for inference |
| --- | --- | --- |
| Text to video | text tokens through cross-attention (Wan) or joint attention (HunyuanVideo, CogVideoX) | the same as an image |
| Image to video | the first frame VAE-encoded and concatenated with the noisy latents along the channels, plus a mask channel; or as extra tokens | one extra VAE encode (a causal VAE encodes one frame quickly); the denoising network's input channels grow |
| First and last frames / keyframes | the same, with the mask marking which frames are given | the same |
| Video continuation / extension | the existing video encoded and placed earlier on the time axis as the condition | the conditioning frames go through the denoising network too, raising the token count |
| Camera control | the extrinsics encoded into per-frame embeddings added to the timestep embedding, or Plücker ray maps concatenated with the input | almost no extra cost |
| Audio-driven (digital humans) | audio features injected through cross-attention | one more encoder and one more cross-attention |
| Reference images / subject consistency | the reference images' tokens concatenated into the sequence (joint attention) | the token count grows and attention's cost rises |

How a condition enters decides its cost: **concatenating along the channels is nearly free, and concatenating into the token sequence pays attention's quadratic price**. In a service, a request where someone passed 3 reference images to the same model has a very different token count and cost from plain text-to-video, and scheduling has to treat it as a different shape (see [Scheduling a generation service](../serving/scheduling.md)).

## The mainstream models {#主流模型一览}

| Model | Parameters | Denoising network | Attention | Text encoder | Typical output | Steps |
| --- | --- | --- | --- | --- | --- | --- |
| CogVideoX-5B | 5B | DiT (expert AdaLN) | full 3D | T5-XXL | 480p over 49 frames | 50 |
| HunyuanVideo | 13B | MMDiT (dual plus single stream) | full 3D | LLaVA-LLaMA 8B and CLIP | 720p over 129 frames | 50 |
| Wan 2.1 | 1.3B / 14B | DiT with cross-attention | full 3D | umT5-XXL | 480p / 720p over 81 frames | 50 |
| Wan 2.2 | 5B (TI2V) / 27B mixture of experts (14B active) | two experts, high and low noise | full 3D | umT5-XXL | 720p over 121 frames | 50 |
| Mochi 1 | 10B | AsymmDiT | full 3D | T5-XXL | 480p over 163 frames | 64 |
| LTX-Video | 2B / 13B | DiT | full 3D | T5-XXL | 768x512 over 121 frames (real time) | 20 to 40 |

Two new designs worth noting: Wan 2.2's **mixture of experts split by noise level**, where the high-noise stage (setting the composition) and the low-noise stage (filling detail) use different 14B experts with only one active per step, doubling the parameters at unchanged computation. That is entirely unlike an LLM's mixture routed per token and is friendlier to an inference system (one expert loaded per step, with a fixed switching point). And LTX-Video's **extreme compression VAE** pushes generation to real time.

!!! interview "How to explain it"
    To explain how video inference differs from image inference, start from three architectural differences: the latents gain a time axis (4x8x8 compression with the first frame encoded separately) and the token count is tens of times an image's; attention is full 3D (as every new model is), over an order of magnitude dearer than factorised but far more consistent, so attention is over eighty percent of a step; and the 3D VAE is causal, which brings first-frame conditioning, streaming segmented decoding and one VAE for both images and video. Add how the conditioning affects cost: concatenating along the channels is free while concatenating into the token sequence pays the square. Finish with a word on Wan 2.2's noise-split mixture of experts: one expert active per step with a fixed switching point, far easier to schedule than an LLM's.

## Exercises {#练习}

1. Use `video_tokens` and `attn_flops`: going from 480p to 720p over 81 frames with Wan 2.1-14B, how many times does full 3D attention's computation grow? And the linear layers'?

??? success "Answer"
    The token count goes from 32,760 to 75,600 (2.3 times), so attention at $\propto N^2$ grows 5.3 times and the linear layers at $\propto N$ grow 2.3. Attention's share keeps rising with the resolution: it is already over seventy percent at 720p.

2. Move the change in the 3D VAE experiment to the first frame (`video2[:, :, 0] += 1.0`). Which latent frames change? What does that say about the first-frame condition's reach?

??? success "Answer"
    Every latent frame changes: causality means the later frames depend on the earlier. When the first frame is the condition, its information reaches the whole video through the VAE's causal convolutions and the denoising network's attention, which is where image-to-video's "the first frame decides everything" comes from; it also says the first frame's VAE encoding quality matters.

3. One image-to-video request carries 1 first frame and another carries 3 reference images (concatenated into the sequence as tokens). With a base token count of 75,600 and 4,096 tokens per reference image, how far apart are the two requests' attention costs? How should a service treat them?

??? success "Answer"
    The first frame is concatenated along the channels and the token count is unchanged; the 3 reference images make the sequence 87,888, raising attention's cost at $\propto N^2$ by 35%. In scheduling they are requests of different shapes: they cannot go into the same batch (different shapes), and their time and memory have to be estimated separately; requests with reference images should have their own queue and timeout.

## Summary {#小结}

- [x] Video latents gain a time axis: $T' = 1 + (T-1)/4$, the token count is tens of times an image's, and attention is the bulk of a step.
- [x] Every new model uses full 3D attention: over an order of magnitude dearer than factorised, but a token can see any position at any moment, and consistency settled the choice.
- [x] A 3D VAE is causal: the first frame is encoded on its own (frame counts of $4k+1$), decoding streams in segments, and one VAE serves images and video; the compression ratio is a cost trade-off between the denoising network and the VAE (LTX's 1:192 buys real time).
- [x] 3D RoPE uses absolute coordinates, so changing the frame count or resolution does not change the model and sequence parallelism needs no communication for it; conditioning along the channels is free while conditioning into the token sequence pays the square; Wan 2.2's noise-split mixture activates one expert per step.
