# How one image is generated: the pipeline dissected

<p class="lead">One line of <code>pipe("a cat")</code> produces a picture, but optimising its inference means knowing which models run inside that line, how many times each one runs, and which of them takes most of the time and the memory. This chapter does not use diffusers' pipeline class; it builds the text encoder, the denoising network and the VAE from minimal configurations on a CPU, strings the denoising loop together by hand, and counts each part's parameters, calls and share of the computation. It then applies the same arithmetic to SD 1.5, SDXL, SD3, FLUX and a video model.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which three kinds of model make up a text-to-image pipeline? How many times is each called?
    2. Why is it said that the denoising network takes essentially all of the computation while the VAE sets the memory peak?
    3. Why can the text encoder be computed in advance, or even cached?
    4. What decides the latent space's shape? How many tokens does a 1024x1024 image come to in SDXL and in FLUX?
    5. Once the pipeline is split into three stages, what can the serving layer do that a single pipeline cannot?

??? success "Answers for the self-test (answer first, then open this)"
    1. The text encoder (CLIP, T5), the denoising network (a UNet or a DiT) and the VAE. The text encoder runs once per prompt (and once more for the empty prompt under guidance), the denoising network runs steps x the guidance factor times, and the VAE decoder runs once (with the encoder running once as well for image-to-image).
    2. The denoising network is called dozens of times and each call is a complete forward pass; the VAE runs once, but it convolves at pixel resolution and its activations are 64 times the latent space's (an f8 compression is 8 times in each direction), so its intermediate activations when decoding 1024x1024 are larger than one denoising step's.
    3. Its input is the prompt alone, independent of the noise and the step count, and its output vectors are unchanged throughout the denoising. One prompt's encoding can be cached; the serving layer can also put the text encoder on another card or on the CPU and pipeline it with the denoising.
    4. The latent space is the pixel resolution divided by the VAE's compression ratio, with the channel count set by the VAE. SDXL: 1024/8 = 128, so 128x128x4, with no patching in the UNet and attention computed on feature maps from 32x32 to 128x128. FLUX: 128x128x16, packed into 2x2 to give 64x64 = 4096 tokens of 64 dimensions each.
    5. Batching by stage (the text encoder can take a large batch), placing the stages on different devices, caching the text encodings and intermediate results, previews (a small decoder producing a low-resolution image partway through the denoising), and overlapping a VAE decode with the next request's denoising.

## The three parts {#三个部件}

<!-- i18n:diagram 7f33b7ed42 -->
```
the prompt ──► the text encoder ──► the condition vectors ─┐
                                                           ▼
random noise ──► [the denoising network x N steps (twice each with guidance)] ──► clean latents ──► the VAE decoder ──► the image
```

![Figure: a text-to-image pipeline's three parts, with the text encoder computed once, the denoising network dozens of times and the VAE decoded once](../assets/figures/pipeline-anatomy.svg){.aig-svg}

Building them on a CPU with diffusers' classes but minimal configurations. The weights are randomly initialised (this chapter cares about shapes and computation, not about how good the picture is), so nothing has to be downloaded:

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

```text title="output"
文本编码器        0.05M 参数
去噪网络 UNet    0.99M 参数
VAE          0.66M 参数
这个玩具 VAE 的压缩比是 1/2（真实的 SD VAE 是 1/8），潜空间通道 4
```

## Running the denoising loop by hand {#手动跑一遍去噪循环}

What the pipeline class does is the few dozen lines below. What matters is counting how many times each part is called:

```python
calls = {"text": 0, "unet": 0, "vae": 0}

@torch.no_grad()
def generate(token_ids, steps=10, guidance=7.5, seed=0):
    # 1. text encoding: once conditional and once unconditional (the empty prompt)
    cond = text_encoder(token_ids).last_hidden_state
    uncond = text_encoder(torch.zeros_like(token_ids)).last_hidden_state
    calls["text"] += 2
    context = torch.cat([uncond, cond])                           # guidance's two paths concatenated into a batch of 2

    # 2. start from pure noise and denoise step by step over the scheduler's timesteps
    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size, generator=g)
    scheduler.set_timesteps(steps)
    latents = latents * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        x = scheduler.scale_model_input(torch.cat([latents, latents]), t)
        eps = unet(x, t, encoder_hidden_states=context).sample     # one forward pass at a batch of 2
        calls["unet"] += 1
        eps_u, eps_c = eps.chunk(2)
        eps = eps_u + guidance * (eps_c - eps_u)                   # the guidance extrapolation
        latents = scheduler.step(eps, t, latents).prev_sample

    # 3. the VAE decode: divide by scaling_factor to return to the VAE's latent scale
    calls["vae"] += 1
    image = vae.decode(latents / vae.config.scaling_factor).sample
    return image

ids = torch.randint(1, 1000, (1, 16))
img = generate(ids, steps=10)
print(f"输出图像 {tuple(img.shape)}，像素范围约 [{img.min():.1f}, {img.max():.1f}]（随机权重，不是一张真的图）")
print(f"调用次数：文本编码器 {calls['text']}，去噪网络 {calls['unet']}（10 步，每步 batch=2 的一次前向），VAE 解码 {calls['vae']}")
```

```text title="output"
输出图像 (1, 3, 32, 32)，像素范围约 [-1.2, 1.2]（随机权重，不是一张真的图）
调用次数：文本编码器 2，去噪网络 10（10 步，每步 batch=2 的一次前向），VAE 解码 1
```

The three parts are called 2, N and 1 times. That structure sets the optimisation's priorities: **one forward pass of the denoising network times N is everything**, while the text encoder and the VAE run once each but have problems of their own, which the arithmetic below shows.

## The arithmetic: each part's share of the computation {#算账每个部件各占多少计算量}

Running each part once with PyTorch's FLOP counter and weighting by the call counts:

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
f_unet = flops(lambda: unet(x2, torch.tensor([500]), encoder_hidden_states=ctx))     # a batch of 2: one guided step
f_vae = flops(lambda: vae.decode(z))

steps = 10
total = 2 * f_text + steps * f_unet + f_vae
print(f"{'部件':<12} {'一次前向 MFLOP':>14} {'调用次数':>8} {'占比':>7}")
for name, f, n in [("文本编码器", f_text, 2), ("去噪网络（CFG）", f_unet, steps), ("VAE 解码", f_vae, 1)]:
    print(f"{name:<12} {f / 1e6:>14.1f} {n:>8} {n * f / total:>7.1%}")
```

```text title="output"
部件               一次前向 MFLOP     调用次数      占比
文本编码器                   0.5        2    0.0%
去噪网络（CFG）             325.9       10   90.6%
VAE 解码                336.4        1    9.4%
```

In this toy, one VAE decode is the same order as one denoising step, because it convolves at pixel resolution. In a real model the ratio is more extreme: SDXL's VAE takes about 2.5 TFLOP to decode a 1024x1024 image, comparable to one UNet step (about 3 TFLOP, doubled by guidance), and its **activation memory peak** is several times one denoising step's, which is why tiled decoding exists (see [The VAE and the latent space](vae-latent.md)).

## The arithmetic for real models {#真实模型的账}

Replacing the three parts with real configurations, estimated from the published parameter counts and the computation per step (the numbers are estimates and vary with the implementation, the resolution and the step count):

```python
# the FLOP of one denoising forward pass (estimated): a DiT as 2 x parameters x tokens (the linear layers) plus 4N²d of attention per layer,
# a UNet worked back from measured latency; the text encoder and the VAE run once each and are not counted here
MODELS = [
    # name,                     text encoder,                      denoising network, TFLOP per pass, steps, guidance
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

```text title="output"
模型                         去噪网络            一次前向 TFLOP     前向次数     去噪总量 PFLOP
SD 1.5 (512²)              UNet 0.86B             1.2       50           0.06
SDXL                       UNet 2.6B             12.0       60           0.72
SD3-medium                 MMDiT 2B              20.0       56           1.12
FLUX.1-dev                 MMDiT 12B             90.0       28           2.52
Wan 2.1-14B (720p 81 帧)    DiT 14B            6,900.0      100         690.00
```

A few readings:

- **SDXL is about 0.7 PFLOP per image.** One H100's dense bf16 throughput is about 1 PFLOP/s, so in theory under a second; in practice 2 to 4 seconds, because the UNet's many convolutions, normalisations and small operations are memory-bound and the utilization is only twenty or thirty percent (see [The inference arithmetic](../perf/accounting.md)).
- **One FLUX forward pass is over 7 times SDXL's**, and even without guidance the total is over 3 times SDXL's. The per-step cost of going from 2.6B to 12B parameters is not offset by dropping guidance.
- **Video is three orders of magnitude away.** Wan 14B generating 5 seconds of 720p video takes about 700 PFLOP, nearly 1000 times SDXL's: the token count (frames x height x width) enters both the linear layers' $N$ and attention's $N^2$ (see [The bottleneck in video inference](../video/bottleneck.md)). One card takes over twenty minutes, which is why multi-GPU parallelism is the norm for video inference.
- The text encoder looks large (T5-XXL at 4.7B), but it runs once on an input of tens to hundreds of tokens and its computation is negligible; its real cost is **memory**: 9.5 GB in bf16, which together with FLUX's 24 GB of weights will not fit on a 16 GB card. Hence offloading it to the CPU after encoding, and storing T5 in fp8 or NF4 (see [Memory and offload](../perf/memory.md)).

## What splitting it open allows {#拆开之后能做什么}

Seeing the pipeline as three stages rather than one black box immediately gives the serving layer several moves:

| Move | How | What it gains |
| --- | --- | --- |
| Caching the text encoding | cache the result by the prompt's hash | a repeated prompt (batch generation, a new seed) skips the text encoder entirely |
| Batching by stage | large batches for the text encoder and the VAE, grouping the denoising by shape | the three stages' best batch sizes differ |
| Pipelining the stages | request B starts denoising while request A decodes its VAE | the GPU never idles, especially at high resolutions where the decode is slow |
| Previews | a few-megabyte decoder such as TAESD produces a low-resolution image partway through | the user sees roughly what is coming after a few steps and can cancel early |
| Heterogeneous placement | T5 on another card or the CPU, the VAE on another card | the large card is left to the denoising network alone |

These are developed in the chapter on scheduling a generation service. All of them rest on this chapter's accounting of who is called how often, who takes the compute and who takes the memory.

!!! interview "How to explain it"
    To explain where the time goes in a text-to-image pipeline, give the structure first: the text encoding once, the denoising N times, the VAE once; then the proportions: the denoising network is over 90% of the computation and the main battlefield. Then point out the two things easily missed: the VAE decode's activation peak can be several times one denoising step's (hence tiling), and a large text encoder like T5 has negligible compute but non-negligible memory (hence offloading or quantization). Finish by saying why video differs: the token count enters attention's quadratic term and the total computation is two or three orders of magnitude above an image's.

## Exercises {#练习}

1. Set `guidance` to 1.0 in `generate` (equivalent to no guidance) and rewrite it to run only the conditional path. What do the call counts and batch sizes become? How much FLOP is saved?

??? success "Answer"
    The text encoder runs once (the empty prompt no longer needs encoding), the denoising network still runs N times but at a batch of 1 rather than 2, and the VAE is unchanged. The denoising network's FLOPs halve and the total nearly halves. FLUX.1-dev bakes the guidance into the model, so this is its natural shape.

2. Measure `vae.encode` and `vae.decode` separately with `FlopCounterMode`. Which is more expensive? Why is the extra cost of image-to-image usually small?

??? success "Answer"
    The decoder is more expensive than the encoder (the decoder's upsampling has more channels at higher resolution; SD's VAE decoder is about 50M parameters against the encoder's 34M). Image-to-image only adds one encode and then starts denoising from an intermediate timestep, skipping some of the steps, so its total cost is often lower than text-to-image's.

3. SDXL generating a 2048x2048 image: how large is the latent space? How many times the 1024² token count does the UNet's lowest-resolution attention layer have? How many times the computation?

??? success "Answer"
    The latent space is 256x256x4. The UNet computes attention on feature maps downsampled by 2 and by 4, and the lowest-resolution layer has 64x64 = 4096 tokens, 4 times the 32x32 = 1024 at 1024². Attention's computation grows with the square of the token count, so 16 times. This is why high-resolution inference uses tiling or generates low and upscales, rather than simply doubling the resolution.

## Summary {#小结}

- [x] Text to image is one text encoding plus N denoising passes (at twice the batch each under guidance) plus one VAE decode; that structure sets the optimisation's priorities.
- [x] The denoising network is over 90% of the computation; the VAE sets the activation memory peak; a large text encoder's compute is negligible but its memory is not.
- [x] SDXL is about 0.7 PFLOP per image, FLUX about 2.5 PFLOP, and 5 seconds of 720p video about 700 PFLOP. Video is three orders of magnitude above images because the token count enters the linear layers' $N$ and attention's $N^2$.
- [x] Only by splitting the pipeline into three stages can the serving layer cache, batch by stage, pipeline and preview.
