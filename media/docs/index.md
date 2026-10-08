# 图像与视频生成推理手册

<p class="lead">文生图、文生视频的模型和大语言模型用的是同一套 Transformer 积木，推理却是另一种形态：没有自回归、没有 KV Cache，一团噪声要过几十次同一个网络才变成一张图；视频模型的注意力序列动辄十万 token，一次生成的算力是一张图的几百倍。这本手册面向已经懂 LLM 推理的人，把扩散 / 流匹配模型的推理与服务从头讲一遍：pipeline 里到底跑了什么、算力和显存花在哪、少步 / 缓存 / 量化 / 多卡各自能省多少、视频为什么难、服务层怎么调度——每一章的代码都在 CPU 上用小配置真的跑过。</p>

## 为什么要学这些

- **推理岗的边界在扩大**：图像、视频生成的线上服务已经和 LLM 服务并列，xDiT、SGLang Diffusion、vLLM-Omni 这些项目把两边的技术栈合到了一起。懂 LLM 推理的人学这本，是把已有的功夫用到第二个战场。
- **它放大了 LLM 推理里的几个难题**：注意力的平方项在视频里是绝对瓶颈，显存峰值出现在 VAE 而不是主干，batch 的收益和 LLM 完全不同。这些差异逼着你把"为什么"想清楚。
- **优化空间大**：同一个模型，从朴素实现到用上蒸馏、缓存、量化、并行，端到端能差 10～50 倍。这是现在推理工程里收益最大的方向之一。

## 学完能做到

- 看到一个生成模型的配置（分辨率、帧数、VAE 压缩比、patch 大小、步数、CFG），能在纸上算出 token 数、单步 FLOP、总算力、显存峰值；
- 给一张卡上的 SDXL / FLUX / Wan 选对采样器、步数、shift、精度和 offload 策略，并解释每一项的代价；
- 读懂 TeaCache、PipeFusion、Ulysses、SVDQuant 这些方法在算什么、省在哪、什么时候会失效；
- 设计一个生成服务的调度：异构请求怎么分组、文本编码和 VAE 怎么流水、多 LoRA 怎么切换、SLO 怎么定；
- 面试里把"扩散模型推理和 LLM 推理有什么不同"讲成一个有数字的故事。

## 学习路线

| 篇 | 章节 | 你会得到 |
| --- | --- | --- |
| 基础 | [扩散与流匹配：推理视角](basics/diffusion-inference.md) → [一张图是怎么生成的](basics/pipeline-anatomy.md) → [去噪网络：从 UNet 到 DiT](basics/unet-to-dit.md) → [VAE 与潜空间](basics/vae-latent.md) → [采样器与调度器](basics/schedulers.md) | 生成的成本公式：步数 × CFG × 单步；token 数与显存峰值怎么算 |
| 性能 | [推理的算账](perf/accounting.md) → [显存与 offload](perf/memory.md) → [算子加速](perf/kernels.md) → [少步生成](perf/distillation.md) → [特征缓存](perf/caching.md) → [量化](perf/quantization.md) → [多卡并行](perf/parallel.md) | 每种加速手段省在公式的哪一项、代价是什么 |
| 视频 | [时空注意力与 3D VAE](video/architecture.md) → [视频推理的瓶颈](video/bottleneck.md) → [长视频与一致性](video/long-video.md) | 十万 token 的注意力怎么办 |
| 服务 | [生成服务的调度](serving/scheduling.md) → [多 LoRA / ControlNet](serving/lora-controlnet.md) → [引擎选型与部署](serving/deploy.md) → [评测与压测](serving/benchmark.md) → [面试与作品](serving/career.md) | 把单次生成变成一个线上服务 |

建议顺序读。已经熟悉扩散模型数学的读者可以从[一张图是怎么生成的](basics/pipeline-anatomy.md)开始；只关心服务层的读者至少要读完基础篇的前三章，否则后面的账算不清。

## 贯穿全书的主线

一张图（或一段视频）的生成成本是

$$
\text{成本} = \underbrace{\text{步数} \times \text{CFG 倍数}}_{\text{前向次数}} \times \underbrace{f(\text{token 数}, \text{参数量})}_{\text{单步成本}}
$$

每一章都在对这个式子的某一项动手：采样器和蒸馏减少前向次数，缓存让一部分前向变便宜，量化和算子优化压低单步成本，并行把单步摊到多张卡上，VAE 和 offload 决定显存能不能放下。服务层的工作，是在这个式子之上安排很多个请求。

!!! inference "和 LLM 推理的对照"
    读每一章时都可以问一句"LLM 里对应的是什么"：没有 KV Cache 对应什么、CFG 对应什么、特征缓存像不像投机解码、序列并行和 LLM 的张量并行差在哪。本书在每一章结尾的「怎么讲清楚」里都会把这条对照线收一次。

## 准备环境

本书的代码只需要 CPU 版 PyTorch 和 diffusers，所有模型都用最小配置随机初始化构建，**不需要下载任何权重**，也不需要 GPU：

```bash
cd media
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python diffusers transformers accelerate safetensors numpy pillow sentencepiece
.venv/bin/python tools/check_code.py          # 跑全部示例，输出与页面逐行比对
```

真实模型的参数量、FLOP、显存和时延在正文里都标明来源或"估算"；想在自己的卡上复现，各章会给出要跑的脚本和要看的指标。

!!! note "关于代码的验证"
    每一页里没有标题的 `python` 代码块都会被 `tools/check_code.py` 按顺序执行，紧跟其后的「输出」块必须和实际输出逐行一致；CI 里每次推送都会跑一遍。所以页面上的每一个数字，都是代码真的算出来的。
