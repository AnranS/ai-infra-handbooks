# The image and video generation inference handbook

<p class="lead">Text-to-image and text-to-video models are built from the same Transformer blocks as large language models, but their inference is a different shape: no autoregression, no KV cache, and a cloud of noise that has to pass through the same network dozens of times before it becomes a picture. A video model's attention sequence runs to a hundred thousand tokens and one generation costs hundreds of times what an image does. This handbook is for people who already understand LLM inference, and it covers inference and serving for diffusion and flow-matching models from the beginning: what actually runs inside a pipeline, where the compute and the memory go, how much few-step sampling, caching, quantization and multi-GPU parallelism each save, why video is hard, and how the serving layer schedules. Every chapter's code has really been run on a CPU with a minimal configuration.</p>

## Why learn this {#为什么要学这些}

- **An inference role's boundaries are widening**: production services for image and video generation now sit alongside LLM services, and projects like xDiT, SGLang Diffusion and vLLM-Omni have merged the two stacks. For someone who knows LLM inference, this book applies that skill to a second battlefield.
- **It amplifies several of LLM inference's problems**: attention's quadratic term is the outright bottleneck in video, the memory peak falls in the VAE rather than the backbone, and batching pays off entirely differently. Those differences force you to work out the why.
- **There is a lot of room to optimise**: for one model, going from a naive implementation to distillation, caching, quantization and parallelism together is a 10 to 50 times difference end to end. This is one of the highest-return directions in inference engineering today.

## What you will be able to do {#学完能做到}

- Given a generative model's configuration (resolution, frame count, VAE compression ratio, patch size, step count, classifier-free guidance), compute on paper the token count, the FLOP per step, the total compute and the memory peak.
- Pick the sampler, step count, shift, precision and offload policy for SDXL, FLUX or Wan on one card, and explain what each one costs.
- Read TeaCache, PipeFusion, Ulysses and SVDQuant and say what each computes, where it saves and when it fails.
- Design the scheduling of a generation service: how heterogeneous requests are grouped, how text encoding and the VAE are pipelined, how multiple LoRAs are switched, and how the service-level objectives are set.
- Turn "how does diffusion inference differ from LLM inference" into a story with numbers in an interview.

## The route through {#学习路线}

| Part | Chapters | What you get |
| --- | --- | --- |
| Basics | [Diffusion and flow matching from an inference point of view](basics/diffusion-inference.md) → [How one image is generated](basics/pipeline-anatomy.md) → [The denoising network: from UNet to DiT](basics/unet-to-dit.md) → [The VAE and the latent space](basics/vae-latent.md) → [Samplers and schedulers](basics/schedulers.md) | generation's cost formula: steps x guidance x per step; how to compute the token count and the memory peak |
| Performance | [The inference arithmetic](perf/accounting.md) → [Memory and offload](perf/memory.md) → [Kernel acceleration](perf/kernels.md) → [Few-step generation](perf/distillation.md) → [Feature caching](perf/caching.md) → [Quantization](perf/quantization.md) → [Multi-GPU parallelism](perf/parallel.md) | which term of the formula each technique saves on, and what it costs |
| Video | [Spatiotemporal attention and the 3D VAE](video/architecture.md) → [The bottleneck in video inference](video/bottleneck.md) → [Long video and consistency](video/long-video.md) | what to do about attention over a hundred thousand tokens |
| Serving | [Scheduling a generation service](serving/scheduling.md) → [Multiple LoRAs and ControlNet](serving/lora-controlnet.md) → [Choosing and deploying an engine](serving/deploy.md) → [Evaluation and load testing](serving/benchmark.md) → [Interviews and a portfolio](serving/career.md) | turning one generation into a production service |

Reading in order is recommended. A reader already familiar with diffusion's mathematics can start from [How one image is generated](basics/pipeline-anatomy.md); a reader who only cares about the serving layer still has to finish the first three chapters of the basics, or the arithmetic later will not add up.

## The thread running through the book {#贯穿全书的主线}

The cost of generating one image (or one video) is

$$
\text{cost} = \underbrace{\text{steps} \times \text{the guidance factor}}_{\text{forward passes}} \times \underbrace{f(\text{token count}, \text{parameters})}_{\text{cost per step}}
$$

Every chapter works on one term of this expression: samplers and distillation reduce the number of forward passes, caching makes some of them cheaper, quantization and kernel optimisation lower the cost per step, parallelism spreads one step over several cards, and the VAE and offload decide whether it fits in memory at all. The serving layer's work is to arrange many requests on top of this expression.

!!! inference "The comparison with LLM inference"
    While reading each chapter, ask what the LLM equivalent is: what corresponds to having no KV cache, what corresponds to classifier-free guidance, whether feature caching resembles speculative decoding, and how sequence parallelism differs from an LLM's tensor parallelism. Each chapter's "How to explain it" ties this thread off.

## Setting up {#准备环境}

The code in this book needs only the CPU build of PyTorch and diffusers; every model is built from a minimal configuration with random initialisation, so **no weights have to be downloaded** and no GPU is needed:

```bash
cd media
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python diffusers transformers accelerate safetensors numpy pillow sentencepiece
.venv/bin/python tools/check_code.py          # run every example and compare the output against the page line by line
```

Real models' parameter counts, FLOPs, memory and latency all carry a source or the word "estimate" in the text; each chapter names the script to run and the metrics to look at if you want to reproduce them on your own card.

!!! note "About verifying the code"
    Every untitled `python` block on a page is executed in order by `tools/check_code.py`, and the output block that follows it has to match the real output line by line; CI runs it on every push. So every number on a page was really computed by the code.
