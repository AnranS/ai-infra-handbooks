# Choosing an engine and deploying: diffusers, ComfyUI, xDiT and the inference frameworks

<p class="lead">The inference stack for text-to-image and text-to-video is moving from "research code plus scripts" towards real inference engines: diffusers is the model definitions and the reference implementation, ComfyUI is an executor that assembles a generation procedure into a graph, xDiT is an acceleration layer for multi-card parallelism, TensorRT takes the compilation route, and SGLang Diffusion and vLLM-Omni bring over an LLM service's scheduling and batching. This chapter sets out what each of them solves, what it is good at and how they relate, then gives the configuration for several typical deployment shapes (online interactive, offline bulk, on-device) and a checklist for before going live.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. At what level does diffusers sit, and ComfyUI? Can they be used together?
    2. Where does xDiT sit in the stack? How does it relate to diffusers?
    3. How much can TensorRT buy on a diffusion model? What does it cost?
    4. Where is the value in "an LLM inference framework doing generation", as with SGLang Diffusion and vLLM-Omni?
    5. How do the configurations differ between online interactive, offline bulk and on-device deployment?

??? success "Answers for the self-test (answer first, then open this)"
    1. diffusers is a model library and reference implementation: model definitions, schedulers, pipeline assembly and weight loading, the foundation of almost everything above it. ComfyUI is a graph executor: it wires the encoding, denoising, decoding, ControlNet and upscaling into a graph, aimed at workflows and non-programmers, reusing a great deal of diffusers' components underneath or implementing its own. They can be used together, and many ComfyUI custom nodes are wrappers around diffusers.
    2. The acceleration layer: it takes over the denoising network's forward pass and applies parallel strategies — guidance parallelism, sequence parallelism (USP), PipeFusion — to diffusers' models, with caching and quantization integrated; it does not deal with scheduling or serving.
    3. It compiles the UNet or DiT into an optimised engine (operator fusion, precision selection, fixed shapes), 1.5 to 2x on SDXL and more under FP8. The costs: long compilation times, fixed shapes (one engine per resolution), special handling for plugins (LoRA, ControlNet), and difficult debugging.
    4. It applies what LLM serving has already matured — request scheduling, pipelining by stage, batching, several models, an OpenAI-style API, observability — to generative models, and puts diffusion and LLMs in one engine (which autoregressive video models and unified multimodal models need); for one generation's kernel-level performance they mostly reuse xDiT's and diffusers' implementations.
    5. Online interactive: latency first, few steps or caching, several cards per request, previews, strict timeouts and cancellation. Offline bulk: throughput first, one request per card, deep queues, the full model and many steps are fine. On-device: memory and power first, INT4 / NF4, a small model, distillation, a CoreML or NPU backend.

## The stack's layers {#栈的分层}

<!-- i18n:diagram 1d0bbd6955 -->
```
serving        SGLang Diffusion · vLLM-Omni · an in-house service (queue, API, many models, observability)
               ───────────────────────────────────────────────────────────────────────────────────────
workflow       ComfyUI (a node graph) · an in-house pipeline orchestration
               ───────────────────────────────────────────────────────────────────────────────────────
acceleration   xDiT (parallelism) · TeaCache / FBCache (caching) · SageAttention · Nunchaku (W4A4) · TensorRT
               ───────────────────────────────────────────────────────────────────────────────────────
models         diffusers (model definitions, schedulers, pipelines, weights) · each vendor's official code
               ───────────────────────────────────────────────────────────────────────────────────────
runtime        PyTorch · CUDA · FlashAttention / cuDNN / cuBLAS
```

| Component | What it solves | What it does not | Suits |
| --- | --- | --- | --- |
| diffusers | the models, the schedulers, the pipelines, weight loading, basic offload and tiling | serving, scheduling, multi-card parallelism (limited) | everyone's starting point; research, prototypes, the base of something in-house |
| ComfyUI | turning a generation procedure into a visual graph; nearly every plugin in the ecosystem | multi-tenant serving, scheduling, parallelism | creative workflows, quick experiments, small deployments |
| xDiT | multi-card parallelism (USP, PipeFusion, guidance parallelism), caching, some quantization | request scheduling, an API | accelerating one generation on a video model |
| TensorRT and TRT-LLM's diffusion support | compilation, FP8 / INT8, peak performance at a fixed shape | flexibility, plugins, fast iteration | a high-throughput production service on a fixed model at a fixed resolution |
| Nunchaku / SVDQuant | W4A4 quantization kernels | everything else | consumer cards |
| SGLang Diffusion, vLLM-Omni | an LLM-style serving layer: scheduling, batching, several models, an API, one engine with LLMs | one generation's kernel-level performance (reused from below) | production serving, especially unified multimodal |

Two trends: **the acceleration layer is sinking** (xDiT's parallelism and TeaCache and the rest are gradually being integrated natively into diffusers and the inference frameworks), and **the serving layer is converging on the LLM frameworks** (diffusion and autoregressive generation scheduled in one engine).

## How to choose {#怎么选}

What decides it is the scenario, not which is "fastest":

```python
import unicodedata

SCENES = {   # how much each scenario needs each of the six capabilities (1 to 5)
    "研究 / 原型":     {"延迟": 1, "吞吐": 1, "灵活": 5, "插件": 4, "多卡": 1, "运维": 1},
    "创作工具（单机）": {"延迟": 3, "吞吐": 1, "灵活": 4, "插件": 5, "多卡": 1, "运维": 2},
    "在线交互服务":     {"延迟": 5, "吞吐": 3, "灵活": 2, "插件": 3, "多卡": 4, "运维": 5},
    "离线批量生成":     {"延迟": 1, "吞吐": 5, "灵活": 2, "插件": 2, "多卡": 2, "运维": 4},
    "视频生成服务":     {"延迟": 4, "吞吐": 3, "灵活": 2, "插件": 1, "多卡": 5, "运维": 5},
}
STACKS = {   # the capabilities each stack provides (1 to 5)
    "diffusers":  {"延迟": 2, "吞吐": 2, "灵活": 5, "插件": 4, "多卡": 1, "运维": 1},
    "ComfyUI":    {"延迟": 2, "吞吐": 2, "灵活": 4, "插件": 5, "多卡": 1, "运维": 2},
    "xDiT":       {"延迟": 4, "吞吐": 3, "灵活": 3, "插件": 2, "多卡": 5, "运维": 2},
    "TensorRT":   {"延迟": 5, "吞吐": 5, "灵活": 1, "插件": 2, "多卡": 2, "运维": 3},
    "LLM 式框架":  {"延迟": 4, "吞吐": 4, "灵活": 2, "插件": 2, "多卡": 4, "运维": 5},
}
def score(need, have):
    return sum(min(have[k], need[k]) for k in need) / sum(need.values())   # the fraction of the needed capability that is met

def pad(text, width):   # Chinese takes two columns, so align by display width
    shown = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - shown)

print((pad("场景", 18) + "".join(pad(name, 12) for name in STACKS)).rstrip())
for scene, need in SCENES.items():
    print((pad(scene, 18) + "".join(pad(f"{score(need, have):.0%}", 12) for have in STACKS.values())).rstrip())
```

```text title="output"
场景              diffusers   ComfyUI     xDiT        TensorRT    LLM 式框架
研究 / 原型       100%        92%         69%         54%         62%
创作工具（单机）  81%         94%         75%         62%         69%
在线交互服务      50%         55%         77%         73%         91%
离线批量生成      56%         62%         75%         88%         94%
视频生成服务      45%         50%         85%         70%         95%
```

(The scores are a matter of experience, meant to illustrate choosing by scenario rather than to be a league table.) How to read it: research and creation take what is flexible with many plugins (diffusers, ComfyUI); online and offline serving take a mature serving layer (an LLM-style framework) or the compilation route (TensorRT); a video service must have multi-card parallelism (xDiT, or a framework that integrates it).

## Three deployment shapes {#三种部署形态}

### Online interactive {#在线交互}

The user is waiting, so latency comes first:

- The model: a few-step distilled version (Turbo / Lightning / schnell) for the draft and the full model to refine; or the full model plus caching.
- One generation: FP8 plus compilation plus CUDA graphs plus SageAttention; several cards per request for video.
- The service: queue by expected duration, pool, cancel, preview (see [Scheduling a generation service](scheduling.md)).
- Timeouts and degradation: once the queueing passes a threshold, lower the resolution or the step count, or reject.
- Shapes: a fixed set of supported resolutions (both compilation and CUDA graphs require it).

### Offline bulk {#离线批量}

Nobody is waiting, so throughput and cost come first:

- One request per card, queues as deep as possible, no multi-card parallelism (the communication overhead is a net loss).
- The full model and the full step count, quality first; a batch is fine (several images from one prompt).
- Ways to save: cheap cards (L40S, a 4090 cluster), preemptible instances, scheduling overnight.
- Split the text encoding, the denoising and the VAE decode across different card pools, each one saturated.

### On-device {#端侧}

Memory, power and the package size come first:

- The model: distilled versions of SD 1.5 or SDXL, a small DiT (a model with a highly compressing VAE, like SANA), INT4 / NF4 weights.
- The backend: CoreML (Apple), an NPU runtime, ONNX Runtime, stable-diffusion.cpp in the llama.cpp ecosystem.
- 2 to 8 steps, a resolution of 512 to 768.
- The text encoder is the bulk of the memory: use a small one (CLIP rather than T5) or quantize it.

## A checklist for before going live {#上线前的检查清单}

| Check | How to verify | Common problem |
| --- | --- | --- |
| Reproducibility at a fixed seed | run the same request twice and compare the PSNR | a different batch composition, a different attention backend (see [Samplers and schedulers](../basics/schedulers.md)) |
| The compilation or engine for every resolution | warm up every one at startup and record the first latency | one was missed, and after going live the first request hangs for tens of seconds |
| The memory peak | stress the largest resolution plus the most plugins plus the VAE decode together | the denoising is fine and the decode OOMs |
| Scheduler instances isolated | compare against a solo run under concurrent requests | a multi-step scheduler's state crossed over |
| Resources released on cancellation | check the memory and the queue after a cancellation | the memory was not released, a worker hung |
| Plugin compatibility | run every LoRA and ControlNet once on the quantized and compiled build | merging broke the quantized weights; the alongside computation is not in the compiled graph |
| Quality regression | fix a set of prompts and seeds and compare LPIPS or look at them | a cache threshold or the quantization degraded a certain kind of image |
| Behaviour under overload | push to 1.5 times the capacity and watch the P99 and the rejection rate | long tasks block the queue |

!!! interview "How to answer in an interview"
    Asked about the technical choices for a text-to-image service, give the layers first: diffusers is the models and the reference implementation, ComfyUI is a workflow executor, xDiT and the like are the acceleration layer, TensorRT is the compilation route, and SGLang Diffusion and vLLM-Omni bring the LLM serving layer over. Then speak by scenario: research and creation want flexibility and plugins, online serving wants a mature serving layer or the compilation route, video must have multi-card parallelism. Then the three shapes' key points: online interactive puts latency first (few steps or caching, several cards per request, previews and cancellation, a fixed set of resolutions), offline bulk puts throughput first (one request per card, deep queues, cheap cards), on-device puts memory first (a small model, INT4, a dedicated backend). Finish with the two trends: the acceleration is sinking and the serving is converging on the LLM frameworks.

## Exercises {#练习}

1. Add a scenario to `SCENES` for "a unified multimodal service (one engine doing both text-to-image and VLM question answering)". Which dimensions does it need most? Which stack scores highest, and why?

??? success "Answer"
    It most needs operations (several models, one API, scheduling) and multi-card, with flexibility and plugins secondary. An LLM-style framework scores highest — they were designed for scheduling several models and several request types in the first place, and an autoregressive video or unified multimodal model needs to share the KV cache and the scheduler with the LLM. This is exactly why SGLang Diffusion and vLLM-Omni appeared.

2. An online service supports 512², 768² and 1024² with TensorRT. How many engines have to be prepared? And if 2 LoRAs and 1 ControlNet are added?

??? success "Answer"
    One engine per resolution (3); double that if guidance's batch of 2 has to be separate from a batch of 1. If a LoRA is merged into the weights, every combination of LoRAs needs its own engine (a combinatorial explosion), so either use a compilation that supports dynamic LoRA (the alongside computation as an engine input) or limit the combinations; a ControlNet is another network, its own engine, one per resolution. The engine count quickly becomes an operational burden, which is the compilation route's main cost.

3. To generate 100,000 SDXL 1024² images offline, with a choice of 8 H100s or 32 4090s, how do you work out which is the better deal?

??? success "Answer"
    Compute the cost per image: from [the accounting chapter](../perf/accounting.md), one SDXL image is about 3 seconds on an H100 and about 15 to 18 on a 4090; 8 H100s make about 9600 an hour and 32 4090s about 7000; multiply by each one's hourly price. An offline task does not care about latency, so only the money per image matters, and a consumer card like the 4090 is usually cheaper; note that a 4090's memory (24 GB) is enough for SDXL but FLUX needs FP8.

## Summary {#小结}

- [x] The stack has four layers: diffusers (the models and the reference implementation) → ComfyUI (workflows) → xDiT / caching / quantization / TensorRT (acceleration) → SGLang Diffusion and vLLM-Omni (an LLM-style serving layer); the acceleration is sinking and the serving is converging on the LLM frameworks.
- [x] Choose by scenario: research and creation want flexibility and plugins, online serving wants a mature serving layer or the compilation route, video must have multi-card parallelism.
- [x] The three shapes: online interactive puts latency first (few steps or caching, several cards per request, previews and cancellation, a fixed set of resolutions), offline bulk puts throughput first (one request per card, deep queues, cheap cards), on-device puts memory first (a small model, INT4, a dedicated backend).
- [x] Verify before going live: reproducibility, every resolution warmed up, the memory peak (including the VAE decode), scheduler isolation, release on cancellation, plugin compatibility, quality regression, behaviour under overload.
