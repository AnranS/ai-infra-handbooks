# Multimodal and SGLang Diffusion: multimodal/ and the separate multimodal_gen/

<p class="lead">Multimodal took two entirely different routes in SGLang. Vision-language models have been in the runtime since the first version's LLaVA: an image becomes placeholder tokens and the visual features are filled in during the forward pass, and in June 2025 their preprocessors were tidied into a <code>multimodal/</code> directory. Image and video generation, by contrast, entered the repository on 6 November 2025 in one commit of 249 files and 64 thousand lines — <code>python/sglang/multimodal_gen/</code> is a runtime of its own, with its own scheduler, pipelines and model layers, sharing only the kernels, the distributed layer and the platform layer with the LLM runtime. This chapter covers why the two routes differ, and the trade-offs in keeping two runtimes in one repository.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What steps does a vision-language model's input go through in SGLang? Which step do the processors in `multimodal/processors/` handle?
    2. Why can a diffusion model's inference not reuse the LLM's scheduler and KV pool?
    3. What do `multimodal_gen/` and `srt/` share, and what do they not?
    4. And what is `dllm/`? Is it the same thing as `multimodal_gen/`?

??? success "Answers for the self-test (answer first, then open this)"
    1. A processor (one per model family) turns an image, video or audio into tensors and a run of placeholder tokens and computes the hash used for caching; the scheduler sees only the token sequence (the placeholders can hit the prefix cache); and during the forward pass the vision encoder computes the features and fills them into the language model's input embeddings at the right positions. The processor handles the first step, possibly in its own process or a thread pool.
    2. A diffusion model has no KV cache and no token-by-token generation: one request is a denoising loop of a fixed number of steps, each step putting the whole image (or a whole video's latents) through the network; the unit of batching is an image, not a token; and there are the guidance doubling, the text encoder and the VAE decode as further stages. Continuous batching, prefix caching and speculative decoding do not apply, and what is needed is scheduling and caching organised by step count and resolution (feature caching, for instance).
    3. Shared: sgl-kernel's operators, `distributed/` and the platform layer, the OpenAI-compatible API style, the `launch_server` entry point and the CLI, and the approach to CUDA graphs. Not shared: the scheduler, the memory pool, the radix tree and the model layers (diffusion's DiT, UNet and VAE live in `multimodal_gen/runtime/models` and `layers`).
    4. `dllm/` is srt's support for "diffusion language models" (LLMs that generate text by diffusion, such as block diffusion), which is still text generation reusing the LLM runtime; `multimodal_gen/` is image and video generation, an entirely different runtime.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/multimodal-diffusion.webp is in Chinese; put it back once the English version exists -->

## Vision-language models: the processors directory {#视觉语言模型处理器目录}

```bash title="multimodal-growth.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/multimodal | head -1 | cut -c1-96
for t in v0.5.0rc0 "$REF"; do
  printf '%-11s multimodal/ %3d 个文件，processors/ %3d 个\n' "$t" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/multimodal | grep -c '\.py$')" "$(git ls-tree -r --name-only "$t" -- python/sglang/srt/multimodal/processors | grep -c '\.py$')"
done
echo "今天 processors/ 里的模型族：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt/multimodal/processors | grep '\.py$' | sed 's|.*/||; s|\.py||' | grep -v '__init__\|base_processor' | tr '\n' ' ' | cut -c1-230)"
```

```text title="output"
2025-06-27  ce3a3e8783  Move multimodal processors into a separate folder (#7581)
v0.5.0rc0   multimodal/  19 个文件，processors/  18 个
29f6d408c0  multimodal/  95 个文件，processors/  64 个
今天 processors/ 里的模型族：bailing_mm clip cohere2_vision cosmos3_edge deepseek_ocr deepseek_v41 deepseek_vl_v2 diffusion_gemma dots_note_omni flatten_runner preprocess v2core video_qa_flattener dots_vlm ernie45_vl executor gemma3 gemma3n gemma4 gemma4_unif
```

[Chapter seven](../service/api-multimodal.md) covered the pipeline of "preprocess first, a placeholder in the middle, the features last". #7581 of 27 June 2025 moved the per-model preprocessors scattered through `managers/` into `multimodal/processors/`, a base class plus one file per model family: Qwen-VL, Llama 4, InternVL, Gemma 3, Pixtral, Kimi-VL, MiniCPM, Phi-4 multimodal, the speech models… The base class fixes the steps ("take the multimodal data out of the request → preprocess → produce the placeholder tokens and the hash") and a subclass fills in the model-specific details. `multimodal_cache.py` caches the visual features by hash, and `mm_utils.py` keeps the tools handed down from the LLaVA days. H2 2025 added an extension: `disaggregation/encoder/` ([chapter 17](../scale/pd.md)) splits the visual encoding into its own service too.

## Image and video generation: one commit of 64 thousand lines {#图像与视频生成一次-64-万行的提交}

```bash title="diffusion-import.sh"
git show --stat=200 --format='%ad  %an  %s' --date=short 7bc1dae095 | sed -n '1p;$p' | cut -c1-96
echo "-- 按目录："; git show --stat=200 --format= 7bc1dae095 | awk '{print $1}' | grep '^python/sglang/multimodal_gen/' | cut -d/ -f4 | sort | uniq -c | sort -rn | head -8 | awk '{printf "   %3d  %s\n", $1, $2}'
git log --date=short --format='%ad  %h  %s' 29f6d408c0 | grep -i 'diffusion announcement' | cut -c1-96
```

```text title="output"
2025-11-06  Mick  WIP: initial multimodal-gen support (#12484)
 249 files changed, 63750 insertions(+), 11 deletions(-)
-- 按目录：
   168  runtime
    47  configs
    12  test
     5  csrc
     3  docs
     2  third_party
     1  utils.py
     1  envs.py
2025-11-07  32f7982800  sglang diffusion announcement (#12856)
```

#12484 "WIP: initial multimodal-gen support" of 6 November (the main line of the Q4 2025 roadmap, issue #12799), with the announcement in #12856 the next day. This is not a feature added inside `srt/` but a complete diffusion inference framework (developed outside SGLang beforehand) merged into the repository: `runtime/` has its own `managers`, `scheduler_client`, `pipelines`, `layers`, `models`, `loader`, `distributed`, `platforms`, `disaggregation` and `realtime`, `configs/` holds each model's configuration, and `csrc/` its dedicated kernels. The README's positioning:

```markdown title="python/sglang/multimodal_gen/README.md @ 29f6d408c0 L6-14" linenums="6"

SGLang diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new models and optimizations.

## Key Features

SGLang Diffusion has the following features:
  - Broad model support: Wan, FastWan, FLUX, Qwen-Image / Qwen-Image 2.1, LongCat-Image, Z-Image, Anima, Ideogram 4, Krea-2, Cosmos3, LTX-2/LTX-2.3/LTX-2.5, MiniMax-H3, FastH3, VDN-H3, LingBot Video MoE, LingBot World, SANA-Video/SANA-WM, JoyEcho, MOVA, GLM-Image, ERNIE-Image, Hunyuan3D, and more
  - Fast inference speed: empowered by optimized `sgl-kernel` kernels, scheduler/runtime improvements, caching acceleration, and native diffusion hot-path optimizations
  - Ease of use: OpenAI-compatible api, CLI, python sdk, and a [ComfyUI plugin](apps/ComfyUI_SGLDiffusion/README.md)
```

Its size today:

```bash title="diffusion-size.sh"
REF=${REF:-29f6d408c0}
echo "multimodal_gen/ 共 $(git ls-tree -r --name-only "$REF" -- python/sglang/multimodal_gen | wc -l) 个文件，.py $(git ls-tree -r --name-only "$REF" -- python/sglang/multimodal_gen | grep -c '\.py$') 个"
echo "runtime/ 一级目录：$(git ls-tree --name-only "$REF" python/sglang/multimodal_gen/runtime/ | sed 's|.*/||' | grep -v '\.py$' | tr '\n' ' ')"
echo "共享的入口：$(git show --stat=200 --format= 7bc1dae095 | awk '{print $1}' | grep -E '^python/sglang/(launch_server|cli)' | tr '\n' ' ')"
```

```text title="output"
multimodal_gen/ 共 1198 个文件，.py 1151 个
runtime/ 一级目录：breakable_cuda_graph cache disaggregation distributed entrypoints layers loader managers models observability pipelines pipelines_core platforms post_training postprocess realtime server_args utils vla weights 
共享的入口：python/sglang/cli/__init__.py python/sglang/cli/generate.py python/sglang/cli/main.py python/sglang/cli/serve.py python/sglang/launch_server.py 
```

`runtime/`'s directory names look much like `srt/`'s (managers, layers, models, loader, distributed, platforms, disaggregation), which says it borrowed the same way of organising; but the content is diffusion's: `pipelines/` is the per-model generation pipeline (text encoding → the denoising loop → the VAE decode), `cache/` is feature caching (neighbouring denoising steps' activations are alike and can be skipped), `breakable_cuda_graph/` is interruptible graph capture, `realtime/` is real-time generation, and `vla/` is vision-language-action models. The image and video generation handbook's chapters on [the inference arithmetic](media://perf/accounting/), [feature caching](media://perf/caching/) and [multi-GPU parallelism](media://perf/parallel/) cover the principles behind exactly these mechanisms.

![Figure: one repository, two runtimes](../assets/figures/sgl-two-runtimes.svg){.aig-svg}

## Why start separately {#为什么另起炉灶}

Putting diffusion models into `srt/`'s scheduler was not unthought of, but the two workloads differ at every layer:

| | LLM (srt) | Diffusion (multimodal_gen) |
| --- | --- | --- |
| The unit of generation | one token, looping until it ends | one image or video, a fixed number of steps |
| Batching | continuous batching by a token budget | batched by image, with resolutions matching or bucketed |
| State | a KV cache growing with the length | latents of a fixed size, no KV |
| Cache reuse | the prefix cache (the radix tree) | feature caching (across steps), condition caching (across requests) |
| Parallelism | TP / EP / DP attention / PD | sequence parallelism, guidance parallelism, a pipeline by stage |
| The bottleneck | decode's memory traffic and launch overhead | the denoising network's compute, the VAE's memory peak |

What is shared is the layers beneath: sgl-kernel's attention and GEMM, `distributed/`'s process groups, the platform layer's hardware support (the README lists NVIDIA, AMD, Intel XPU, Ascend, Apple MPS and Moore Threads), the approach to CUDA graphs, and the OpenAI-style interface with the `launch_server` entry point. That is a more realistic boundary for reuse than "reuse the scheduler".

`dllm/` (#12588 "Initial block diffusion language model support", 2025-11-26) is the opposite: diffusion-style **text** generation still runs on `srt/`'s runtime with only a different decoding loop — it reuses the KV pool and the scheduler and adds a set of algorithm files.

## Design trade-offs {#设计取舍}

- **Two runtimes in one repository.** The kernels, the CI, the releases and the community are shared; the price is a larger repository and CI triggered by directory (#12940 "skip full CI suite for multimodal_gen changes").
- **Processors per model family.** As with the tool-call parsers ([chapter 20](entrypoints.md)), there is no common format and they have to be written one at a time, with the base class fixing the steps.
- **The visual encoding can be separated.** An encoder service (E/P/D) keeps a large vision tower off the decode nodes' memory.

## What happened afterwards {#后来怎么样了}

- `multimodal/` has 95 files at the baseline commit; the hash cache for multimodal requests, video frame extraction and audio input were filled in gradually.
- `multimodal_gen/` has 1198 files at the baseline commit with over twenty subdirectories under `runtime/`, and the models supported grew from Wan, FLUX and Qwen-Image to more than a dozen families, with a ComfyUI plugin and a Python SDK.
- In 2026 both sides are connecting Rust's multimodal preprocessing (`rust/sglang-mm`, [chapter 21](gateway.md)).

## Exercises {#练习}

**1. One processor.** Read the baseline commit's `multimodal/processors/base_processor.py`, list the methods a subclass has to implement, then see how `qwen_vl.py` computes an image's placeholder token count.

??? success "A way to approach it"
    The base class has an entry point like `process_mm_data_async` and steps such as `load_mm_data`; Qwen-VL divides the image's resolution by the patch size and then by the merge factor to get the token count.

**2. The boundary of sharing.** Use `git grep -l 'from sglang.srt' 29f6d408c0 -- python/sglang/multimodal_gen | wc -l` to count how many of srt's modules the diffusion runtime imports, and classify them by submodule.

??? success "A way to approach it"
    Mostly `distributed`, `layers` (attention, normalisation, quantization), `utils`, parts of `server_args` and `platforms`; never `managers/scheduler.py` or `mem_cache/`.

**3. Splitting the CI.** Read #12940's workflow changes and explain how the repository keeps a change under `multimodal_gen/` from triggering every LLM test.

??? success "A way to approach it"
    Trigger conditions filtered by path, and a separate workflow for the diffusion part.

!!! interview "How to answer in an interview"
    "Should an inference framework support both LLMs and diffusion models?" — Answer with SGLang's approach: the two workloads differ in the unit of generation, the batching, the state, the caching and the parallelism, so they are two runtimes; what is shared is the kernels, the distributed layer, the platform layer and the interface style. Being able to say that `multimodal_gen/` is an independent framework merged in in 2025-11 while `dllm/` is diffusion-style text generation reusing the LLM runtime shows you can tell the two apart.

## Summary {#小结}

- [x] Vision-language models: the preprocessors were gathered per model family into `multimodal/processors/` (#7581, 2025-06), with the pipeline "preprocess → placeholder → fill in the features" unchanged.
- [x] Image and video generation: `multimodal_gen/` was merged in one commit of 249 files and 64 thousand lines (#12484, 2025-11-06) as a separate runtime, sharing only the kernels, the distributed layer, the platform layer and the interface style.
- [x] `dllm/` is diffusion-style text generation reusing the LLM runtime and is not the same thing as `multimodal_gen/`.
