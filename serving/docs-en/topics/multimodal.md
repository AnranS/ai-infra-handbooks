# Multimodal inference

<p class="lead">A vision-language model (VLM) puts a vision encoder in front of the language model: an image is first cut into small patches and encoded into a string of "image tokens", which are joined with the text tokens and handed to the language model. For an inference system this brings new costs and new questions: how many tokens an image becomes, how long the vision encoder takes, how image tokens' positions are encoded, and whether an image that appears several times can be reused. This chapter measures each of these on the real Qwen3.5-0.8B (natively multimodal: one model both reads text and looks at images), then discusses the corresponding designs in inference engines.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many tokens does a 1280×720 image become in Qwen3.5? And at double the resolution?
    2. What share of a request's prefill does the vision encoder's computation take?
    3. What is M-RoPE? How far does a 196-token image advance the position of the text after it?
    4. What is an encoder cache? Why does prefix caching need special handling for multimodal requests?

??? success "Answers (try first, then expand to compare)"
    1. Qwen3.5 uses 16×16 patches merged 2×2, so each image token covers about 32×32 pixels: 1280×720 has its height rounded to 704, giving 44×80 patches and 880 tokens after merging. At double the resolution, the token count becomes about 4×.
    2. The vision encoder has few parameters but takes a third to a half of prefill time (many image tokens, and the encoder must process every patch); it runs only once, in prefill.
    3. Multimodal RoPE: image tokens are encoded with three position components, time, height and width (text tokens have all three equal, reducing to ordinary RoPE). A 14×14 image of 196 tokens advances the following text's position by only 14 (continuing from the image's largest position plus one), not 196.
    4. An encoder cache caches the vision encoder's output, so the same image (in multi-turn conversations, or across chunked prefill) need not be re-encoded. Prefix caching must include a hash of the image content in its keys: in the token sequence an image is just a run of identical placeholder tokens, so different images give the same token sequence, and keying on tokens alone would hit another image's KV.

<!-- comic ../assets/comics/multimodal.webp is in Chinese; put it back once the English version exists -->

## How many tokens an image becomes {#一张图变成多少个-token}

Qwen3.5's vision encoder (the same architecture as Qwen3-VL) is a 12-layer ViT: the image is cut into 16×16-pixel patches, and after encoding, each 2×2 group of adjacent patches is merged into one token for the language model. So the number of image tokens is about pixel count / (32 × 32), and it varies dynamically with resolution (images are no longer all scaled to a fixed size): width and height are first rounded to multiples of 32, the total pixel count is then clamped to a range, and images that are too small get enlarged (at least 65536 pixels, about 256×256):

```python
import hashlib
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.set_num_threads(16)
path = "models/Qwen3.5-0.8B"
processor = AutoProcessor.from_pretrained(path)
image_pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
for w, h in [(224, 224), (448, 448), (896, 896), (1280, 720)]:
    out = processor(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[Image.new("RGB", (w, h), "white")],
                    return_tensors="pt")
    t, gh, gw = out["image_grid_thw"][0].tolist()
    n_tokens = (out["input_ids"] == image_pad).sum().item()
    print(f"{w:4d}×{h:<4d}：{gh}×{gw} 个 patch → 合并后 {n_tokens:4d} 个图像 token")
```

```text title="输出"
 224×224 ：16×16 个 patch → 合并后   64 个图像 token
 448×448 ：28×28 个 patch → 合并后  196 个图像 token
 896×896 ：56×56 个 patch → 合并后  784 个图像 token
1280×720 ：44×80 个 patch → 合并后  880 个图像 token
```

The 224×224 image was enlarged to 256×256 (below the pixel minimum), and 1280×720 had its height rounded to 704. At double the resolution, the token count becomes 4×. A high-resolution screenshot easily yields over a thousand tokens, longer than most text prompts; video is many frames, even more. Image tokens take prefill compute and KV Cache in the language model just like text tokens, so **controlling image resolution** (parameters such as `max_pixels`) is the most direct cost knob of a multimodal service.

Change the resolution to see the patch count, the token count and the encoder's compute:

<div class="aig-widget" data-widget="vision-tokens"></div>

## How long the vision encoder takes {#视觉编码器要算多久}

Load the full model, draw an image with text and a shape, ask a question, and time separately: the vision encoder alone, the full prefill (vision encoding + the language model processing all tokens), and generating the answer:

```python
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()
vision_params = sum(p.numel() for p in model.model.visual.parameters())
print(f"参数量：视觉编码器 {vision_params / 1e6:.0f}M，总计 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

image = Image.new("RGB", (448, 448), "white")
draw = ImageDraw.Draw(image)
draw.text((40, 40), "Prefill", fill="black", font=ImageFont.load_default(size=48))
draw.text((40, 120), "Decode", fill="black", font=ImageFont.load_default(size=48))
draw.ellipse([150, 250, 300, 400], fill="red")

def build(question):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor(text=[text], images=[image], return_tensors="pt")

inputs = build("图片里写了哪两个英文单词？下面画的是什么形状、什么颜色？")
with torch.no_grad():
    t0 = time.perf_counter()
    model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    vision_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model(**inputs)
    prefill_s = time.perf_counter() - t0
    generated = model.generate(**inputs, max_new_tokens=30, do_sample=False)
n_image = (inputs["input_ids"] == image_pad).sum().item()
print(f"输入 {inputs['input_ids'].shape[1]} 个 token（其中图像 {n_image} 个）")
print(f"视觉编码 {vision_s:.2f} s，完整 prefill {prefill_s:.2f} s：视觉编码约占 {vision_s / prefill_s:.0%}")
print("回答：", processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

```text
参数量：视觉编码器 101M，总计 853M
输入 226 个 token（其中图像 196 个）
视觉编码 0.22 s，完整 prefill 0.63 s：视觉编码约占 35%
回答： 图片里写了“Prefill”和“Decode”两个英文单词。下面画的是一个红色的圆形。
```

The model read the text and recognized the shape correctly (this is a small 0.8B model, and it misreads text drawn too large or too fancy). The vision encoder has only about 100 million parameters (the language part about 750 million), yet takes about a third of prefill time: it runs a full Transformer over every patch (before merging, 4 times as many as image tokens), with full attention in all 12 layers. Models with larger vision encoders have a higher share: Qwen2.5-VL-3B's vision encoder has about 670 million parameters and takes about half of prefill in the same measurement. This computation happens only once, in prefill; the decode phase does not involve the vision encoder at all.

## Positions of image tokens: M-RoPE {#图像-token-的位置m-rope}

Text token positions are one-dimensional; image tokens have two directions, height and width, and video adds time. Qwen's multimodal models use **M-RoPE** (multidimensional RoPE) since Qwen2-VL: each token has three position components (time, height, width), and RoPE's frequencies are split into three groups, each rotated by one of the components. Qwen2-VL and Qwen2.5-VL cut the frequencies into three contiguous ranges from high to low; Qwen3-VL and Qwen3.5 switched to **interleaved** assignment (time, height and width take turns at each frequency), so every direction covers the full band from high to low; Qwen3.5's config has `mrope_interleaved: true` and `mrope_section: [11, 11, 10]` (it rotates only the first 64 dimensions of each head, 32 frequencies in total). First, how the three components are computed:

```python
position_ids, _ = model.model.get_rope_index(inputs["input_ids"], inputs["mm_token_type_ids"],
                                             image_grid_thw=inputs["image_grid_thw"])
ids = inputs["input_ids"][0]
first, last = (ids == image_pad).nonzero()[0].item(), (ids == image_pad).nonzero()[-1].item()
show = lambda a, b: [tuple(position_ids[:, 0, i].tolist()) for i in range(a, b)]
print("图像之前的文字 (时间, 高, 宽)：", show(first - 2, first))
row = inputs["image_grid_thw"][0, 2].item() // 2                      # tokens per row after merging
print("图像的前 3 个 token：", show(first, first + 3), " 第二行开头：", show(first + row, first + row + 1))
print("图像的最后一个 token：", show(last, last + 1), " 图像之后的文字：", show(last + 1, last + 3))
```

```text title="输出"
图像之前的文字 (时间, 高, 宽)： [(2, 2, 2), (3, 3, 3)]
图像的前 3 个 token： [(4, 4, 4), (4, 4, 5), (4, 4, 6)]  第二行开头： [(4, 5, 4)]
图像的最后一个 token： [(4, 17, 17)]  图像之后的文字： [(18, 18, 18), (19, 19, 19)]
```

Text tokens have three equal components (reducing to ordinary one-dimensional RoPE); an image token's time component is fixed at the image's starting position, and its height and width components are the starting position plus its row and column in the 14×14 grid; text after the image continues from "the image's largest position + 1". So a 196-token image advances the following text's position by only 14, not 196. For inference engines this means **positions can no longer be taken directly from token indices**; three-dimensional positions must be computed by these rules and continued correctly during decode (vLLM and SGLang both have dedicated M-RoPE position computation).

## Reuse: the encoder cache and prefix caching {#复用encoder-cache-与前缀缓存}

In a multi-turn conversation, the same image reappears in every turn's request. The vision encoding depends only on the image itself, so it can be cached by a hash of the image content; this is the **encoder cache**. Wrap the model's `get_image_features` with a cache and ask two different questions about the same image:

```python
cache = {}
original = model.model.get_image_features
def cached_image_features(pixel_values, image_grid_thw=None, **kwargs):
    key = hashlib.sha256(pixel_values.numpy().tobytes()).hexdigest()   # key on the image's content
    if key not in cache:
        cache[key] = original(pixel_values, image_grid_thw, **kwargs)
    return cache[key]
model.model.get_image_features = cached_image_features

with torch.no_grad():
    for question in ["描述这张图。", "图中的圆是什么颜色？"]:
        batch = build(question)
        t0 = time.perf_counter()
        model(**batch)
        print(f"{question}  prefill {time.perf_counter() - t0:.2f} s（缓存中的图片数：{len(cache)}）")
```

```text
描述这张图。  prefill 0.78 s（缓存中的图片数：1）
图中的圆是什么颜色？  prefill 0.48 s（缓存中的图片数：1）
```

The second request skipped vision encoding, and prefill time fell by over a third, exactly the part vision encoding takes. Prefix caching applies to multimodal requests too, with a caveat: image tokens in the input sequence are all the same placeholder `<|image_pad|>`, so two different images produce **exactly the same token sequence**. Hashing tokens alone would wrongly use one image's KV for another. So inference engines include a hash of the image content as an extra key when computing block hashes (vLLM's `extra_keys` contain hashes of multimodal inputs; see [prefix caching](../engine/prefix-cache.md)).

## Designs in inference engines {#推理引擎中的设计}

- **Scheduling**: vision encoding needs compute too, so vLLM's scheduler keeps a separate compute budget for the encoder (`max_num_encoder_input_tokens`) and an encoder cache (`EncoderCacheManager`), deciding while scheduling prefill which images need encoding this step (see `_try_schedule_encoder_inputs` in [the vLLM walkthrough](../source/vllm.md));
- **Chunked prefill and images**: an image's tokens are best processed within a single chunk, or the encoder output must be kept across steps;
- **EPD disaggregation**: going further, split vision encoding (Encode) into its own instances, deployed and scaled separately from prefill and decode (vLLM's EC connector and SGLang's `disaggregation/encoder` both do this);
- **Preprocessing**: image decoding, resizing and patching are CPU work that can become a bottleneck at high concurrency, needing multiple processes or GPU preprocessing;
- **CUDA Graphs**: the vision encoder's input shape varies with image resolution, so it is usually not captured, or captured in buckets by resolution.

!!! source "Source code"
    - **vLLM**: multimodal input processing is in `vllm/multimodal/`; the encoder budget in the scheduler and `EncoderCacheManager` (`vllm/v1/core/encoder_cache_manager.py`); `mm_encoder_model_runner.py` on the worker side; EPD disaggregation in `vllm/distributed/ec_transfer/`; the extra multimodal keys of block hashes in `_gen_mm_extra_hash_keys` in `kv_cache_utils.py`.
    - **SGLang**: `srt/multimodal/` (per-model processors), `srt/managers/mm_utils.py` (caching and splicing image embeddings), `srt/mem_cache/multimodal_cache.py`, and `srt/disaggregation/encoder/`.

!!! interview "In an interview"
    "How does multimodal inference differ from text-only inference?": **token count** (dynamic resolution, with tokens growing linearly with pixels; the biggest cost knob) → **vision encoding** (only in prefill, but possibly half of it; reuse via an encoder cache; EPD disaggregation) → **position encoding** (M-RoPE; position no longer equals index) → **cache correctness** (image placeholders are identical, so prefix-cache hashes must include the image content) → **preprocessing** (a CPU bottleneck). Using this chapter's numbers to convey the magnitudes (1280×720 is about 900 tokens; vision encoding takes a third to a half of prefill) is very convincing.

## Exercises {#练习}

**1. Capping resolution.** Most images a service receives are phone screenshots (1170×2532). Roughly how many image tokens do they produce under Qwen3.5's rules? With `max_pixels` capped at 1280×32×32, how many? What does it cost?

??? success "Answer"
    1170 × 2532 / (32 × 32) ≈ 2893 tokens (actual rounding to multiples of 32 changes this slightly); capped at 1280 × 32 × 32 pixels, at most about 1280 tokens, cutting prefill and KV costs by more than half. The cost is a downscaled image in which small text may become illegible, lowering accuracy on OCR-like tasks. You can set different caps by task type, or have the client crop the region of interest first.

**2. Why doesn't decode need the vision encoder?** Does the decode phase of a multimodal request differ from a text-only one?

??? success "Answer"
    The visual information was encoded into image tokens during prefill, went through every layer of the language model, and was stored in the KV Cache (in hybrid models like Qwen3.5, the linear attention layers fold it into their fixed-size state). During decode, new tokens read this KV through attention and so "see" the image, with no need to run the vision encoder again. So decode is almost the same as for text only; the only differences are that positions continue by M-RoPE's rules and the KV Cache is longer (it includes the image tokens).

## Summary {#小结}

- [x] Image tokens ≈ pixel count / (32×32) (Qwen3.5 / Qwen3-VL: 16×16 patches merged 2×2; Qwen2.5-VL uses 28×28), varying dynamically with resolution; resolution is the first cost knob of a multimodal service.
- [x] The vision encoder has few parameters but takes a third to a half of prefill time; it runs only in prefill and can be reused through an encoder cache.
- [x] M-RoPE encodes image tokens with three-dimensional positions (time, height, width), with Qwen3-VL and Qwen3.5 interleaving the three frequency groups; text after an image continues from the maximum position plus one.
- [x] Prefix caching must include a hash of the image content in its keys; engines also need an encoder budget, an encoder cache, EPD disaggregation and preprocessing optimizations.
