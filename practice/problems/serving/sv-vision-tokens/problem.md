---
title: 一张图片会变成多少个 token
chapter: topics/multimodal.md
difficulty: 简单
tags: [多模态, 动态分辨率, 估算]
---
Qwen 的多模态模型支持任意分辨率的图片：先把图片缩放到长宽都是 `factor` 的倍数（factor = patch 边长 × 2，因为编码后还要 2×2 合并：Qwen2-VL、Qwen2.5-VL 的 patch 是 14，factor 28；Qwen3-VL、Qwen3.5 的 patch 是 16，factor 32），像素总数限制在 `[min_pixels, max_pixels]`，尽量保持原来的宽高比。实现：

1. `smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=28*28*1280)`，规则（与官方实现一致）：
   - 宽高比超过 200 时抛出 `ValueError`；
   - 先 `h = round(height / factor) * factor`，`w = round(width / factor) * factor`（Python 的 `round`）；
   - 如果 `h * w > max_pixels`：`beta = sqrt(height * width / max_pixels)`，`h = floor(height / beta / factor) * factor`，`w = floor(width / beta / factor) * factor`；
   - 否则如果 `h * w < min_pixels`：`beta = sqrt(min_pixels / (height * width))`，`h = ceil(height * beta / factor) * factor`，`w = ceil(width * beta / factor) * factor`；
   - 返回 `(h, w)`；
2. `num_image_tokens(height, width, **kw)`：缩放后切 patch，再 2×2 合并成一个 token：`(h // factor) * (w // factor)`，`factor` 和其他参数一样从 `kw` 里取（默认 28）；
3. `prefill_tokens(text_tokens, images, **kw)`：一个请求的 prefill token 总数，`images` 是 `[(height, width), ...]`，每张图片另外有 2 个特殊 token（`<|vision_start|>`、`<|vision_end|>`）。

```python
smart_resize(1080, 1920)                              # (728, 1316)：超过 max_pixels，按比例缩小
num_image_tokens(1080, 1920)                          # 26 × 47 = 1222
num_image_tokens(1080, 1920, max_pixels=28*28*16384)  # 39 × 69 = 2691
qwen35 = dict(factor=32, min_pixels=65536, max_pixels=16777216)   # Qwen3.5 的处理器配置
num_image_tokens(720, 1280, **qwen35)                # 22 × 40 = 880：720 取整成 704
num_image_tokens(224, 224, **qwen35)                 # 8 × 8 = 64：低于下限，先放大到 256×256
```

<!-- 题解 -->
照规则翻译即可。像素上限放宽时，一张 1080p 图片约 2700 个 token，相当于好几页文字——多模态请求的 prefill 很重，视觉编码器（ViT）本身的计算也不小，
所以推理引擎会把图片编码的结果缓存起来（按图片哈希），并且在调度时单独计算视觉部分的开销。
