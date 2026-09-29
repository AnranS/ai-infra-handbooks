import math


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("宽高比不能超过 200")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


def num_image_tokens(height, width, **kw):
    h, w = smart_resize(height, width, **kw)
    return (h // 28) * (w // 28)


def prefill_tokens(text_tokens, images, **kw):
    return text_tokens + sum(num_image_tokens(h, w, **kw) + 2 for h, w in images)
