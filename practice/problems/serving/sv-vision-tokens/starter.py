import math


def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=28 * 28 * 1280):
    return height // factor * factor, width // factor * factor      # 没有处理像素上下限


def num_image_tokens(height, width, **kw):
    pass


def prefill_tokens(text_tokens, images, **kw):
    pass
