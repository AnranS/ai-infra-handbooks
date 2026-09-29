import math


def bytes_per_param(bits, group, scale_bytes):
    return bits / 8 + (scale_bytes / group if group else 0.0)


def min_gpus(total, bpp, gpu_mem, reserve=0.3):
    return math.ceil(total * bpp / (gpu_mem * (1 - reserve)))


def decode_floor(active, bpp, gpus, bw):
    return active * bpp / (gpus * bw)


def speedup_needed(bpp_old, bpp_new):
    return bpp_old / bpp_new
