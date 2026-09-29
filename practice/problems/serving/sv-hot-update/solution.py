import numpy as np


def quantize(w):
    w = np.asarray(w, dtype=np.float32)
    scale = (np.abs(w).max(axis=1, keepdims=True) / 127).astype(np.float32)
    scale[scale == 0] = 1.0
    return np.round(w / scale).astype(np.int8), scale


def hot_update(layer, new_w):
    new_w = np.asarray(new_w, dtype=np.float32)
    if new_w.shape != layer.qweight.shape:
        raise ValueError(f"shape mismatch: {new_w.shape} vs {layer.qweight.shape}")
    q, s = quantize(new_w)
    layer.qweight[...] = q
    layer.scale[...] = s


def pause_costs(mode, done, left, prompt, step_ms):
    n = len(done)
    if mode == "abort":
        return {"wait_s": 0.0, "redecode": sum(done), "reprefill": n * prompt, "mixed": 0, "kv_fresh": True}
    if mode == "wait":
        return {"wait_s": max(left, default=0) * step_ms / 1000, "redecode": 0, "reprefill": 0, "mixed": 0,
                "kv_fresh": True}
    if mode == "keep":
        return {"wait_s": 0.0, "redecode": 0, "reprefill": 0, "mixed": n, "kv_fresh": False}
    if mode == "retract":
        return {"wait_s": 0.0, "redecode": 0, "reprefill": n * prompt + sum(done), "mixed": n, "kv_fresh": True}
    raise ValueError(f"unknown mode {mode!r}")
