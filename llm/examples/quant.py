"""quant.py —— 对称的伪量化（量化后立刻反量化），用于评估误差。"""

import torch


def fake_quant_int(w: torch.Tensor, bits: int, granularity: str = "channel", group: int = 128) -> torch.Tensor:
    out_f, in_f = w.shape
    if granularity == "tensor":
        g = w.reshape(1, 1, -1)
    elif granularity == "channel":
        g = w.reshape(out_f, 1, in_f)
    else:                                                     # "group"
        g = w.reshape(out_f, in_f // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (g / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(w.shape)


def fake_quant_fp8(w: torch.Tensor) -> torch.Tensor:
    scale = w.abs().max() / 448.0                             # E4M3 的最大值是 448
    return (w / scale).to(torch.float8_e4m3fn).float() * scale


def rel_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    return ((approx - exact).norm() / exact.norm()).item()
