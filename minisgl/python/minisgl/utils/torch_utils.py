from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """临时修改默认 dtype：在这个上下文里 torch.empty(...) 直接得到目标精度的张量。"""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
