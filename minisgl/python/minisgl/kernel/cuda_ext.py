"""即时编译并加载 csrc/ext.cu（只在 CUDA 上使用；编译失败时返回 None，调用方退回 PyTorch 实现）。"""

from __future__ import annotations

import functools
import pathlib
from typing import Any

_CSRC = pathlib.Path(__file__).parent / "csrc"


@functools.cache
def load_ext() -> Any:
    try:
        from torch.utils.cpp_extension import load

        return load(name="minisgl_kernels", sources=[str(_CSRC / "ext.cu")],
                    extra_include_paths=[str(_CSRC)], extra_cuda_cflags=["-O3"], verbose=False)
    except Exception:  # 没有 nvcc、没有 GPU 等
        return None
