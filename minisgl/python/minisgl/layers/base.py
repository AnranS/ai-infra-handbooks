"""算子基类。

为什么不用 torch.nn.Module？推理引擎不需要自动求导、hook、train/eval 模式，也不需要 Module 的
参数注册机制；它需要的只是"按名字收集和加载权重"。BaseOP 用对象的 __dict__ 实现这件事：
所有不以下划线开头的 Tensor 属性就是权重，BaseOP 属性就是子模块，名字按属性路径拼接，
恰好与 Hugging Face checkpoint 里的键名一致（model.layers.0.self_attn.qkv_proj.weight）。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeVar

import torch

StateDict = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: StateDict | None = None) -> StateDict:
        result = result if result is not None else {}
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)
        return result

    def load_state_dict(
        self, state_dict: StateDict, *, prefix: str = "", _internal: bool = False
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert param.shape == item.shape, (name, param.shape, item.shape)
                assert param.dtype == item.dtype, (name, param.dtype, item.dtype)
                setattr(self, name, item)  # 直接替换引用：模型是在 meta 设备上建的，没有真实内存
            elif isinstance(param, BaseOP):
                param.load_state_dict(state_dict, prefix=_concat_prefix(prefix, name), _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """没有权重的算子（例如 RoPE、注意力层本身）。"""

    def state_dict(self, *, prefix: str = "", result: StateDict | None = None) -> StateDict:
        return result if result is not None else {}

    def load_state_dict(
        self, state_dict: StateDict, *, prefix: str = "", _internal: bool = False
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """算子列表，名字是 0、1、2……，对应 checkpoint 里的 layers.0、layers.1。"""

    def __init__(self, ops: List[T]):
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: StateDict | None = None) -> StateDict:
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self, state_dict: StateDict, *, prefix: str = "", _internal: bool = False
    ) -> None:
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
