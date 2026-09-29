"""激活重计算。不允许使用 torch.utils.checkpoint。检查脚本会验证：

- 梯度与不做重计算时一致；
- 前向时为反向保存的张量字节数显著减少（只保存函数的输入）。
"""

import torch  # noqa: F401


def checkpoint(fn, *args):
    """返回 fn(*args)。前向时不保存 fn 内部的中间结果，反向时用保存的输入重新执行一遍 fn 再求梯度。
    fn 内部的参数（比如 nn.Module 的权重）也要得到正确的梯度。"""
    raise NotImplementedError
