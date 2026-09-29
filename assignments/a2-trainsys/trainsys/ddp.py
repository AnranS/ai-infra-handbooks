"""分桶、与反向重叠的数据并行。检查脚本会验证：

- 训练结果与单进程在完整 batch 上训练一致（逐元素 allclose）；
- 至少有一次 all-reduce 是在 loss.backward() 返回之前、以 async_op=True 发起的（通信与反向重叠）。
"""

import torch
import torch.distributed as dist  # noqa: F401


class BucketedDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_bytes: int = 64 * 1024):
        """构造时把 rank 0 的参数广播给所有 rank；按反向产生梯度的顺序（大致是参数的逆序）分桶。"""
        super().__init__()
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def finish_gradient_sync(self) -> None:
        """在 optimizer.step() 之前调用：等待所有桶的通信完成，并把每个参数的 .grad 换成所有 rank 的平均值。"""
        raise NotImplementedError
