from __future__ import annotations

from typing import List

import torch
import torch.distributed as dist


class TorchDistributedImpl:
    """用 torch.distributed 做集合通信。GPU 上走 NCCL，CPU 上走 gloo。"""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if dist.get_world_size() == 1:
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        """沿第 0 维拼接各 rank 的张量：[n, ...] -> [world_size * n, ...]。"""
        world_size = dist.get_world_size()
        if world_size == 1:
            return x
        x = x.contiguous()
        out = x.new_empty((world_size * x.shape[0],) + tuple(x.shape[1:]))
        if dist.get_backend() == "gloo":  # gloo 不支持 all_gather_into_tensor
            dist.all_gather(list(out.chunk(world_size, dim=0)), x)
        else:
            dist.all_gather_into_tensor(out, x)
        return out


class DistributedCommunicator:
    """各层通过它通信。plugins 列表的最后一个生效，便于换成 PyNCCL 等更快的实现。"""

    plugins: List[TorchDistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)
