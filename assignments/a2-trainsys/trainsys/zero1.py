"""ZeRO-1：把优化器状态切分到各个 rank。检查脚本会验证：

- 与单进程的 torch.optim.AdamW（同样的超参数）训练结果一致；
- 每个 rank 上优化器状态的字节数约为不切分时的 1 / world_size。
"""

import torch
import torch.distributed as dist  # noqa: F401


class ZeRO1:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """params：模型的全部参数（每个 rank 都持有完整的参数）。可以在内部使用 torch.optim.AdamW 更新自己负责的那一片。"""
        raise NotImplementedError

    def zero_grad(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        """调用前梯度已经在各 rank 之间求过平均（例如由 BucketedDDP 完成）。更新自己负责的参数分片，
        再把更新后的参数同步给所有 rank。"""
        raise NotImplementedError

    def state_bytes(self) -> int:
        """本 rank 上优化器状态（一阶矩、二阶矩等张量）占用的字节数。"""
        raise NotImplementedError
