import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with torch.device("meta"):   # 不分配任何内存地构建一个 70B 模型量级的 MLP
    mlp = torch.nn.Sequential(torch.nn.Linear(8192, 28672, bias=False), torch.nn.Linear(28672, 8192, bias=False))
n = sum(p.numel() for p in mlp.parameters())
print(f"参数量 {n}，bf16 下 {n * 2 / 2**30:.2f} GiB，权重在 {mlp[0].weight.device} 上")

with FakeTensorMode():
    x = torch.empty(4, 128, 8192)        # 一个 batch 的隐藏状态，但不占内存
    h = torch.nn.functional.silu(x @ torch.empty(8192, 28672))
    print(type(h).__name__, tuple(h.shape), f"这个中间激活在 bf16 下需要 {h.numel() * 2 / 2**20:.0f} MiB")
