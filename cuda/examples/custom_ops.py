import torch


@torch.library.custom_op("demo::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # 真实系统里这里调用自己的 CUDA / Triton kernel；这里用 PyTorch 代替
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@rms_norm.register_fake
def _(x, w, eps):
    return torch.empty_like(x)   # 只描述输出的形状和类型，不做计算


@torch.library.custom_op("demo::store_kv", mutates_args=("cache",))
def store_kv(cache: torch.Tensor, slots: torch.Tensor, value: torch.Tensor) -> None:
    cache.index_copy_(0, slots, value)   # 把新 token 的 KV 写进缓存的指定槽位


print(torch.ops.demo.rms_norm.default._schema)
print(torch.ops.demo.store_kv.default._schema)

torch.manual_seed(0)
x, w = torch.randn(2, 16), torch.ones(16)
print("CPU 上计算：", tuple(torch.ops.demo.rms_norm(x, w, 1e-6).shape))

m = torch.ops.demo.rms_norm(torch.empty(4096, 8192, device="meta"), torch.empty(8192, device="meta"), 1e-6)
print("meta 设备上只推导形状：", tuple(m.shape), m.device)

cache = torch.zeros(8, 4)
torch.ops.demo.store_kv(cache, torch.tensor([1, 5]), torch.ones(2, 4))
print("写入的槽位：", cache.sum(dim=1).nonzero().flatten().tolist())


def block(x, w):
    return torch.ops.demo.rms_norm(x, w, 1e-6).relu()


compiled = torch.compile(block, fullgraph=True, backend="eager")   # fullgraph：整段代码必须被捕获成一张图
print("torch.compile 能完整捕获：", torch.allclose(compiled(x, w), block(x, w)))
