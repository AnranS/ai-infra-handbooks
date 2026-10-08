import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
mesh = init_device_mesh("cpu", (world,))          # GPU 上是 init_device_mesh("cuda", ...)


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))


torch.manual_seed(123)
X, Y = torch.randn(32, 64), torch.randint(0, 10, (32,))
local = slice(rank * 32 // world, (rank + 1) * 32 // world)
loss_fn = torch.nn.CrossEntropyLoss()

ref, model = make_model(), make_model()
for layer in model:                               # 每个线性层是一个 FSDP 单元：用到它时才 all-gather 它的参数
    if isinstance(layer, torch.nn.Linear):
        fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

w = model[0].weight
shapes = [None] * world
dist.all_gather_object(shapes, tuple(w.to_local().shape))
opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-2)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)    # 优化器直接作用在切分后的参数上：状态也是切分的
for step in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()
    opt.zero_grad()
    loss_fn(model(X[local]), Y[local]).backward()
    opt.step()

full = [p.full_tensor() for p in model.parameters()]   # 调试时把切片拼回完整参数
diff = max((a - b).abs().max().item() for a, b in zip(ref.parameters(), full))
if rank == 0:
    print(f"第一层权重的类型：{type(w).__name__}，完整形状 {tuple(w.shape)}，各 rank 本地的形状 {shapes}")
    print("训练 3 步后与单进程一致：", diff < 1e-5)
dist.destroy_process_group()
