import torch
import torch.distributed as dist

from zero_adam import ZeroAdam

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()


def make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 10))


torch.manual_seed(123)
X, Y = torch.randn(32, 64), torch.randint(0, 10, (32,))
local = slice(rank * 32 // world, (rank + 1) * 32 // world)
loss_fn = torch.nn.CrossEntropyLoss()

ref, model = make_model(), make_model()
opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-2)
opt = ZeroAdam(model.parameters(), lr=1e-2)
for step in range(3):
    opt_ref.zero_grad()
    loss_fn(ref(X), Y).backward()
    opt_ref.step()
    opt.zero_grad()
    loss_fn(model(X[local]), Y[local]).backward()
    opt.step()

diff = max((a - b).abs().max().item() for a, b in zip(ref.parameters(), model.parameters()))
full_state = 3 * sum(p.numel() for p in model.parameters()) * 4   # 不切分时：主参数 + 两个矩，fp32
if rank == 0:
    print("训练 3 步后与单进程的 Adam 一致：", diff < 1e-5)
    print(f"每个 rank 的优化器状态 {opt.state_bytes()} 字节，不切分时 {full_state} 字节，约为 1/{round(full_state / opt.state_bytes())}")
dist.destroy_process_group()
