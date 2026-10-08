import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                   # 每个 rank 造出同一份完整权重，模拟"从同一个 checkpoint 切"

B, d, h = 8, 64, 256
x = torch.randn(B, d)
W1 = torch.randn(d, h) / d ** 0.5                      # 第一层：按列切（列并行）
W2 = torch.randn(h, d) / h ** 0.5                      # 第二层：按行切（行并行）

ref = torch.relu(x @ W1) @ W2                          # 单进程参考实现：黄金对照

W1_local = W1.chunk(P, dim=1)[rank]                    # 每个 rank 只拿自己那片
W2_local = W2.chunk(P, dim=0)[rank]
y = torch.relu(x @ W1_local) @ W2_local                # 列并行之后不需要通信，行并行之后要 all-reduce
dist.all_reduce(y)

err = (y - ref).abs().max().item()
if rank == 0:
    print(f"张量并行度 {P}，每个 rank 只持有 {W1_local.shape[1]} / {h} 个隐藏维")
    print(f"和单进程结果逐元素一致：{torch.allclose(y, ref, atol=1e-5)}（最大绝对误差 < 1e-5：{err < 1e-5}）")
    print("整个 MLP 只在最后 all-reduce 一次：这就是张量并行的通信量")
dist.destroy_process_group()
