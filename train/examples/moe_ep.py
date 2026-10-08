import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single   # 带 autograd 的版本：反向自动做反方向的 all-to-all

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
E, K, H, F, T = 8, 2, 16, 32, 12              # 专家数、top-k、隐藏维度、专家中间维度、每个 rank 的 token 数
LOCAL = E // P                                 # 每个 rank 放 E/P 个专家


def make_params():
    torch.manual_seed(0)
    return torch.randn(E, H) * 0.5, torch.randn(E, F, H) / H ** 0.5, torch.randn(E, H, F) / F ** 0.5


def expert(x, w1, w2):
    return torch.nn.functional.silu(x @ w1.T) @ w2.T


router, W1, W2 = make_params()
torch.manual_seed(1)
X_all = torch.randn(P * T, H)                  # 所有 rank 的 token，用来算单进程参照
X = X_all[rank * T:(rank + 1) * T]

# ---- 单进程参照：所有专家都在本地
ref_w1, ref_w2 = W1.clone().requires_grad_(), W2.clone().requires_grad_()
probs = torch.softmax(X_all @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)
ref = torch.zeros_like(X_all)
for e in range(E):
    tok, slot = (idx == e).nonzero(as_tuple=True)
    ref.index_add_(0, tok, gate[tok, slot, None] * expert(X_all[tok], ref_w1[e], ref_w2[e]))
ref.square().sum().backward()

# ---- 专家并行：rank r 持有专家 [r*LOCAL, (r+1)*LOCAL)
w1 = W1[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
w2 = W2[rank * LOCAL:(rank + 1) * LOCAL].clone().requires_grad_()
probs = torch.softmax(X @ router.T, dim=-1)
gate, idx = probs.topk(K, dim=-1)                          # [T, K]
flat_e = idx.flatten()                                     # 每个 (token, k) 要去的专家
order = flat_e.argsort(stable=True)                        # 按专家（也就是按目标 rank）排序
send_x = X.repeat_interleave(K, dim=0)[order]              # 每个 token 复制 K 份，按目标排好
send_counts = torch.bincount(flat_e // LOCAL, minlength=P) # 发给每个 rank 多少个
recv_counts = torch.empty_like(send_counts)
dist.all_to_all_single(recv_counts, send_counts)           # 先交换数量，才知道要收多少
recv_x = all_to_all_single(torch.empty(int(recv_counts.sum()), H), send_x,
                           recv_counts.tolist(), send_counts.tolist())          # dispatch
send_e = flat_e[order] % LOCAL                             # 本地专家编号也一起发过去
recv_e = torch.empty(int(recv_counts.sum()), dtype=torch.long)
dist.all_to_all_single(recv_e, send_e, recv_counts.tolist(), send_counts.tolist())

recv_y = torch.zeros_like(recv_x)
for e in range(LOCAL):                                     # 本地专家各算各的 token
    m = recv_e == e
    recv_y = recv_y.index_put((m.nonzero(as_tuple=True)[0],), expert(recv_x[m], w1[e], w2[e]))
back = all_to_all_single(torch.empty_like(send_x), recv_y, send_counts.tolist(), recv_counts.tolist())   # combine
y_sorted = torch.empty_like(back).index_copy(0, order, back)                    # 还原成 (token, k) 的原始顺序
out = (y_sorted.view(T, K, H) * gate[..., None]).sum(dim=1)
out.square().sum().backward()

ok_out = torch.allclose(out, ref.detach()[rank * T:(rank + 1) * T], atol=1e-5)
ok_grad = torch.allclose(w1.grad, ref_w1.grad[rank * LOCAL:(rank + 1) * LOCAL], atol=1e-4) and \
    torch.allclose(w2.grad, ref_w2.grad[rank * LOCAL:(rank + 1) * LOCAL], atol=1e-4)
flags = torch.tensor([int(ok_out), int(ok_grad)])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
loads = [None] * P
dist.all_gather_object(loads, int(recv_counts.sum()))
if rank == 0:
    print(f"{P} 个 rank、{E} 个专家、top-{K}：每个 rank 收到的 token 数 {loads}")
    print("输出与单进程一致：", bool(flags[0]), "；本地专家的权重梯度一致：", bool(flags[1]))
dist.destroy_process_group()
