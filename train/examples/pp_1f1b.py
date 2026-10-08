import torch
import torch.distributed as dist

dist.init_process_group("gloo")
stage, p = dist.get_rank(), dist.get_world_size()
M, MB, H = 4, 2, 16                                     # micro-batch 个数、每个 micro-batch 的样本数、隐藏维度


def make_layers():
    torch.manual_seed(0)
    return [torch.nn.Sequential(torch.nn.Linear(H, H), torch.nn.Tanh()) for _ in range(2 * p)]


torch.manual_seed(1)
X, Y = torch.randn(M * MB, H), torch.randn(M * MB, H)
layers = make_layers()

# 单进程参照：整个模型、整个 batch（损失是全 batch 的平均，等于各 micro-batch 平均损失的平均）
ref = torch.nn.Sequential(*make_layers())
torch.nn.functional.mse_loss(ref(X), Y).backward()
ref_grads = [q.grad for q in ref.parameters()]

# 流水线：stage s 负责第 2s、2s+1 层
mine = torch.nn.Sequential(*layers[2 * stage:2 * stage + 2])
first, last = stage == 0, stage == p - 1
inputs, outputs, trace, pending = {}, {}, [], []


def forward(i):
    if first:
        x = X[i * MB:(i + 1) * MB]
    else:
        x = torch.empty(MB, H)
        dist.recv(x, stage - 1)                          # 收上一个 stage 的激活
        x.requires_grad_()
    y = mine(x)
    inputs[i], outputs[i] = x, y
    if not last:
        pending.append(dist.isend(y.detach(), stage + 1))   # 异步发给下一个 stage，不阻塞
    trace.append(f"F{i}")


def backward(i):
    y = outputs.pop(i)
    if last:
        (torch.nn.functional.mse_loss(y, Y[i * MB:(i + 1) * MB]) / M).backward()
    else:
        g = torch.empty(MB, H)
        dist.recv(g, stage + 1)                          # 收下一个 stage 传回的梯度
        y.backward(g)
    x = inputs.pop(i)
    if not first:
        pending.append(dist.isend(x.grad, stage - 1))    # 把对输入的梯度传回上一个 stage
    trace.append(f"B{i}")


warm = min(p - stage - 1, M)                             # 1F1B：先做几个前向"预热"，然后一前一后交替
for i in range(warm):
    forward(i)
for i in range(M - warm):
    forward(warm + i)
    backward(i)
for i in range(M - warm, M):
    backward(i)
for w in pending:
    w.wait()

ok = all(torch.allclose(q.grad, r, atol=1e-6) for q, r in zip(mine.parameters(), ref_grads[4 * stage:4 * stage + 4]))
traces = [None] * p
dist.all_gather_object(traces, " ".join(trace))
flags = torch.tensor([int(ok)])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
if stage == 0:
    for s, t in enumerate(traces):
        print(f"stage {s} 的执行顺序：{t}")
    print("各 stage 的参数梯度与单进程一致：", bool(flags.item()))
dist.destroy_process_group()
