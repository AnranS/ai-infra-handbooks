import torch

from model import GPT, GPTConfig

data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)
i = torch.randint(0, len(data) - 65, (32,), generator=torch.Generator().manual_seed(0))
x = torch.stack([data[j:j + 64] for j in i])
y = torch.stack([data[j + 1:j + 65] for j in i])


def grads(micro_batches):
    torch.manual_seed(0)
    model = GPT(cfg)
    for xs, ys in zip(x.chunk(micro_batches), y.chunk(micro_batches)):
        _, loss = model(xs, ys)
        (loss / micro_batches).backward()                     # 每个小 batch 的 loss 除以份数：梯度累加起来就是大 batch 的平均
    return torch.cat([p.grad.flatten() for p in model.parameters()])


full = grads(1)
for k in (2, 4, 8):
    diff = (grads(k) - full).abs().max().item()
    print(f"32 条分成 {k} 份累加：与一次算 32 条的梯度最大差小于 1e-6：{diff < 1e-6}（梯度最大值 {full.abs().max():.2f}）")
