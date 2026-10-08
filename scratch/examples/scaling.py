import math

import torch

from model import GPT, GPTConfig

torch.set_num_threads(8)                                       # 这个实验要训 4 个模型，多用几个线程
data = torch.load("tokens.pt")
train_data, val_data = data["train"].long(), data["val"].long()
T, B, STEPS = 128, 16, 400
val_gen = torch.Generator().manual_seed(123)
val_batches = []
for _ in range(8):
    i = torch.randint(0, len(val_data) - T - 1, (32,), generator=val_gen)
    val_batches.append((torch.stack([val_data[j:j + T] for j in i]), torch.stack([val_data[j + 1:j + T + 1] for j in i])))


def run(d_model, n_layer):
    torch.manual_seed(0)
    model = GPT(GPTConfig(d_model=d_model, n_layer=n_layer, n_head=d_model // 32, seq_len=T))
    decay = [p for p in model.parameters() if p.dim() >= 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0}],
                            lr=3e-3, betas=(0.9, 0.95))
    gen, curve = torch.Generator().manual_seed(0), []
    for step in range(STEPS):
        lr = 3e-3 * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        i = torch.randint(0, len(train_data) - T - 1, (B,), generator=gen)
        x = torch.stack([train_data[j:j + T] for j in i])
        y = torch.stack([train_data[j + 1:j + T + 1] for j in i])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                curve.append(sum(model(vx, vy)[1].item() for vx, vy in val_batches) / len(val_batches))
            model.train()
    return model.num_params(), curve, loss.item()


print("  d  层数  参数量    验证 loss：第 100    200    300    400 步   最后的训练 loss")
for d, L in [(64, 2), (96, 3), (128, 4), (192, 4)]:
    n, curve, train_loss = run(d, L)
    print(f"{d:3d}  {L:3d}  {n / 1e6:5.2f}M        " + "  ".join(f"{v:5.2f}" for v in curve) + f"      {train_loss:5.2f}")
