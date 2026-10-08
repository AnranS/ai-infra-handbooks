"""白盒蒸馏：让小模型去拟合大模型的整个输出分布，而不只是"正确的那一个 token" """
import math

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

torch.set_num_threads(8)
data = torch.load("tokens.pt")
train_data, val_data = data["train"].long(), data["val"].long()
T, B, STEPS = 128, 16, 400
val_gen = torch.Generator().manual_seed(123)
val_batches = []
for _ in range(8):
    i = torch.randint(0, len(val_data) - T - 1, (32,), generator=val_gen)
    val_batches.append((torch.stack([val_data[j:j + T] for j in i]),
                        torch.stack([val_data[j + 1:j + T + 1] for j in i])))

ckpt = torch.load("ckpt.pt", weights_only=False)
teacher = GPT(GPTConfig(**ckpt["model_config"]))
teacher.load_state_dict(ckpt["model"])
teacher.eval().requires_grad_(False)


@torch.no_grad()
def evaluate(model):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
    model.train()
    return loss


def run(alpha, temperature=2.0):
    """alpha=1 是普通训练；alpha<1 时把 (1-alpha) 的权重给"向老师的分布看齐"这一项"""
    torch.manual_seed(0)
    student = GPT(GPTConfig(d_model=64, n_layer=2, n_head=2, seq_len=T))
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(0)
    for step in range(STEPS):
        lr = 3e-3 * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        i = torch.randint(0, len(train_data) - T - 1, (B,), generator=gen)
        x = torch.stack([train_data[j:j + T] for j in i])
        y = torch.stack([train_data[j + 1:j + T + 1] for j in i])
        logits, ce = student(x, y)
        loss = ce
        if alpha < 1.0:
            with torch.no_grad():
                soft = (teacher(x) / temperature).softmax(-1)
            kl = F.kl_div((logits.float() / temperature).log_softmax(-1).flatten(0, 1),
                          soft.flatten(0, 1), reduction="batchmean")      # 每个位置一个分布，按 token 取平均
            loss = alpha * ce + (1 - alpha) * temperature ** 2 * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
    return student


print(f"老师：{teacher.num_params() / 1e6:.2f}M 参数，验证 loss {evaluate(teacher):.3f}")
plain = run(alpha=1.0)
print(f"学生只学正确答案（普通训练）：{plain.num_params() / 1e6:.2f}M 参数，验证 loss {evaluate(plain):.3f}")
for alpha in (0.5, 0.0):
    student = run(alpha=alpha)
    print(f"学生 + 蒸馏（α={alpha}，T=2）：验证 loss {evaluate(student):.3f}")
