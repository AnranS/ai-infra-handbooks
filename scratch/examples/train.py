import math
from dataclasses import asdict, dataclass

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    batch_size: int = 16          # 每步 16 条、每条 128 个 token
    max_steps: int = 600
    lr: float = 3e-3              # 峰值学习率：小模型可以用得很大
    min_lr: float = 3e-4          # 余弦衰减到峰值的 1/10
    warmup: int = 60
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 100
    seed: int = 0


tc, mc = TrainConfig(), GPTConfig()
torch.manual_seed(tc.seed)
data = torch.load("tokens.pt")
train_data, val_data = data["train"].long(), data["val"].long()
gen = torch.Generator().manual_seed(tc.seed)


def get_batch(split_data, batch_size, generator):
    """随机取 batch_size 段长 seq_len + 1 的连续 token：前 seq_len 个是输入，往后错一位是目标"""
    i = torch.randint(0, len(split_data) - mc.seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([split_data[j:j + mc.seq_len] for j in i])
    y = torch.stack([split_data[j + 1:j + mc.seq_len + 1] for j in i])
    return x, y


val_batches = [get_batch(val_data, 32, torch.Generator().manual_seed(123)) for _ in range(8)]   # 固定的验证 batch


@torch.no_grad()
def evaluate(model):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in val_batches) / len(val_batches)
    model.train()
    return loss


def lr_at(step):
    """线性 warmup，之后余弦衰减到 min_lr"""
    if step < tc.warmup:
        return tc.lr * (step + 1) / tc.warmup
    progress = (step - tc.warmup) / max(1, tc.max_steps - tc.warmup)
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * progress))


model = GPT(mc)
decay = [p for p in model.parameters() if p.dim() >= 2]       # 矩阵和词嵌入做权重衰减
no_decay = [p for p in model.parameters() if p.dim() < 2]     # 归一化层的增益不做
opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                         {"params": no_decay, "weight_decay": 0.0}], lr=tc.lr, betas=(0.9, 0.95))
print(f"模型 {model.num_params() / 1e6:.2f}M 参数；每步 {tc.batch_size * mc.seq_len} 个 token，"
      f"共 {tc.max_steps} 步 = {tc.max_steps * tc.batch_size * mc.seq_len / len(train_data):.1f} 遍训练集")
print(f"初始验证 loss {evaluate(model):.3f}（均匀猜测是 ln {mc.vocab_size} = {math.log(mc.vocab_size):.3f}）")
print("  步   学习率   训练 loss   验证 loss   梯度范数")
running = 0.0
for step in range(tc.max_steps):
    for group in opt.param_groups:
        group["lr"] = lr_at(step)
    x, y = get_batch(train_data, tc.batch_size, gen)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)   # 返回裁剪前的全局范数
    opt.step()
    running += loss.item()
    if (step + 1) % tc.eval_every == 0:
        print(f"{step + 1:4d}  {lr_at(step):.2e}   {running / tc.eval_every:8.3f}   {evaluate(model):8.3f}   {norm:8.3f}")
        running = 0.0

torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": tc.max_steps,
            "model_config": asdict(mc), "train_config": asdict(tc), "rng": gen.get_state()}, "ckpt.pt")

tok = Tokenizer.from_file("tokenizer.json")
sample_gen = torch.Generator().manual_seed(42)
model.eval()
for prompt in ("却说曹操", "玄德曰："):
    idx = torch.tensor([tok.encode(prompt).ids])
    out = model.generate(idx, 60, temperature=0.8, top_k=50, generator=sample_gen)
    print(f"【{prompt}】" + tok.decode(out[0, idx.shape[1]:].tolist()).replace("\n", " "))
