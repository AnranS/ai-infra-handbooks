"""指令微调：从预训练的 checkpoint 出发，只在"助手说的话"上算 loss"""
import math

import torch

from chat import IGNORE, accuracy, answer, load, load_tokenizer, read
from model import GPT, GPTConfig

torch.manual_seed(0)
tok = load_tokenizer()
ckpt = torch.load("ckpt.pt", weights_only=False)
cfg = GPTConfig(**{**ckpt["model_config"], "vocab_size": tok.get_vocab_size()})
model = GPT(cfg)
state = ckpt["model"]
with torch.no_grad():                                         # 词表多了两个 token：词嵌入多两行，其余权重照搬
    state["embed.weight"] = torch.cat([state["embed.weight"], model.embed.weight[8192:].clone()])
    state["head.weight"] = state["embed.weight"]              # 输入和输出共享同一个矩阵
model.load_state_dict(state)
print(f"预训练模型 {ckpt['step']} 步、{model.num_params() / 1e6:.2f}M 参数；词表 8192 → {cfg.vocab_size}，词嵌入补了两行")

x_tr, y_tr = load(tok, "sft_train.jsonl", cfg.seq_len)
x_va, y_va = load(tok, "sft_val.jsonl", cfg.seq_len)
who = [(c[-2]["content"], c[-1]["content"]) for c in read("sft_who.jsonl")][-200:]
print(f"训练 {len(x_tr)} 条、验证 {len(x_va)} 条；每条 {cfg.seq_len} 个位置，"
      f"只有 {(y_tr != IGNORE).float().mean():.1%} 要算 loss")


@torch.no_grad()
def val_loss():
    model.eval()
    loss = sum(model(x_va[i:i + 64], y_va[i:i + 64])[1].item() for i in range(0, len(x_va), 64))
    model.train()
    return loss / math.ceil(len(x_va) / 64)


ASK = ["这句话是谁说的：「吾与汝同生共死，汝可速去。」", "接下来写：孔明曰：「吾自有计。」",
       "「操大喜，遂引兵望寿春而来。」的下一句是什么？"]
cont = [c[-2]["content"] for c in read("sft_val.jsonl") if "接下来写" in c[-2]["content"]][:100]


def task_style():
    """答案的"长相"对不对：「谁说的」该是两三个字的人名，续写该是一整句话"""
    short = sum(len(answer(model, tok, q, 8)) <= 4 for q, _ in who[:100])
    long_ = sum(len(answer(model, tok, q, 40)) >= 8 for q in cont)
    return short / 100, long_ / len(cont)


print(f"\n微调前：验证 loss {val_loss():.3f}，「谁说的」准确率 {accuracy(model, tok, who):.1%}"
      f"（12 个人里挑，随机是 8.3%）")
print("把问题直接喂给它（它没见过聊天模板），只会顺着往下编：")
for q in ASK[:2]:
    print(f"  问：{q}\n  答：{answer(model, tok, q, 20, templated=False)}")

steps, batch, peak, warmup = 400, 32, 1e-3, 40
opt = torch.optim.AdamW(model.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.1)
gen = torch.Generator().manual_seed(1)
print(f"\n微调 {steps} 步 × {batch} 条 = {steps * batch / len(x_tr):.1f} 遍，峰值学习率 {peak:g}（预训练用的是 3e-3）")
print("  步   训练 loss   验证 loss")
running = 0.0
for step in range(steps):
    lr = peak * (step + 1) / warmup if step < warmup else \
        0.1 * peak + 0.45 * peak * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr
    i = torch.randint(0, len(x_tr), (batch,), generator=gen)
    _, loss = model(x_tr[i], y_tr[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
    if (step + 1) % 100 == 0:
        print(f"{step + 1:4d}   {running / 100:8.3f}   {val_loss():8.3f}")
        running = 0.0

short, long_ = task_style()
print(f"\n微调后：「谁说的」准确率 {accuracy(model, tok, who):.1%}；"
      f"答人名的题里 {short:.0%} 答了 4 个字以内，续写的题里 {long_:.0%} 答出了完整的一句")
for q in ASK:
    print(f"  问：{q}\n  答：{answer(model, tok, q)}")
torch.save({"model": model.state_dict(), "model_config": vars(cfg)}, "sft.pt")
tok.save("tokenizer_chat.json")
