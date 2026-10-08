# GPT-2 规模的小模型，一张 16 GB 卡上的训练配置。模型结构见"从零训练（二）"。
import math
import time

import torch

from model import GPT, GPTConfig          # 从零训练（二）里的模型

dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")          # 让 fp32 的矩阵乘也走 TF32

cfg = GPTConfig(n_layer=12, n_head=12, d_model=768, seq_len=1024, vocab_size=32000)
model = GPT(cfg).to(dev)
model = torch.compile(model)                        # 第一次编译几分钟，之后复用缓存

MICRO_BS, ACCUM = 8, 16                             # 有效 batch = 8 × 16 × 1024 ≈ 131k token
STEPS, WARMUP, LR = 20000, 400, 6e-4

# AdamW：只给二维以上的参数加 weight decay（LayerNorm 和 bias 不加）
decay = [p for p in model.parameters() if p.dim() >= 2]
nodecay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95), eps=1e-8, fused=True)


def lr_at(step):                                     # 线性 warmup + 余弦退火到 10%
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    t = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.1 * LR + 0.45 * LR * (1 + math.cos(math.pi * t))


def get_batch():                                     # 换成你自己的数据加载
    raise NotImplementedError


for step in range(STEPS):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for micro in range(ACCUM):                       # 梯度累积：小显存也能有大 batch
        x, y = get_batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)                        # GPT.forward 传了 targets 就顺便算 loss（内部用 fp32 算 softmax）
        (loss / ACCUM).backward()                        # loss 要除以累积步数
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok = MICRO_BS * ACCUM * cfg.seq_len
        print(f"step {step:>6} loss {loss.item():.4f} grad_norm {gnorm:.2f} "
              f"{tok / dt:>8.0f} tok/s 显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
