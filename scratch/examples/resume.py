import torch

from model import GPT, GPTConfig

data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)   # 用一个更小的模型，几秒钟就能跑完


def make(seed):
    torch.manual_seed(seed)
    model = GPT(cfg)
    return model, torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)


def train(model, opt, gen, steps):
    for _ in range(steps):
        i = torch.randint(0, len(data) - cfg.seq_len - 1, (8,), generator=gen)
        x = torch.stack([data[j:j + cfg.seq_len] for j in i])
        y = torch.stack([data[j + 1:j + cfg.seq_len + 1] for j in i])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


model, opt = make(seed=0)                                      # 对照：一口气训练 40 步
train(model, opt, torch.Generator().manual_seed(1), 40)
reference = {k: v.clone() for k, v in model.state_dict().items()}

model, opt = make(seed=0)                                      # 训练 20 步就"中断"，存下 checkpoint
gen = torch.Generator().manual_seed(1)
train(model, opt, gen, 20)
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "data_rng": gen.get_state()}, "mid.pt")


def resume(restore_optimizer, restore_data_rng):
    model, opt = make(seed=123)                                # 另一个随机初始化：全靠 checkpoint 恢复
    ckpt = torch.load("mid.pt")
    model.load_state_dict(ckpt["model"])
    if restore_optimizer:
        opt.load_state_dict(ckpt["optimizer"])                 # Adam 的一阶、二阶矩和步数
    gen = torch.Generator().manual_seed(999)
    if restore_data_rng:
        gen.set_state(ckpt["data_rng"])                        # 数据读到了哪里
    train(model, opt, gen, 20)
    return max((a - reference[k]).abs().max().item() for k, a in model.state_dict().items())


for name, o, d in [("恢复模型、优化器状态和数据位置", True, True), ("没有恢复数据位置", True, False),
                   ("没有恢复优化器状态", False, True)]:
    print(f"{name}：续训 20 步后，与一口气训练 40 步的参数最大差 {resume(o, d):.1e}")
