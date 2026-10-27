import math

import torch

import model as M

ckpt = torch.load("ckpt.pt")
cfg = M.GPTConfig(**ckpt["model_config"])
net = M.GPT(cfg)
net.load_state_dict(ckpt["model"])
net.eval()

L, s = cfg.seq_len, 4                                        # 训练时每条 128 个 token；读 4 倍长，扩展倍数也取 4
val = torch.load("tokens.pt")["val"].long()
xs = torch.stack([val[i:i + s * L + 1] for i in range(0, len(val) - s * L - 1, s * L)])   # 验证集切成不重叠的 512 段
d = cfg.d_model // cfg.n_head
inv = 1.0 / cfg.rope_theta ** (torch.arange(0, d, 2).float() / d)                         # 原来每一对维度的转速

ntk = 1.0 / (cfg.rope_theta * s ** (d / (d - 2))) ** (torch.arange(0, d, 2).float() / d)
dim_of = lambda turns: d * math.log(L / (turns * 2 * math.pi)) / (2 * math.log(cfg.rope_theta))
low, high = max(math.floor(dim_of(32)), 0), min(math.ceil(dim_of(1)), d - 1)
ramp = ((torch.arange(d // 2).float() - low) / (high - low)).clamp(0, 1)                  # 0：不动，1：按 s 倍插值
yarn = inv / s * ramp + inv * (1 - ramp)


def use(new_inv, scale=1.0):
    """换掉 model.py 里算 cos、sin 的函数：转速换成 new_inv，再乘一个系数"""
    def rope_cos_sin(T, head_dim, theta, device=None):
        freqs = torch.outer(torch.arange(T, device=device).float(), new_inv)
        return freqs.cos() * scale, freqs.sin() * scale
    M.rope_cos_sin = rope_cos_sin


print(f"验证集切成 {len(xs)} 段，每段 {s * L} 个 token；下面是位置 0～127、128～255、256～383、384～511 的平均 loss")
with torch.no_grad():
    for name, new_inv, scale in [("原样", inv, 1.0), ("PI（位置 ÷ 4）", inv / s, 1.0), ("NTK", ntk, 1.0),
                                 ("YaRN", yarn, 0.1 * math.log(s) + 1)]:
        use(new_inv, scale)
        nll = torch.nn.functional.cross_entropy(net(xs[:, :-1]).flatten(0, 1), xs[:, 1:].flatten(), reduction="none")
        per = nll.view(len(xs), s * L).mean(0)
        print(f"{name}：" + "  ".join(f"{per[a:a + L].mean():.2f}" for a in range(0, s * L, L)))
