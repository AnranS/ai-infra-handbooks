import torch
import torch.nn as nn
import torch.nn.functional as F
from muon import Muon

V, D, T = 64, 96, 64
g = torch.Generator().manual_seed(0)
# 一个"语言"：二阶马尔可夫链，下一个 token 由前两个决定（每个上下文只有少数几个常见的后继）
P = (torch.randn(V, V, V, generator=g) * 3).softmax(-1)
seqs = torch.randint(0, V, (8192, 2), generator=g)
for _ in range(T - 1):
    nxt = torch.multinomial(P[seqs[:, -2], seqs[:, -1]], 1, generator=g)
    seqs = torch.cat([seqs, nxt], dim=1)                    # 8192 条长 65 的序列
train, val = seqs[:7168], seqs[7168:]
print(f"这个语言的熵：每个 token {-(P * P.log()).sum(-1).mean():.2f} nat（loss 的下限）")


class Block(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.qkv, self.proj = nn.Linear(D, 3 * D, bias=False), nn.Linear(D, D, bias=False)
        self.up, self.down = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, D))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        self.blocks = nn.Sequential(Block(), Block())
        self.norm, self.head = nn.RMSNorm(D), nn.Linear(D, V, bias=False)

    def forward(self, idx):
        return self.head(self.norm(self.blocks(self.emb(idx) + self.pos(torch.arange(idx.shape[1])))))


def train_run(name, lr, steps=400, batch=32):
    torch.manual_seed(0)
    model = TinyGPT()
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks") and p.ndim == 2]
    others = [p for n, p in model.named_parameters() if not (n.startswith("blocks") and p.ndim == 2)]
    if name == "AdamW":
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)]
    else:                                                    # 嵌入、输出层、归一化仍然用 AdamW
        opts = [Muon(hidden, lr=lr), torch.optim.AdamW(others, lr=lr, weight_decay=0.0)]
    gen, curve = torch.Generator().manual_seed(1), []
    for step in range(1, steps + 1):
        x = train[torch.randint(0, len(train), (batch,), generator=gen)]
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
        if step % 100 == 0:
            with torch.no_grad():
                curve.append(F.cross_entropy(model(val[:, :-1]).flatten(0, 1), val[:, 1:].flatten()).item())
    return curve


print("验证集 loss        第 100 步  第 200 步  第 300 步  第 400 步")
for name, lrs in [("AdamW", (3e-3, 1e-2, 2e-2)), ("Muon", (1e-2, 2e-2, 4e-2))]:
    for lr in lrs:
        print(f"{name:5s} lr={lr:<6}  " + "".join(f"{v:9.3f}" for v in train_run(name, lr)))
