"""LoRA：冻住整份权重，只训练挂在注意力投影上的两个小矩阵，教模型一个新的回答格式"""
import re

import torch
import torch.nn as nn

from chat import answer, encode, load_tokenizer, read
from model import GPT, GPTConfig

torch.manual_seed(0)
tok = load_tokenizer()
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
model = GPT(cfg)
model.load_state_dict(sft["model"])


class LoRALinear(nn.Module):
    """y = Wx + (α/r)·BAx。A 高斯初始化、B 全零：挂上去的那一刻，输出和原模型一模一样"""

    def __init__(self, base: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale

    def merge(self):
        """把 BA 加回 W，换回一个普通的 Linear：推理时没有任何额外开销"""
        merged = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        merged.weight.data = self.base.weight.data + self.scale * (self.B.weight.data @ self.A.weight.data)
        return merged


def apply_lora(model, rank=8):
    """挂在注意力的 qkv 和输出投影上（最常见的选择），返回要训练的参数"""
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, rank)
        block.attn.proj = LoRALinear(block.attn.proj, rank)
    return [p for n, p in model.named_parameters() if n.endswith((".A.weight", ".B.weight"))]


# 新任务：同样是"谁说的"，但要求用一句完整的话回答——SFT 过的模型只会蹦人名
ASK = "这句话是谁说的？请用一句话回答：「{}」"
pairs = [(re.search("「(.+)」", c[-2]["content"]).group(1), c[-1]["content"]) for c in read("sft_who.jsonl")]
train, test = pairs[:400], pairs[-200:]
x_tr, y_tr = zip(*(encode(tok, [{"role": "user", "content": ASK.format(q)},
                                {"role": "assistant", "content": f"这是{a}说的。"}], cfg.seq_len) for q, a in train))
x_tr, y_tr = torch.stack(x_tr), torch.stack(y_tr)
FORMAT = re.compile(r"^这是.{1,3}说的。$")


def score():
    """格式对的比例，以及格式和人名都对的比例"""
    outs = [answer(model, tok, ASK.format(q), 10) for q, _ in test]
    ok = sum(bool(FORMAT.match(o)) for o in outs) / len(outs)
    right = sum(o == f"这是{a}说的。" for o, (_, a) in zip(outs, test)) / len(outs)
    return ok, right, outs[0]


ok, right, sample = score()
print(f"微调好的模型遇到这个新要求：格式对 {ok:.0%}、人名也对 {right:.0%}；它答的是「{sample}」")

for p in model.parameters():
    p.requires_grad_(False)
lora_params = apply_lora(model, rank=8)
n_lora = sum(p.numel() for p in lora_params)
total = sum(p.numel() for p in model.parameters())
print(f"挂上 rank=8 的 LoRA：{n_lora:,} 个可训练参数 / 共 {total:,} 个 = {n_lora / total:.2%}")
ok, _, _ = score()
print(f"B 初始化成全零，挂上的那一刻行为不变：格式对还是 {ok:.0%}")

steps, batch, lr = 200, 16, 3e-3                               # 只训练 LoRA，学习率可以比全量微调大一个量级
opt = torch.optim.AdamW(lora_params, lr=lr, betas=(0.9, 0.95))
gen = torch.Generator().manual_seed(2)
print(f"\n只训练这 {n_lora:,} 个参数（400 条新格式的数据，{steps} 步 × {batch} 条）")
print("  步   训练 loss   格式对   人名也对   它答的是")
running = 0.0
for step in range(steps):
    i = torch.randint(0, len(x_tr), (batch,), generator=gen)
    _, loss = model(x_tr[i], y_tr[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    running += loss.item()
    if (step + 1) % 50 == 0:
        ok, right, sample = score()
        print(f"{step + 1:4d}   {running / 50:8.3f}   {ok:6.0%}   {right:8.0%}   「{sample}」")
        running = 0.0

probe = x_tr[:4]
with torch.no_grad():
    before = model(probe)
for block in model.blocks:                                     # 合并：BA 加回权重，模型结构回到原样
    block.attn.qkv, block.attn.proj = block.attn.qkv.merge(), block.attn.proj.merge()
with torch.no_grad():
    after = model(probe)
print(f"合并回权重后最大误差 {(before - after).abs().max():.2e}，参数量回到 {sum(p.numel() for p in model.parameters()):,}")
