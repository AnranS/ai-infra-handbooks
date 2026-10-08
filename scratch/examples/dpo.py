"""DPO：用"原文 vs 模型自己写的"组成偏好对，不训练奖励模型，直接把策略往偏好的那一边推"""
import copy

import torch
import torch.nn.functional as F

from chat import IGNORE, chat_ids, encode, load_tokenizer, read
from model import GPT, GPTConfig

torch.manual_seed(0)
tok = load_tokenizer()
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
policy = GPT(cfg)
policy.load_state_dict(sft["model"])
ref = copy.deepcopy(policy).eval().requires_grad_(False)       # 参考模型：冻住的那一份，整个训练过程不变


@torch.no_grad()
def sample(prompt, max_new, generator):
    """让当前模型自己写一个回答（采样，不是贪心）——这就是"被拒绝"的那一条"""
    policy.eval()
    idx = torch.tensor([chat_ids(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)[0]])
    out = []
    for _ in range(max_new):
        logits = policy(idx[:, -cfg.seq_len:])[:, -1]
        nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
        if nxt.item() == tok.token_to_id("<|im_end|>"):
            break
        out.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    policy.train()
    return tok.decode(out, skip_special_tokens=False)


gen = torch.Generator().manual_seed(3)
prompts = [(c[-2]["content"], c[-1]["content"]) for c in read("sft_val.jsonl") if "这句话是谁说" not in c[-2]["content"]]
pairs = []
for q, good in prompts:
    bad = sample(q, max_new=len(tok.encode(good, add_special_tokens=False).ids) + 4, generator=gen).strip()
    if bad and bad != good:
        pairs.append((q, good, bad))
print(f"{len(pairs)} 对偏好数据：chosen 是原文的下一句，rejected 是模型自己采样出来的")
print(f"  问：{pairs[0][0]}\n  chosen  ：{pairs[0][1]}\n  rejected：{pairs[0][2]}")


def batch_of(items):
    xs, ys = [], []
    for q, good, bad in items:
        for a in (good, bad):
            x, y = encode(tok, [{"role": "user", "content": q}, {"role": "assistant", "content": a}], cfg.seq_len)
            xs.append(x)
            ys.append(y)
    return torch.stack(xs), torch.stack(ys)                    # 偶数行是 chosen，奇数行是 rejected


def logp(model, x, y):
    """每条样本在"助手回答"那些位置上的 log p 之和"""
    lp = torch.log_softmax(model(x).float(), dim=-1)
    token_lp = lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_lp * (y != IGNORE)).sum(-1)


train, test = pairs[:-100], pairs[-100:]
x_te, y_te = batch_of(test)


@torch.no_grad()
def judge():
    """测试集上：模型给原文的概率高于给自己那条的比例，以及两者的平均差距"""
    policy.eval()
    lp = logp(policy, x_te, y_te)
    policy.train()
    margin = lp[0::2] - lp[1::2]
    return (margin > 0).float().mean().item(), margin.mean().item()


beta, steps, batch, lr = 0.1, 240, 8, 2e-5
opt = torch.optim.AdamW(policy.parameters(), lr=lr, betas=(0.9, 0.95))
idx_gen = torch.Generator().manual_seed(4)
acc, margin = judge()
print(f"\n训练前：原文更可能的比例 {acc:.1%}，平均差距 {margin:+.2f}")
print(f"DPO {steps} 步 × {batch} 对，β={beta}，学习率 {lr:g}")
print("  步    DPO loss   原文更可能    平均差距")
running = 0.0
for step in range(steps):
    i = torch.randint(0, len(train), (batch,), generator=idx_gen).tolist()
    x, y = batch_of([train[k] for k in i])
    pi, re = logp(policy, x, y), logp(ref, x, y)
    logits = (pi[0::2] - pi[1::2]) - (re[0::2] - re[1::2])      # 策略拉开的差距，减去参考模型本来就有的差距
    loss = -F.logsigmoid(beta * logits).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step()
    running += loss.item()
    if (step + 1) % 60 == 0:
        acc, margin = judge()
        print(f"{step + 1:4d}    {running / 60:8.4f}   {acc:9.1%}    {margin:+8.2f}")
        running = 0.0
