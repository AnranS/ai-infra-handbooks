# From scratch (5): cheaper tuning, better alignment

<p class="lead">The last chapter turned the base model into a small assistant that answers in the right format. Real post-training has three more pieces: changing behaviour without touching all the weights (LoRA), making the model prefer the better of two replies (DPO), and moving a large model's skill into a small one (distillation). This chapter writes all three from scratch and produces real numbers for each on the same 1.8M-parameter model; at the end it exports the model into LLaMA's weight layout so that off-the-shelf inference frameworks can load it directly.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which layers does LoRA attach to? Why is B initialized to zero? Does it slow the model down at inference?
    2. What do the rank $r$ and the scale $\alpha$ each control?
    3. Why does DPO need a "reference model"? What happens without it?
    4. What do larger and smaller values of DPO's $\beta$ mean?
    5. Why does white-box distillation put a temperature on the logits? What is the $T^2$ in the loss for?

??? success "Answers (try first, then expand to compare)"
    1. Usually to attention's q/k/v and output projections (sometimes the MLP too). $B=0$ makes $BA=0$, so the moment it is attached the model's output is unchanged and training starts from "the original model" rather than from a random perturbation. At inference you add $BA$ back into $W$ (merging), which costs nothing; unmerged, each layer does two extra small matrix multiplies.
    2. $r$ controls the rank of the change, which is its capacity; $\alpha/r$ is a scale that keeps the size of the update roughly constant when $r$ changes, which is why $\alpha$ is usually adjusted along with $r$.
    3. The reference model is the frozen copy from the start of training. DPO optimizes "the gap the policy opens up relative to the reference", which acts as a regularizer: without it the model can lower the loss by pushing the probability of *both* replies down, and it degenerates into nonsense very quickly.
    4. The larger $\beta$ is, the less the policy is allowed to drift from the reference (more conservative); the smaller it is, the more aggressive and the easier to over-optimize. Around 0.1 is common.
    5. The temperature flattens the distribution so that the student learns not just "the largest one" but the relative sizes of the next few — that is the dark knowledge. The gradient picks up a factor of $1/T^2$, and multiplying by $T^2$ is what makes the distillation term's gradient comparable in scale to the cross-entropy term's.

## LoRA: training 1% of the parameters {#lora只训练-1-的参数}

Full fine-tuning keeps a gradient and two Adam states per parameter; for a 1.8M model that is nothing, for a 7B model it is over 100 GB. **LoRA**'s idea is that the weight change $\Delta W$ from fine-tuning is usually low-rank, so instead of a full-rank $\Delta W$, store two thin matrices:

$$W' x = Wx + \frac{\alpha}{r} BA\,x,\qquad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r},\; r \ll d$$

$W$ stays frozen and only $A$ and $B$ are trained. This section gives the model a **new answer format**: still "who said this", but the answer has to be a complete sentence. The fine-tuned model only blurts out names, which makes it a good test of whether LoRA can change behaviour with 1% of the parameters.

```python title="lora.py"
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


# a new task: still "who said it", but the answer has to be a full sentence; the fine-tuned model only blurts out names
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

steps, batch, lr = 200, 16, 3e-3                               # training only the LoRA weights, the learning rate can be an order of magnitude larger than full fine-tuning
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
for block in model.blocks:                                     # merge: add BA back into the weights and the model is its old self again
    block.attn.qkv, block.attn.proj = block.attn.qkv.merge(), block.attn.proj.merge()
with torch.no_grad():
    after = model(probe)
print(f"合并回权重后最大误差 {(before - after).abs().max():.2e}，参数量回到 {sum(p.numel() for p in model.parameters()):,}")
```

```text title="output"
微调好的模型遇到这个新要求：格式对 0%、人名也对 0%；它答的是「玄德」
挂上 rank=8 的 LoRA：24,576 个可训练参数 / 共 1,828,224 个 = 1.34%
B 初始化成全零，挂上的那一刻行为不变：格式对还是 0%

只训练这 24,576 个参数（400 条新格式的数据，200 步 × 16 条）
  步   训练 loss   格式对   人名也对   它答的是
  50      1.711     100%        16%   「这是操说的。」
 100      0.407       0%         0%   「说是孔明说的。」
 150      0.387     100%        18%   「这是孔明说的。」
 200      0.377      98%        18%   「这是操说的。」
合并回权重后最大误差 1.24e-05，参数量回到 1,803,648
```

- **$B=0$ is the key**: the instant LoRA is attached, $BA=0$ and the model's output is identical, so training starts from "the original model". If both matrices were random, attaching them would be adding noise;
- **Changing behaviour is cheap, changing knowledge is not**: the format is learned in 50 steps (0% → 100%), but "is the name right" stops at 18% — the previous chapter scored 23.5% on the same 200 questions when asking for a bare name, so the new format even costs a little. A low-rank perturbation can change **how it says things**, not **what it knows**. That is exactly LoRA's place: **adaptation**, not **injecting knowledge**;
- **It wobbles on the way**: at step 100 it writes 「说是孔明说的。」, which a strict regular expression scores as 0. With only 24k trainable parameters and 400 samples, training is not smooth; a few dozen steps later it settles. Real projects likewise pick the checkpoint by a validation metric rather than taking the last step;
- **Merging costs nothing afterwards**: $W \leftarrow W + \frac{\alpha}{r}BA$ puts the model back to its original shape, as fast at inference as if LoRA had never been there. Not merging is fine too, and has the advantage that one base model can carry many adapters and switch per request (that is what multi-LoRA in inference frameworks does, see [inference systems](serving://));
- **Which layers**: here it is attached to attention's `qkv` and output projection, the most common choice. Some implementations attach only to square layers ($d_\text{in} = d_\text{out}$), some to the MLP as well; the more you attach, the more trainable parameters and usually the slightly better the result.

## DPO: making the model prefer the better reply {#dpo让模型偏向更好的那个回答}

SFT only tells the model "this is the right answer"; it never says what makes a reply *better*. **RLHF** does it by sampling several replies → having a human (or a model) say which is better → training a reward model → optimizing with PPO. **DPO** drops the two middle steps and trains the policy on preference pairs directly:

$$\mathcal{L} = -\log \sigma\Big(\beta \big[(\log \pi(y_w|x) - \log \pi(y_l|x)) - (\log \pi_\text{ref}(y_w|x) - \log \pi_\text{ref}(y_l|x))\big]\Big)$$

Inside the bracket is "the gap the policy opens up" minus "the gap the reference model already had". We build the preference data like this: the question is a continuation task, **chosen is the original next sentence**, and **rejected is a sentence the model sampled itself** — which is the real RLHF loop (the model samples, something external scores), with "compare against the original text" standing in for human scoring.

```python title="dpo.py"
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
ref = copy.deepcopy(policy).eval().requires_grad_(False)       # the reference model: the frozen copy, unchanged throughout training


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
    return torch.stack(xs), torch.stack(ys)                    # even rows are chosen, odd rows are rejected


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
    logits = (pi[0::2] - pi[1::2]) - (re[0::2] - re[1::2])      # the gap the policy opens up, minus the gap the reference model already had
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
```

```text title="output"
324 对偏好数据：chosen 是原文的下一句，rejected 是模型自己采样出来的
  问：接下来写：向者不克而还，盖因羌兵不至。
  chosen  ：今可先遣人会羌人于陇右，然后进兵出石营，从董亭直取南安。
  rejected：夫人亦妻子枭失，可王基便行。

训练前：原文更可能的比例 38.0%，平均差距 -22.63
DPO 240 步 × 8 对，β=0.1，学习率 2e-05
  步    DPO loss   原文更可能    平均差距
  60      0.5761       42.0%      -18.98
 120      0.4081       42.0%      -16.14
 180      0.3148       43.0%      -14.01
 240      0.2265       43.0%      -12.39
```

- **The reference model is the frozen copy**: it only runs forward and is never updated. Without it the model can lower the loss by pushing both replies' probabilities down, and it starts talking nonsense fast;
- **Watch the gap, not just the ratio**: before training the model gives the original text 22.6 less log-prob than its own sentence; after 240 steps that is down to 12.4, and the DPO loss falls from 0.58 to 0.23 — that is the quantity that is really moving. The share where "the original is more likely" only climbs from 38% to 43%: rejected was **sampled by the model itself**, so its probability is naturally high, and flipping the ordering outright takes far more data and steps. Keep pushing and the share keeps rising, but the format SFT taught starts to come loose — real DPO is the same balancing act between alignment and degeneration, which is why you watch generated samples and not just the loss;
- **The learning rate has to be tiny** (2e-5 here): DPO is a small adjustment to an already trained model, and a larger rate will wreck the format SFT taught it;
- **Two sequences per forward pass**: chosen and rejected go in the same batch, and both the policy and the reference run once, so each step costs four times an SFT step. Real implementations **precompute and cache** the reference model's log-probs, which saves half of it;
- For RL proper (GRPO, PPO, online sampling and reward models) see the [distributed training handbook](train://algo/rl-algorithms/).

## Distillation: learning the teacher's distribution {#蒸馏让小模型学大模型的分布}

Given the same data, how much difference is there between a small model learning on its own and one learning from a large model? **White-box distillation** takes the teacher's whole output distribution as the target:

$$\mathcal{L} = \alpha \cdot \mathrm{CE}(y) + (1-\alpha)\, T^2 \cdot \mathrm{KL}\big(p_\text{student}^{(T)} \,\|\, p_\text{teacher}^{(T)}\big)$$

Cross-entropy looks only at "the one correct token"; the KL term also teaches the student what the second and third most likely ones are — the so-called dark knowledge. The temperature $T$ flattens the distribution, and $T^2$ brings the distillation term's gradient back to a scale comparable with the cross-entropy's.

```python title="distill.py"
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
                          soft.flatten(0, 1), reduction="batchmean")      # one distribution per position, averaged over tokens
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
```

```text title="output"
老师：1.80M 参数，验证 loss 6.089
学生只学正确答案（普通训练）：0.62M 参数，验证 loss 6.495
学生 + 蒸馏（α=0.5，T=2）：验证 loss 6.360
学生 + 蒸馏（α=0.0，T=2）：验证 loss 6.280
```

- **The teacher is the 1.8M model from chapter 2**, the student has only 0.62M (`d=64`, 2 layers), and the data, steps and learning rate are the same as chapter 3's scaling experiment; the only difference is the extra KL term in the loss;
- **Both $\alpha=0$ (learn only from the teacher) and $\alpha=0.5$ (half and half)** beat plain cross-entropy: for the same compute, the student ends up closer to the teacher. But it will not surpass the teacher — distillation transfers the distribution the teacher already has;
- Real distillation comes in two kinds: **white-box** (you have the teacher's logits, as here) and **black-box** (you only have the text the teacher generates, which is "synthesize data with a large model and then do SFT"). A large share of the instruction data behind open small models is black-box distillation;
- Like quantization and pruning, distillation is a standard way to make a model smaller; see the LLM handbook's [quantization](llm://inference/quantization/).

## Exporting: handing it to an inference framework {#导出交给现成的推理框架}

A trained model has to be usable by other people. Our architecture (pre-norm + RMSNorm, SwiGLU, RoPE, bias-free linear layers, tied embeddings) is exactly LLaMA's, so **renaming the weights is all it takes** to turn it into a standard `LlamaForCausalLM` checkpoint:

```python title="export.py"
"""导出成 LLaMA 的权重布局：改个名字，现成的推理框架就能加载它"""
import json
import shutil
from pathlib import Path

import torch

from chat import IM_END, IM_START, ROLE, load_tokenizer
from model import GPT, GPTConfig

out = Path("minisanguo")
out.mkdir(exist_ok=True)
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
src, dst = sft["model"], {}
head_dim, hidden = cfg.d_model // cfg.n_head, None

for key, w in src.items():
    if key == "embed.weight":
        dst["model.embed_tokens.weight"] = w
    elif key == "head.weight":
        dst["lm_head.weight"] = w                              # with tied embeddings transformers binds this itself, but writing it out does no harm
    elif key == "norm.weight":
        dst["model.norm.weight"] = w
    else:
        i, rest = key.split(".")[1], key.split(".", 2)[2]
        p = f"model.layers.{i}."
        if rest == "norm1.weight":
            dst[p + "input_layernorm.weight"] = w
        elif rest == "norm2.weight":
            dst[p + "post_attention_layernorm.weight"] = w
        elif rest == "attn.qkv.weight":                        # we fused q, k and v into one matrix; split it back into three
            q, k, v = w.chunk(3, dim=0)
            dst[p + "self_attn.q_proj.weight"], dst[p + "self_attn.k_proj.weight"] = q, k
            dst[p + "self_attn.v_proj.weight"] = v
        elif rest == "attn.proj.weight":
            dst[p + "self_attn.o_proj.weight"] = w
        elif rest == "mlp.gate_up.weight":                     # SwiGLU's gate and up are fused too
            gate, up = w.chunk(2, dim=0)
            dst[p + "mlp.gate_proj.weight"], dst[p + "mlp.up_proj.weight"] = gate, up
            hidden = gate.shape[0]
        elif rest == "mlp.down.weight":
            dst[p + "mlp.down_proj.weight"] = w

tok = load_tokenizer()
TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'assistant' %}{{ m['content'] + '" + IM_END + "\\n' }}"
            "{% else %}{{ '" + IM_START + "' + {'system': '系统', 'user': '用户'}[m['role']] + '\\n' + m['content']"
            " + '" + IM_END + "\\n' }}{% if m['role'] == 'user' %}{{ '" + IM_START + "助手\\n' }}{% endif %}"
            "{% endif %}{% endfor %}")
config = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": cfg.d_model,
    "intermediate_size": hidden, "num_hidden_layers": cfg.n_layer, "num_attention_heads": cfg.n_head,
    "num_key_value_heads": cfg.n_head, "head_dim": head_dim, "vocab_size": cfg.vocab_size,
    "max_position_embeddings": cfg.seq_len, "rope_theta": cfg.rope_theta, "hidden_act": "silu",
    "rms_norm_eps": torch.finfo(torch.float32).eps,            # this is what nn.RMSNorm uses when eps is not given
    "tie_word_embeddings": True, "attention_bias": False, "mlp_bias": False, "torch_dtype": "float32",
    "bos_token_id": tok.token_to_id(IM_START), "eos_token_id": tok.token_to_id(IM_END),
}
(out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
(out / "generation_config.json").write_text(json.dumps({"eos_token_id": config["eos_token_id"]}, indent=1), encoding="utf-8")
(out / "tokenizer_config.json").write_text(json.dumps({
    "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": IM_START, "eos_token": IM_END, "pad_token": IM_END,
    "unk_token": "<unk>", "model_max_length": cfg.seq_len, "chat_template": TEMPLATE}, ensure_ascii=False, indent=1),
    encoding="utf-8")
shutil.copy("tokenizer_chat.json", out / "tokenizer.json")
torch.save(dst, out / "pytorch_model.bin")

print(f"{len(src)} 个张量 → {len(dst)} 个（qkv 拆成 3 份、gate_up 拆成 2 份）")
for name in list(dst)[:3] + ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.gate_proj.weight"]:
    print(f"  {name:48s} {tuple(dst[name].shape)}")
print("目录内容：", "、".join(sorted(p.name for p in out.iterdir())),
      f"（共 {sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB）")

model = GPT(cfg)                                               # load it back into our own model to confirm that no tensor went missing
back = {"embed.weight": dst["model.embed_tokens.weight"], "head.weight": dst["lm_head.weight"],
        "norm.weight": dst["model.norm.weight"]}
for i in range(cfg.n_layer):
    p = f"model.layers.{i}."
    back[f"blocks.{i}.norm1.weight"] = dst[p + "input_layernorm.weight"]
    back[f"blocks.{i}.norm2.weight"] = dst[p + "post_attention_layernorm.weight"]
    back[f"blocks.{i}.attn.qkv.weight"] = torch.cat([dst[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
    back[f"blocks.{i}.attn.proj.weight"] = dst[p + "self_attn.o_proj.weight"]
    back[f"blocks.{i}.mlp.gate_up.weight"] = torch.cat([dst[p + f"mlp.{n}_proj.weight"] for n in ("gate", "up")])
    back[f"blocks.{i}.mlp.down.weight"] = dst[p + "mlp.down_proj.weight"]
model.load_state_dict(back)
x = torch.randint(0, cfg.vocab_size, (2, 32))
with torch.no_grad():
    ref = GPT(cfg)
    ref.load_state_dict(src)
    print(f"装回来再前向一遍，最大误差 {(model(x) - ref(x)).abs().max():.2e}")
```

```text title="output"
27 个张量 → 39 个（qkv 拆成 3 份、gate_up 拆成 2 份）
  model.embed_tokens.weight                        (8194, 128)
  model.layers.0.input_layernorm.weight            (128,)
  model.layers.0.post_attention_layernorm.weight   (128,)
  model.layers.0.self_attn.q_proj.weight           (128, 128)
  model.layers.0.mlp.gate_proj.weight              (320, 128)
目录内容： config.json、generation_config.json、pytorch_model.bin、tokenizer.json、tokenizer_config.json （共 7.6 MB）
装回来再前向一遍，最大误差 0.00e+00
```

Four things go into the exported directory, and none of them is optional:

- **`config.json`**: the architectural hyperparameters. `architectures` decides which implementation loads it; `tie_word_embeddings` tells the framework that the input and output embeddings are shared; `rms_norm_eps` has to match training, or the outputs drift slightly;
- **The weights**: the names have to line up (`model.layers.0.self_attn.q_proj.weight` and so on). For speed we fused q/k/v into one matrix and gate with up into another, so exporting has to split them — and, going the other way, many inference frameworks **fuse** them again on load, because the fused matrix multiply is faster;
- **The tokenizer**: `tokenizer.json` plus `tokenizer_config.json`, whose `chat_template` is a Jinja template that decides how `apply_chat_template` lays a conversation out as text — it must be **exactly** the template training used, down to the newlines;
- **`generation_config.json`**: generation defaults such as the stop token.

With that directory you can load it with transformers, or hand it straight to vLLM / SGLang for an OpenAI-compatible server:

```python title="serve.py" run="no"
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("minisanguo")
model = AutoModelForCausalLM.from_pretrained("minisanguo")
text = tok.apply_chat_template([{"role": "user", "content": "这句话是谁说的：「吾自有计。」"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=20)[0, ids.shape[1]:], skip_special_tokens=True))

# or start a server (needs a GPU):
#   vllm serve ./minisanguo --max-model-len 128
#   python -m sglang.launch_server --model-path ./minisanguo --context-length 128
```

That completes the road from an empty directory to a conversational model an inference framework can serve. The model itself has 1.8M parameters and answers haltingly, but every step — corpus, tokenizer, model, training loop, scaling, instruction tuning, preference alignment, distillation, export — is the same one the real pipeline takes. What is left is to scale the data and the compute by a few orders of magnitude, and to move to a real GPU (the next chapter).

!!! interview "How to explain it"
    To explain "what post-training consists of": **SFT** teaches the format and the tasks → **preference alignment** (RLHF / DPO) teaches which reply is better → when memory is tight, **LoRA** (freeze the base, train only the low-rank $BA$, initialize $B$ to zero so the starting point is unchanged, merge afterwards for zero inference cost) → to make it small, **distill** (white-box from the logits, black-box from the generated text). The crux of DPO is that **reference model**: what is optimized is "the gap relative to the reference", otherwise the model cheats by pushing all the probabilities down; $\beta$ controls how far it may drift. Finish with **export**: once the architecture matches LLaMA / Qwen, renaming the weights plus a config and a chat template is enough for vLLM / SGLang to serve it.

## Exercises {#练习}

**1. The effect of rank.** Change LoRA's `rank` to 1, 4 and 32. How many trainable parameters does each give, and how many steps does learning the new format take?

??? success "An approach"
    The parameter count is proportional to the rank (24,576 at rank 8, 3,072 at rank 1). "Switch to another output format" is a very low-rank change, so rank 1 usually learns it too, just more slowly; the more complex the thing to learn (another language, a new domain's phrasing), the higher the rank needed. Note that changing the rank changes the $\alpha/r$ scale, so keeping $\alpha = 2r$ preserves the magnitude.

**2. Drop the reference model.** Remove the reference term from the DPO loss (keep only $\beta(\log\pi(y_w) - \log\pi(y_l))$), train the same number of steps, then ask it a few questions.

??? success "An approach"
    The loss falls faster, but the model degenerates quickly: pushing both replies' probabilities down is the cheapest direction for that loss. The generated text gets shorter, repetitive, even empty. The reference term is what pins down "how much has been gained relative to the starting point", which is what DPO actually optimizes.

**3. The distillation temperature.** Change `temperature` to 1 and 5. How does the validation loss move? Then remove the $T^2$ and look again.

??? success "An approach"
    At $T=1$ the KL degenerates into "stare at the teacher's largest entry" and little dark knowledge gets through; too large a $T$ flattens the distribution until nothing is learned, so 2 to 4 is the usual range. Without the $T^2$, the distillation term's gradient is $T^2$ times smaller, which quietly lowers its weight, and the mix with cross-entropy is no longer the ratio you chose.

**4. Run the exported model.** Install `transformers`, load the exported directory with `AutoModelForCausalLM.from_pretrained("minisanguo")`, build the prompt with `apply_chat_template`, generate, and check it matches our own `answer()`.

??? success "An approach"
    Matching is what proves the export is right. When it does not match, check in this order: whether the **weight names** line up (`load_state_dict`'s `missing_keys` / `unexpected_keys`), whether `rms_norm_eps` and `rope_theta` in the **config** are the ones used in training, and whether the text the **chat template** renders is character-for-character what training laid out. The last one goes wrong most often and is the easiest to overlook.

## Summary {#小结}

- [x] **LoRA**: freeze the base and train only the low-rank $BA$; initializing $B$ to zero keeps the starting point unchanged; merging afterwards costs nothing at inference. It changes **behaviour** cheaply and **knowledge** hardly at all.
- [x] **DPO**: optimize the policy directly on preference pairs, subtracting the gap the reference model already had, or the model cheats by pushing all probabilities down; $\beta$ controls how far it may drift; the learning rate is another order of magnitude below SFT's.
- [x] Real preference data comes from "the model samples, something external scores"; we used the original text as the gold standard, but the loop is the same.
- [x] **White-box distillation**: fit the teacher's whole temperature-softened distribution with KL, multiplied by $T^2$ so the gradient scale matches cross-entropy's; the student gets closer to the teacher but never surpasses it. Black-box distillation is "synthesize data with a large model and then do SFT".
- [x] **Export**: once the architecture matches LLaMA / Qwen, renaming the weights plus writing `config.json` and the `chat_template` is enough for transformers, vLLM and SGLang to load it; the chat template must be exactly the one used in training.
