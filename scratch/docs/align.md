# 从零训练（五）：更省地调、对得更齐

<p class="lead">上一章把基座模型变成了一个会按格式回答的小助手。真实的后训练还有三件事：想换个行为但不想动全部权重（LoRA）、想让模型在两个回答里偏向更好的那个（DPO）、想把大模型的本事搬进小模型（蒸馏）。这一章把三件都从零写一遍，每一件都在同一个 1.8M 的模型上跑出真实的数字；最后把模型导出成 LLaMA 的权重布局，让现成的推理框架能直接加载它。</p>

!!! note "这一章跑什么"
    **主线**：`export.py`（约 2 秒）→ 产出 `minisanguo/`，可以交给 transformers、vLLM、SGLang。
    **实验**：`lora.py`（约 1 分钟）、`dpo.py`（约 1.7 分钟）、`distill.py`（约 3.3 分钟），三件各自独立，挑着跑也行。

!!! question "自测：能答上来就可以跳过本章"
    1. LoRA 挂在哪些层上？为什么 B 要初始化成全零？推理时它会让模型变慢吗？
    2. 秩 $r$ 和缩放 $\alpha$ 分别控制什么？
    3. DPO 为什么需要一个"参考模型"？没有它会怎样？
    4. DPO 的 $\beta$ 调大调小，分别意味着什么？
    5. 白盒蒸馏为什么要给 logits 加温度？损失里的 $T^2$ 是做什么的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 常挂在注意力的 q/k/v 和输出投影上（有时也挂 MLP）。$B=0$ 让 $BA=0$，挂上的那一刻模型输出完全不变，训练从"原模型"出发而不是从一个随机扰动出发。推理时把 $BA$ 加回 $W$ 就行（合并），没有任何额外开销；不合并则每层多两次小矩阵乘。
    2. $r$ 控制这个"改动"的秩，也就是容量；$\alpha/r$ 是缩放，用来在换 $r$ 时保持更新的尺度大致不变，所以调 $r$ 时一般同时调 $\alpha$。
    3. 参考模型是训练开始时那一份冻住的模型。DPO 优化的是"策略相对参考模型拉开的差距"，它起到正则的作用：没有它，模型可以靠把两个回答的概率都压低来降低损失，很快退化成胡说。
    4. $\beta$ 越大，越不允许策略偏离参考模型（更保守）；越小，越激进、越容易过度优化。常用 0.1 左右。
    5. 温度把分布变平滑，让学生不仅学到"最大的那一个"，还能学到次大的几项之间的相对关系——这才是"暗知识"。梯度里会多出一个 $1/T^2$ 的因子，乘上 $T^2$ 之后，蒸馏项和交叉熵项的梯度尺度才可比。

## LoRA：只训练 1% 的参数

全量微调要给每个参数都存一份梯度和两份 Adam 状态；1.8M 的模型无所谓，7B 的模型就是 100 GB 以上。**LoRA** 的想法是：微调带来的权重改动 $\Delta W$ 通常是低秩的，那就别存满秩的 $\Delta W$，只存两个瘦长的矩阵：

$$W' x = Wx + \frac{\alpha}{r} BA\,x,\qquad A \in \mathbb{R}^{r \times d_\text{in}},\; B \in \mathbb{R}^{d_\text{out} \times r},\; r \ll d$$

$W$ 冻住不动，只训练 $A$ 和 $B$。这一节给模型换一个**新的回答格式**：同样是"这句话是谁说的"，但要求用一句完整的话回答。微调过的模型只会蹦人名，正好用来看 LoRA 能不能用 1% 的参数把行为改过来。

```python title="lora.py" ci="loose"
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
```

```text title="输出"
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

- **$B=0$ 是关键**：挂上 LoRA 的那一刻 $BA=0$，模型输出一字不差，训练是从"原模型"出发的。如果两个矩阵都随机初始化，一开始就相当于给模型加了噪声；
- **改行为很便宜，改知识很难**：50 步就把格式学会了（0% → 100%），但「人名对不对」停在 18%——上一章同样这 200 道题、直接问人名时是 23.5%，换成新格式之后反而略降。低秩的扰动能改**怎么说**，改不了**知道什么**。这正是 LoRA 的定位：**适配**，不是**灌知识**；
- **中间会抖**：第 100 步它写成「说是孔明说的。」，严格按正则判格式就算 0 分。只有 24k 个可训练参数、400 条数据，训练本来就不平滑，再走几十步就稳住了；真实项目里同样要按验证指标选 checkpoint，而不是取最后一步；
- **合并之后没有额外开销**：$W \leftarrow W + \frac{\alpha}{r}BA$，模型结构回到原样，推理时和没挂过一样快。不合并也行，好处是同一个基座可以挂很多个适配器、按请求切换（推理框架里的 multi-LoRA 就是这么做的，见[推理系统](serving://)）；
- **挂在哪些层**：这里挂在注意力的 `qkv` 和输出投影上，是最常见的选择。有的实现只挂方阵（$d_\text{in} = d_\text{out}$ 的线性层），有的连 MLP 一起挂；挂得越多，可训练参数越多，效果通常也更好一点。

## DPO：让模型偏向更好的那个回答

SFT 只告诉模型"这是正确答案"，没有告诉它"什么样的回答更好"。**RLHF** 的做法是：采样若干回答 → 人（或模型）标注哪个更好 → 训练奖励模型 → 用 PPO 优化。**DPO** 把中间两步省掉，直接用偏好对训练策略：

$$\mathcal{L} = -\log \sigma\Big(\beta \big[(\log \pi(y_w|x) - \log \pi(y_l|x)) - (\log \pi_\text{ref}(y_w|x) - \log \pi_\text{ref}(y_l|x))\big]\Big)$$

括号里是"策略拉开的差距"减去"参考模型本来就有的差距"。偏好数据我们这样造：问题是续写题，**chosen 用原文的下一句**，**rejected 用模型自己采样出来的那一句**——这就是真实 RLHF 的流程（模型自采样 + 外部打分），只是把"人工打分"换成了"和原文比"。

```python title="dpo.py" ci="loose"
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
```

```text title="输出"
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

- **参考模型是冻住的那一份**：它只前向、不更新。少了它，模型可以把两个回答的概率一起压低来降低损失，很快就开始胡说；
- **看差距，不要只看比例**：训练前模型给原文的 log-prob 比给自己写的低 22.6，训练 240 步后收到 12.4，DPO loss 从 0.58 降到 0.23——这是真正在动的量。而「原文更可能」的比例只从 38% 爬到 43%：rejected 是模型**自己采样**出来的，概率天生就高，要把排序整个翻过来需要多得多的数据和步数。继续硬训比例还会涨，但 SFT 学到的格式也会开始松动——真实的 DPO 同样是在对齐和退化之间找平衡，所以要盯着生成样例，而不是只盯 loss；
- **学习率要很小**（这里 2e-5）：DPO 是在一个已经训好的模型上做小幅调整，学习率大一点就会把 SFT 学到的格式带崩；
- **一次前向两条序列**：chosen 和 rejected 拼在同一个 batch 里，策略和参考各算一次，所以每步的计算量是 SFT 的四倍。真实实现会把参考模型的 log-prob **预先算好存起来**，省掉一半；
- 更进一步的 RL（GRPO、PPO、在线采样与奖励模型）见[分布式训练手册](train://algo/rl-algorithms/)。

## 蒸馏：让小模型学大模型的分布

同样的数据，小模型自己学，和"跟着大模型学"，差别有多大？**白盒蒸馏**把老师的整个输出分布当成目标：

$$\mathcal{L} = \alpha \cdot \mathrm{CE}(y) + (1-\alpha)\, T^2 \cdot \mathrm{KL}\big(p_\text{学生}^{(T)} \,\|\, p_\text{老师}^{(T)}\big)$$

交叉熵只看"正确的那一个 token"，KL 这一项还把"第二、第三可能是什么"也教给学生——这就是所谓的暗知识。温度 $T$ 把分布拉平，$T^2$ 用来把蒸馏项的梯度尺度拉回和交叉熵可比。

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
```

```text title="输出"
老师：1.80M 参数，验证 loss 6.089
学生只学正确答案（普通训练）：0.62M 参数，验证 loss 6.495
学生 + 蒸馏（α=0.5，T=2）：验证 loss 6.360
学生 + 蒸馏（α=0.0，T=2）：验证 loss 6.280
```

- **老师是第二章训好的 1.8M 模型**，学生只有 0.62M（`d=64`、2 层），训练数据、步数、学习率都和第三章的 scaling 实验一致，唯一的差别是损失里多了 KL 这一项；
- **$\alpha=0$（只学老师）和 $\alpha=0.5$（一半一半）**都比纯交叉熵好：同样的算力，学生更靠近老师。但学生不会超过老师——蒸馏传的是老师已有的分布；
- 真实的蒸馏有两类：**白盒**（能拿到老师的 logits，像这里）和**黑盒**（只能拿到老师生成的文本，就是"用大模型合成数据再做 SFT"）。开源小模型的指令数据里，有很大一部分就是黑盒蒸馏来的；
- 蒸馏和量化、剪枝一样，是把模型做小的常规手段，见大模型手册的[量化](llm://inference/quantization/)。

## 导出：交给现成的推理框架

训好的模型要能被别人用。我们的结构（pre-norm + RMSNorm、SwiGLU、RoPE、不带 bias 的线性层、共享词嵌入）和 LLaMA 完全一致，所以**只要改权重的名字**，就能变成一个标准的 `LlamaForCausalLM` 检查点：

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
        dst["lm_head.weight"] = w                              # 共享词嵌入时 transformers 会自己绑定，写出来也无妨
    elif key == "norm.weight":
        dst["model.norm.weight"] = w
    else:
        i, rest = key.split(".")[1], key.split(".", 2)[2]
        p = f"model.layers.{i}."
        if rest == "norm1.weight":
            dst[p + "input_layernorm.weight"] = w
        elif rest == "norm2.weight":
            dst[p + "post_attention_layernorm.weight"] = w
        elif rest == "attn.qkv.weight":                        # 我们把 q、k、v 合成了一个矩阵，拆回三个
            q, k, v = w.chunk(3, dim=0)
            dst[p + "self_attn.q_proj.weight"], dst[p + "self_attn.k_proj.weight"] = q, k
            dst[p + "self_attn.v_proj.weight"] = v
        elif rest == "attn.proj.weight":
            dst[p + "self_attn.o_proj.weight"] = w
        elif rest == "mlp.gate_up.weight":                     # SwiGLU 的 gate 和 up 也是合在一起的
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
    "rms_norm_eps": torch.finfo(torch.float32).eps,            # nn.RMSNorm 不传 eps 时就用它
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

model = GPT(cfg)                                               # 反过来装回我们自己的结构，确认一个张量都没丢
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

```text title="输出"
27 个张量 → 39 个（qkv 拆成 3 份、gate_up 拆成 2 份）
  model.embed_tokens.weight                        (8194, 128)
  model.layers.0.input_layernorm.weight            (128,)
  model.layers.0.post_attention_layernorm.weight   (128,)
  model.layers.0.self_attn.q_proj.weight           (128, 128)
  model.layers.0.mlp.gate_proj.weight              (320, 128)
目录内容： config.json、generation_config.json、pytorch_model.bin、tokenizer.json、tokenizer_config.json （共 7.6 MB）
装回来再前向一遍，最大误差 0.00e+00
```

导出目录里有四样东西，缺一不可：

- **`config.json`**：结构超参数。`architectures` 决定用哪一个实现来加载它；`tie_word_embeddings` 告诉框架输入输出共享词嵌入；`rms_norm_eps` 要和训练时一致，否则输出会有细微偏差；
- **权重**：名字必须对上（`model.layers.0.self_attn.q_proj.weight` 这一套）。我们为了快把 q/k/v 合成了一个矩阵、gate 和 up 也合在一起，导出时要拆开——反过来，很多推理框架加载时又会把它们**合并**回去，因为合并的矩阵乘更快；
- **分词器**：`tokenizer.json` 加上 `tokenizer_config.json`，后者里的 `chat_template` 是一段 Jinja 模板，决定 `apply_chat_template` 怎么把对话拼成文本——必须和训练时用的模板**一模一样**，差一个换行都会让效果掉下来；
- **`generation_config.json`**：停止 token 之类的生成默认值。

有了这个目录，就可以交给 transformers 加载，或者直接让 vLLM / SGLang 起一个 OpenAI 接口的服务：

```python title="serve.py" run="no"
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("minisanguo")
model = AutoModelForCausalLM.from_pretrained("minisanguo")
text = tok.apply_chat_template([{"role": "user", "content": "这句话是谁说的：「吾自有计。」"}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, max_new_tokens=20)[0, ids.shape[1]:], skip_special_tokens=True))

# 或者起一个服务（需要 GPU）：
#   vllm serve ./minisanguo --max-model-len 128
#   python -m sglang.launch_server --model-path ./minisanguo --context-length 128
```

到这里，"从一个空目录到一个能被推理框架服务的对话模型"就走完了。模型本身只有 1.8M 参数、答得磕磕绊绊，但每一步——语料、分词器、模型、训练循环、扩容、指令微调、偏好对齐、蒸馏、导出——和真实的流程是同一条。接下来要做的只是把数据和算力各放大几个数量级，以及换一张真卡（下一章）。

!!! interview "怎么讲清楚"
    讲"后训练都有哪些环节"：**SFT** 学格式和任务 → **偏好对齐**（RLHF / DPO）学"哪个回答更好" → 需要省显存时用 **LoRA**（冻住基座，只训练低秩的 $BA$，$B$ 初始化为零保证起点不变，合并后推理零开销）→ 要做小就**蒸馏**（白盒学 logits 分布、黑盒学生成的文本）。DPO 的要点是那个**参考模型**：优化的是"相对参考模型拉开的差距"，否则模型会靠压低所有概率来作弊；$\beta$ 控制允许偏离多远。最后讲**导出**：结构对齐 LLaMA / Qwen 之后，改权重名 + 写 config 和 chat_template，就能交给 vLLM / SGLang 服务。

## 练习

**1. 秩的影响。** 把 LoRA 的 `rank` 改成 1、4、32，可训练参数各是多少？学会新格式分别要多少步？

??? success "参考思路"
    参数量和 rank 成正比（这里 rank=8 是 24,576 个，rank=1 就是 3,072 个）。"换一个输出格式"是很低秩的改动，rank=1 往往也学得会，只是慢一些；要学的东西越复杂（多语言、新领域的表达方式），需要的秩越大。注意换 rank 时 $\alpha/r$ 这个缩放会变，一般让 $\alpha = 2r$ 保持尺度。

**2. 拿掉参考模型。** 把 DPO 损失里参考模型那一项去掉（只留 $\beta(\log\pi(y_w) - \log\pi(y_l))$），训练同样的步数，再问它几个问题。

??? success "参考思路"
    loss 会降得更快，但模型很快开始退化：把两个回答的概率一起往下压，是降低这个损失最省力的方向。生成出来的文字会变短、变重复甚至空。参考模型那一项把"相对于起点拉开了多少"固定住，才是 DPO 真正优化的量。

**3. 蒸馏的温度。** 把 `temperature` 改成 1 和 5，验证 loss 怎么变？去掉损失里的 $T^2$ 再看一次。

??? success "参考思路"
    $T=1$ 时 KL 退化成"只盯着老师最大的那一项"，暗知识传得少；$T$ 太大则分布被拉得太平，什么都学不到，常用 2～4。去掉 $T^2$ 之后，蒸馏项的梯度会小 $T^2$ 倍，相当于偷偷把它的权重调小了，和交叉熵混合时比例就不对了。

**4. 把导出的模型跑起来。** 装上 `transformers`，用 `AutoModelForCausalLM.from_pretrained("minisanguo")` 加载导出的目录，用 `apply_chat_template` 拼提示再生成，看看和我们自己的 `answer()` 结果是否一致。

??? success "参考思路"
    一致才说明导出是对的。不一致时按顺序查：**权重名**有没有对上（`load_state_dict` 的 `missing_keys` / `unexpected_keys`）、**config** 里的 `rms_norm_eps` 和 `rope_theta` 是否和训练时相同、**chat_template** 渲染出来的文本和训练时拼的那一串是否逐字符相同。最后一条最容易出问题，也最容易被忽略。

## 小结

- [x] **LoRA**：冻住基座，只训练低秩的 $BA$；$B$ 初始化为零保证挂上那一刻行为不变；合并回权重后推理零开销。它改**行为**很便宜，改**知识**很难。
- [x] **DPO**：用偏好对直接优化策略，损失里要减去参考模型本来就有的差距，否则模型会靠压低所有概率作弊；$\beta$ 控制允许偏离多远；学习率要比 SFT 再小一个量级。
- [x] 偏好数据的真实做法是"模型自采样 + 外部打分"；我们用原文当金标准，流程是一样的。
- [x] **白盒蒸馏**：用加了温度的 KL 拟合老师的整个分布，乘 $T^2$ 让梯度尺度和交叉熵可比；学生更接近老师，但不会超过老师。黑盒蒸馏就是"用大模型合成数据再做 SFT"。
- [x] **导出**：结构对齐 LLaMA / Qwen 之后，改权重名 + 写 `config.json` 和 `chat_template`，transformers、vLLM、SGLang 就能直接加载；chat_template 必须和训练时一模一样。
