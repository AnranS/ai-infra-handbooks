# 从零训练（二）：模型与训练循环

<p class="lead">上一章把《三国演义》变成了 45 万个 token。这一章写一个小 GPT，再写一个完整的训练循环：取数据、学习率调度、梯度裁剪、验证、存 checkpoint、生成文字。在 CPU 上训练两分半钟，模型就能写出"却说曹操引军投徐州"这样的句子。每一行代码都对应着大模型训练里的一个真实环节，最后还要验证一件常被忽略的事：中断之后续训，结果和一口气训完是否完全一样。</p>

!!! note "这一章跑什么"
    **主线**：`model.py`（模型定义，后面每一章都 import 它）、`train.py`（约 2.5 分钟）→ 产出 `ckpt.pt`。
    **实验**：`sizes.py` 把几个尺寸的账算出来、`resume.py` 验证"中断续训和不中断完全一致"、`sampling.py` 比较采样参数。

!!! question "自测：能答上来就可以跳过本章"
    1. 一个随机初始化的语言模型，第一步的 loss 应该是多少？为什么这是一个有用的检查？
    2. 为什么要 warmup？为什么 AdamW 的 $\beta_2$ 常取 0.95 而不是默认的 0.999？
    3. 哪些参数不做权重衰减？
    4. 训练 loss 还在降、验证 loss 不怎么降了，说明什么？该怎么办？
    5. 要想中断后续训的结果和不中断完全一样，checkpoint 里必须存哪些东西？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 约 $\ln V$（V 是词表大小）：随机初始化的模型对每个 token 给出接近均匀的分布。如果第一步的 loss 明显偏离它，说明初始化、损失的计算或数据有问题。
    2. 训练刚开始时 Adam 的二阶矩估计还不准，参数也是随机的，大学习率容易让训练发散，warmup 让学习率从小慢慢升上去。$\beta_2 = 0.95$ 让二阶矩更快地跟上梯度的变化，大模型训练里能减少 loss 突刺。
    3. 归一化层的权重（缩放）和偏置，通常还有嵌入层；只对矩阵乘的权重做权重衰减。
    4. 过拟合的开始：模型在记训练集。办法：更多的数据、更小的模型、更强的正则（dropout、权重衰减）、提前停止（选验证 loss 最低的 checkpoint）。
    5. 模型参数、优化器状态（Adam 的两个矩和步数）、学习率调度的步数、数据读取的位置，以及随机数生成器的状态。

## 模型

和 LLaMA、Qwen 同一类结构（每个部件的来龙去脉见大模型手册的[从零组装一个大模型](llm://transformer/build-llm/)），只是小得多：

```python title="model.py"
"""一个小 GPT：RMSNorm、RoPE、SwiGLU、共享词嵌入——和 LLaMA / Qwen 同一类结构，只是小得多"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    seq_len: int = 128
    rope_theta: float = 10000.0


def rope_cos_sin(T, head_dim, theta, device=None):
    inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    freqs = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return freqs.cos(), freqs.sin()                          # [T, head_dim / 2]


def apply_rope(x, cos, sin):                                  # x: [B, H, T, D]，前后两半配对旋转
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = 4 * cfg.d_model * 2 // 3 // 32 * 32 or 32     # SwiGLU：三个矩阵，中间维取约 8/3·d，参数量和 4d 的两层 MLP 相当
        self.gate_up = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.mlp = Attention(cfg), MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)            # 前置归一化 + 残差
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                  # 输入输出共享词嵌入：小模型里词表占了大部分参数
        self.apply(self._init)
        for name, p in self.named_parameters():               # 写回残差流的投影按层数缩小，深层网络一开始也不会发散
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        cos, sin = rope_cos_sin(idx.shape[1], self.cfg.d_model // self.cfg.n_head, self.cfg.rope_theta, idx.device)
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

    def num_params(self):
        return sum(p.numel() for p in self.parameters())      # 共享的词嵌入只算一次

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.seq_len:])[:, -1] / temperature
            if top_k is not None:
                kth = logits.topk(top_k).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
```

几个训练相关的细节：

- **共享词嵌入**：输出层直接用输入的词嵌入矩阵。词表 8192、维度 128 时，这一个矩阵就有 1.05M 参数，共享之后模型从 2.9M 降到 1.8M；
- **初始化**：权重取标准差 0.02 的正态分布，而写回残差流的两个投影（注意力的 `proj`、MLP 的 `down`）再除以 $\sqrt{2L}$。每一层往残差流上加两次，层数越多，初始时加上去的量就越要小，否则残差流的方差随层数增长，深的模型一开始就不稳定；
- **前置归一化**：先归一化再进注意力和 MLP，残差流本身不经过归一化，梯度能直接流回前面的层；
- 损失用 fp32 计算（`logits.float()`）：词表上的 softmax 求和对精度敏感，混合精度训练时也要这样做。

## 这些尺寸是怎么定的

`GPTConfig` 里的几个数字不是随手填的。先把账算清楚：

```python title="sizes.py"
"""几个尺寸怎么选：词表、深浅、注意力头的 KV——先把账算出来，再决定"""
from model import GPT, GPTConfig


def breakdown(cfg):
    m = GPT(cfg)
    embed = cfg.vocab_size * cfg.d_model                      # 输入输出共享，只算一次
    return m.num_params(), embed


print("词表大小对一个 d=128 的小模型意味着什么（输入输出共享词嵌入）：")
print("  词表    总参数   词嵌入占比")
for v in (4096, 8192, 16384, 32768):
    total, embed = breakdown(GPTConfig(vocab_size=v))
    print(f"{v:6d}   {total / 1e6:5.2f}M   {embed / total:8.0%}")

print("\n同样约 1.3～1.5M 的非词嵌入参数，摊在宽度上还是深度上：")
print("   d  层数   非词嵌入   总参数   每层参数")
for d, n_layer in ((320, 1), (192, 3), (128, 8), (96, 13)):
    total, embed = breakdown(GPTConfig(d_model=d, n_layer=n_layer, n_head=max(1, d // 32)))
    print(f"{d:4d}  {n_layer:3d}    {(total - embed) / 1e6:5.2f}M   {total / 1e6:5.2f}M   {(total - embed) / n_layer / 1e3:7.0f}K")

print("\n推理时每个 token 的 KV cache（d=768、28 层、fp16，按一个 token 算）：")
print("  查询头  KV 头   每 token 的 KV   相对 MHA")
d_model, n_layer, n_head = 768, 28, 12
for kv in (12, 4, 2, 1):
    per_token = 2 * n_layer * kv * (d_model // n_head) * 2    # K 和 V，各 2 字节
    print(f"{n_head:6d}  {kv:5d}   {per_token / 1024:11.1f} KB   {kv / n_head:8.0%}")
```

```text title="输出"
词表大小对一个 d=128 的小模型意味着什么（输入输出共享词嵌入）：
  词表    总参数   词嵌入占比
  4096    1.28M        41%
  8192    1.80M        58%
 16384    2.85M        74%
 32768    4.95M        85%

同样约 1.3～1.5M 的非词嵌入参数，摊在宽度上还是深度上：
   d  层数   非词嵌入   总参数   每层参数
 320    1     1.21M    3.83M      1209K
 192    3     1.33M    2.90M       443K
 128    8     1.51M    2.56M       189K
  96   13     1.44M    2.23M       111K

推理时每个 token 的 KV cache（d=768、28 层、fp16，按一个 token 算）：
  查询头  KV 头   每 token 的 KV   相对 MHA
    12     12          84.0 KB       100%
    12      4          28.0 KB        33%
    12      2          14.0 KB        17%
    12      1           7.0 KB         8%
```

- **词表**：小模型里词嵌入是最大的一块。词表翻倍，序列大约短 10%～15%，但词嵌入参数直接翻倍——8192 的词表已经占了我们这个模型的 58%。所以从零训练的小模型几乎都用很小的词表（几千到一万），而几十亿参数的模型用十万以上（Qwen2 是 151k、Llama 3 是 128k），那时词嵌入只占百分之几；
- **深还是宽**：同样的非词嵌入参数，宽而浅的模型每层很大、层数很少；深而窄的反过来。小模型上的系统实验（MobileLLM 那一组）发现，参数量固定时**深度比宽度更重要**——层数多的模型更容易学到抽象的表示。但"窄"有下限：`d_model` 太小时，注意力头的维度（`d_model / n_head`）会小到装不下信息，再加层也补不回来。我们取 `d=128、4 层、头维度 32`，是在"能快速训完"和"不至于太窄"之间折中；再大一点的开源小模型（例如 64M 级别的 MiniMind）用的是 `d=768、8 层`；
- **KV 头**：注意力里让多个查询头共用一组 K/V，就是 GQA。参数省得不多，**省的是推理时的 KV cache**：12 个查询头配 4 个 KV 头，KV cache 直接降到三分之一。我们的模型上下文只有 128、也不做服务，所以用最简单的 MHA（每个头一组 K/V）；真实的模型几乎都用 GQA，原理见大模型手册的[注意力机制](llm://transformer/attention/)；
- **`rope_theta`**：RoPE 的频率基数。短上下文用 10000 就够；要训练或外推到几万 token，基数要调大（现在常见 1e6），否则远距离的位置区分不开（见[位置编码与 RoPE](llm://transformer/position/)）；
- **跟着现成的生态走**：结构和命名尽量和 LLaMA / Qwen 对齐（pre-norm + RMSNorm、SwiGLU、RoPE、不带 bias 的线性层），好处是训好之后改个权重名就能被 transformers、vLLM、SGLang 直接加载——第五章会真的导出一次。

## 训练循环

```python title="train.py" ci="loose"
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
```

```text title="输出"
模型 1.80M 参数；每步 2048 个 token，共 600 步 = 3.0 遍训练集
初始验证 loss 9.055（均匀猜测是 ln 8192 = 9.011）
  步   学习率   训练 loss   验证 loss   梯度范数
 100  2.97e-03      7.169      6.675      0.638
 200  2.58e-03      6.209      6.397      0.484
 300  1.89e-03      5.893      6.238      0.507
 400  1.12e-03      5.646      6.139      0.525
 500  5.26e-04      5.484      6.064      0.544
 600  3.00e-04      5.396      6.029      0.547
【却说曹操】引军投徐州守城外。 赵云与赵云入，甘宁，孔明往徐州。操曰：「如他在长安，彼不得相争，却不可相保，故且自使为蜀，又知曹操之计。」玄德曰：「丞相有何也！」遂下令，
【玄德曰：】「今云长也！」 二人大喜，叱众将曰：「今二人已来，可起兵。」二人恐黄忠，徐晃见赵云。赵云赶来。曹操在山坡而走，却却引兵到城下。马岱分付曰：「主公已退，吾把军皆退。」
```

一行一行看这个循环：

- **取数据**：每步从训练集里随机取 16 段、每段 129 个 token，前 128 个是输入，错开一位是目标——一段文字同时提供了 128 个"预测下一个 token"的样本；
- **AdamW 的两组参数**：矩阵和词嵌入做 0.1 的权重衰减，RMSNorm 的增益不做（它们控制尺度，衰减会把它们往 0 拉）；$\beta_2 = 0.95$ 比默认的 0.999 更短的"记忆"，二阶矩能更快跟上梯度的变化，大模型训练里几乎都这样设，出现 loss 突刺时恢复得更快；
- **学习率**：前 60 步线性 warmup（Adam 早期的二阶矩估计不准，一开始就用大学习率容易走偏），之后余弦衰减到峰值的 1/10。学习率是最重要的超参数，下一章的练习会扫一遍；
- **梯度裁剪**：全局范数超过 1.0 就按比例缩小。这里打印的是裁剪前的范数，稳定在 0.5 左右，说明训练很平稳；真实训练里要监控它，突然变大往往是 loss 突刺的前兆（见[训练稳定性](train://algo/stability/)）；
- **验证**：固定的 8 个验证 batch，每 100 步算一次，数字才能前后比较。

读输出：

- **初始 loss 是 9.055，接近 $\ln 8192 = 9.011$**：随机初始化的模型对每个 token 的预测接近均匀分布。如果一开始的 loss 远大于 $\ln V$，说明初始化或者损失的计算有问题——这是训练开始前最便宜的检查；
- **训练 loss 与验证 loss 的差距在拉大**：第 200 步时两者还差不多，第 600 步时训练 5.40、验证 6.03。模型已经把训练集看了 3 遍，开始"背书"了：越来越熟悉训练集的具体句子，对没见过的后 12 回提升却越来越慢。数据只有 40 万个 token 时，这是必然的——真实的预训练通常每个 token 只看一遍，几乎不会遇到这个问题。办法是更多的数据、更小的模型、更少的步数，或者加正则；
- **生成的文字**已经有了这本书的样子：人名、"曰：「……」"的对话格式、"引兵""退军"这样的套话，只是情节前后不连贯——1.8M 参数、训练两分半钟，只能学到这些。

## 续训：checkpoint 里必须存什么

大规模训练一定会中断：机器故障、抢占、调整配置。续训的要求是：**从 checkpoint 接着训，和没有中断过的结果一致**。用一个更小的模型验证，并看看少存了什么会怎样：

```python title="resume.py"
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
```

```text title="输出"
恢复模型、优化器状态和数据位置：续训 20 步后，与一口气训练 40 步的参数最大差 0.0e+00
没有恢复数据位置：续训 20 步后，与一口气训练 40 步的参数最大差 7.5e-02
没有恢复优化器状态：续训 20 步后，与一口气训练 40 步的参数最大差 7.2e-02
```

- **模型参数**当然要存；
- **优化器状态**：Adam 的一阶矩、二阶矩和步数（偏差校正要用）。只恢复参数、优化器从零开始，等于重新做了一次 warmup 前的冷启动，训练轨迹完全不同；大模型续训时这常常表现为一个 loss 突刺；
- **数据读到了哪里**：随机数生成器的状态，或者数据加载器的位置。不恢复的话，续训会重复读一部分数据、漏掉另一部分；
- 此外还有**学习率调度的步数**（本例用的是常数学习率）、混合精度的损失缩放因子，以及多卡训练时每个 rank 的数据位置。大模型的 checkpoint 有几 TB，存一次要几分钟，所以真实系统用异步保存、分布式 checkpoint（每个 rank 存自己的分片，见[训练框架](train://practice/frameworks-rl/)），并定期测试"能不能真的恢复"。

## 生成：温度与 top-k

温度、top-k、top-p 对同一个分布各做了什么，拨一拨（大模型手册里的同一个工具）：

<div class="aig-widget" data-widget="softmax"></div>

加载训练好的 checkpoint，看模型在"孔明曰：「"之后的预测，以及不同采样参数的效果：

```python title="sampling.py" ci="loose"
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

ckpt = torch.load("ckpt.pt")
model = GPT(GPTConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
prompt = torch.tensor([tok.encode("孔明曰：「").ids])

with torch.no_grad():
    probs = model(prompt)[0, -1].softmax(-1)
top = probs.topk(5)
print("下一个 token 最可能是：", "  ".join(f"{tok.decode([int(i)])} {p:.1%}" for p, i in zip(top.values, top.indices)))
for temperature, top_k in [(0.3, None), (1.0, None), (1.0, 20), (1.5, None)]:
    gen = torch.Generator().manual_seed(7)
    out = model.generate(prompt, 40, temperature=temperature, top_k=top_k, generator=gen)
    text = tok.decode(out[0, prompt.shape[1]:].tolist()).replace("\n", " ")
    print(f"温度 {temperature}{'，top-k ' + str(top_k) if top_k else ''}：{text}")
```

```text title="输出"
下一个 token 最可能是： 吾 3.9%  汝 3.3%  我 2.7%  今 1.9%  丞相 1.9%
温度 0.3：今丞相已兵，可自杀之。」 孔明从之，遂令人入。玄德怒曰：「吾若来，如何？」玄德曰：「吾且来，吾欲报我
温度 1.0：汝二人自此书兵栖，初更之首，多看倘之众，难行之；若拒断，亲自至四门。今日各处远借曹操妻小，则军师可杀众。」术
温度 1.0，top-k 20：今丞相已归，则不杀之者，则其当灭之：乃当归大事，非反也。今日今虽死，则可归荆州，必图我，何
温度 1.5：汝二人自此书兵栖队初更坡者之命，倘之众唤入难冻中箭。昨夜驰塞度不动四门，必然各处俄借安疾，则军师可为名，奈何疲术
```

- 模型认为下一个字最可能是"吾""汝""我""今""丞相"——都是这本书里孔明说话常见的开头，但每个的概率都只有百分之几，分布很平；
- **温度**把 logits 除以 $T$ 再做 softmax：$T$ 小，分布更尖，输出更保守、句式单调（0.3 时连着几句都是"吾……"）；$T$ 大，分布更平，1.5 时已经开始胡言乱语；
- **top-k** 只在概率最高的 $k$ 个 token 里采样，截掉长尾的低概率 token，在保留多样性的同时避免离谱的字。推理引擎里的采样参数就是这几个（见推理系统手册的[采样与 API](serving://engine/sampler-api/)）。

!!! interview "怎么讲清楚"
    讲"从零训练一个语言模型，你会怎么做、要注意什么"：**数据**（清洗、去重、分词器、按文档切分验证集）→ **模型**（前置归一化、RoPE、SwiGLU、共享词嵌入；初始化时残差投影按层数缩小）→ **优化**（AdamW，$\beta_2 = 0.95$，矩阵做权重衰减、归一化层不做；warmup + 余弦或 WSD；梯度裁剪）→ **监控**（初始 loss 应接近 $\ln V$；训练与验证 loss 的差距、梯度范数；定期看生成样例）→ **工程**（checkpoint 存模型、优化器、调度器和数据位置，能精确续训；先在小模型上把整条流程跑通，再扩大规模）。

## 练习

**1. 去掉 warmup 会怎样？** 把 `warmup` 改成 1，重新训练，比较前 200 步的 loss 和梯度范数。

??? success "参考思路"
    第一步 Adam 的更新相当于 $\eta \cdot \mathrm{sign}(g)$：每个参数都按满学习率走一步，不管它的梯度多小。随机初始化的网络对这种"所有方向同时大步走"最敏感，loss 会先冲高或在高处震荡一阵，梯度范数也会出现尖峰；小模型通常还能恢复，大模型就可能直接发散。warmup 让 Adam 先积累几十步的二阶矩估计，再把步子迈大。

**2. 更大的学习率。** 把 `lr` 从 3e-3 改成 1e-2 和 3e-2，训练会怎样？最好的学习率和模型大小有什么关系？

??? success "参考思路"
    学习率太大时，loss 下降更快但更抖，梯度范数经常超过裁剪阈值，最终的验证 loss 反而更差；再大就会发散（loss 变成 nan 或停在高位）。一般规律是模型越大（尤其是越宽），最优学习率越小，这也是 μP 这类参数化方法要解决的问题：让小模型上调好的学习率能直接迁移到大模型。实际做法是在小模型上扫学习率，再按经验规律外推。

**3. 过拟合怎么办？** 训练集只有 40 万个 token。列出至少三种缓解过拟合的办法，并说明它们在大模型预训练里是否常用。

??? success "参考答案"
    - **更多数据**：最有效，大模型预训练的主要做法（每个 token 一般只看一遍）；
    - **更少的步数 / 早停**：在验证 loss 最低处停下，小数据集上常用；
    - **更小的模型**：参数越少越难"背下"训练集；
    - **正则**：dropout、更大的权重衰减。大模型预训练几乎不用 dropout（数据量足够大，本来就不会过拟合，dropout 只会浪费算力），微调小数据集时才会用；
    - **数据增强**：对文本不太常用，但可以用不同的切分位置、混入相近领域的数据。

## 小结

- [x] 小 GPT：前置 RMSNorm、RoPE、SwiGLU、共享词嵌入；初始化时写回残差流的投影按 $1/\sqrt{2L}$ 缩小。
- [x] 训练循环：随机取片段 → 前向求 loss → 反向 → 梯度裁剪 → AdamW（$\beta_2 = 0.95$，归一化层不衰减）→ warmup + 余弦学习率；定期在固定的验证 batch 上评估。
- [x] 初始 loss 应接近 $\ln V$；训练与验证 loss 的差距拉大意味着过拟合，数据少时必然出现。
- [x] 续训必须恢复模型、优化器状态、学习率调度的步数和数据位置，才能和不中断完全一致。
- [x] 采样：温度控制分布的尖锐程度，top-k 截掉低概率的长尾。
