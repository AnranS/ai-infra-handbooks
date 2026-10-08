# From scratch (2): the model and the training loop

<p class="lead">The previous chapter turned <em>Romance of the Three Kingdoms</em> into 450,000 tokens. This chapter writes a small GPT and then a complete training loop: fetching data, the learning-rate schedule, gradient clipping, validation, saving checkpoints and generating text. Two and a half minutes of training on a CPU is enough for the model to write sentences in the book's style. Every line of code corresponds to something real in large-model training, and at the end we verify something often overlooked: whether resuming after an interruption gives exactly the same result as training straight through.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What should the first step's loss be for a randomly initialised language model? Why is that a useful check?
    2. Why warm up? Why is AdamW's $\beta_2$ often 0.95 rather than the default 0.999?
    3. Which parameters get no weight decay?
    4. The training loss is still falling but the validation loss has stopped. What does that mean? What do you do?
    5. For a resumed run to match an uninterrupted one exactly, what has to be in the checkpoint?

??? success "Answers for the self-test (answer first, then open this)"
    1. About $\ln V$ (V being the vocabulary size): a randomly initialised model gives a nearly uniform distribution over each token. A first-step loss far from that means something is wrong with the initialisation, the loss computation or the data.
    2. At the start of training, Adam's second-moment estimate is not yet accurate and the parameters are random, so a large learning rate easily diverges; warming up raises it gradually from small. $\beta_2 = 0.95$ lets the second moment follow changes in the gradient faster, which reduces loss spikes in large-model training.
    3. The normalisation layers' weights (the gains) and biases, and usually the embedding too; only the matrix-multiply weights get weight decay.
    4. The beginning of overfitting: the model is memorising the training set. The remedies: more data, a smaller model, stronger regularisation (dropout, weight decay), early stopping (taking the checkpoint with the lowest validation loss).
    5. The model parameters, the optimizer states (Adam's two moments and the step count), the learning-rate schedule's step, the data reader's position, and the random number generators' state.

## The model {#模型}

The same family of architecture as LLaMA and Qwen (where each part came from is in [Assembling a large model from scratch](llm://transformer/build-llm/) in the large-model handbook), only much smaller:

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


def apply_rope(x, cos, sin):                                  # x: [B, H, T, D], the two halves paired and rotated
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
        hidden = 4 * cfg.d_model * 2 // 3 // 32 * 32 or 32     # SwiGLU: three matrices with an intermediate dimension of about 8/3 d, the same parameter count as a two-layer MLP at 4d
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
        x = x + self.attn(self.norm1(x), cos, sin)            # pre-normalisation plus the residual
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight                  # the input and output share an embedding: in a small model the vocabulary takes most of the parameters
        self.apply(self._init)
        for name, p in self.named_parameters():               # the projections that write back into the residual stream are scaled down by the layer count, so even a deep network does not diverge at the start
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
        return sum(p.numel() for p in self.parameters())      # the shared embedding is counted once

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

A few details that matter for training:

- **A shared embedding**: the output layer uses the input embedding matrix directly. With a vocabulary of 8192 and 128 dimensions, that one matrix is 1.05M parameters, and sharing takes the model from 2.9M down to 1.8M.
- **Initialisation**: the weights are normal with a standard deviation of 0.02, except the two projections that write back into the residual stream (attention's `proj` and the MLP's `down`), which are divided by $\sqrt{2L}$. Each layer adds to the residual stream twice, so the more layers there are, the smaller the initial additions have to be, or the residual stream's variance grows with depth and a deep model is unstable from the start.
- **Pre-normalisation**: normalise before attention and the MLP, leaving the residual stream itself unnormalised, so the gradients flow straight back to the earlier layers.
- The loss is computed in fp32 (`logits.float()`): a softmax over the vocabulary sums in a way that is sensitive to precision, and mixed-precision training does the same.

## The training loop {#训练循环}

```python title="train.py" ci="loose"
import math
from dataclasses import asdict, dataclass

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    batch_size: int = 16          # 16 sequences per step, 128 tokens each
    max_steps: int = 600
    lr: float = 3e-3              # the peak learning rate: a small model can take a large one
    min_lr: float = 3e-4          # a cosine decay to a tenth of the peak
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


val_batches = [get_batch(val_data, 32, torch.Generator().manual_seed(123)) for _ in range(8)]   # a fixed set of validation batches


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
decay = [p for p in model.parameters() if p.dim() >= 2]       # the matrices and the embedding get weight decay
no_decay = [p for p in model.parameters() if p.dim() < 2]     # the normalisation layers' gains do not
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
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)   # returns the global norm before clipping
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

```text title="output"
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

Line by line through this loop:

- **Fetching data**: each step takes 16 random segments of 129 tokens from the training set, the first 128 being the input and the same shifted by one the target, so one piece of text provides 128 next-token-prediction samples at once.
- **AdamW's two parameter groups**: the matrices and the embedding get a weight decay of 0.1 and the RMSNorm gains do not (they control the scale, and decay would pull them toward 0); $\beta_2 = 0.95$ is a shorter memory than the default 0.999, so the second moment follows changes in the gradient faster. Nearly all large-model training sets it this way, and it recovers faster from a loss spike.
- **The learning rate**: a linear warm-up over the first 60 steps (Adam's early second-moment estimate is inaccurate and a large learning rate from the start easily goes astray), then a cosine decay to a tenth of the peak. The learning rate is the most important hyperparameter, and the next chapter's exercises sweep it.
- **Gradient clipping**: scale down whenever the global norm exceeds 1.0. What is printed here is the norm before clipping, steady at about 0.5, which says the training is smooth; real training monitors it, and a sudden rise is often the precursor to a loss spike (see [Training stability](../algo/stability.md)).
- **Validation**: a fixed 8 validation batches evaluated every 100 steps, so that the numbers are comparable over time.

Reading the output:

- **The initial loss is 9.055, close to $\ln 8192 = 9.011$**: a randomly initialised model's prediction for each token is nearly uniform. A starting loss far above $\ln V$ means something is wrong with the initialisation or the loss computation, which is the cheapest check there is before training.
- **The gap between the training and validation losses is widening**: at step 200 they are about the same, and at step 600 the training loss is 5.40 against 6.03. The model has been over the training set 3 times and is starting to memorise: it gets more and more familiar with the training set's particular sentences while improving ever more slowly on the unseen last 12 chapters. With only 400,000 tokens this is inevitable; real pretraining usually sees each token once and barely meets this problem. The remedies are more data, a smaller model, fewer steps, or regularisation.
- **The generated text** already has the book's character: the names, the dialogue format, the stock phrases for leading troops and withdrawing. Only the events do not follow from each other, which is all that 1.8M parameters and two and a half minutes can learn.

## Resuming: what has to be in the checkpoint {#续训checkpoint-里必须存什么}

Large-scale training will be interrupted: machine failures, preemption, configuration changes. The requirement when resuming is that **continuing from the checkpoint matches an uninterrupted run**. Verified on a smaller model, together with what happens when something is left out:

```python title="resume.py"
import torch

from model import GPT, GPTConfig

data = torch.load("tokens.pt")["train"].long()
cfg = GPTConfig(d_model=64, n_head=2, n_layer=2, seq_len=64)   # a smaller model, which finishes in a few seconds


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


model, opt = make(seed=0)                                      # the control: 40 steps straight through
train(model, opt, torch.Generator().manual_seed(1), 40)
reference = {k: v.clone() for k, v in model.state_dict().items()}

model, opt = make(seed=0)                                      # interrupt after 20 steps and save a checkpoint
gen = torch.Generator().manual_seed(1)
train(model, opt, gen, 20)
torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "data_rng": gen.get_state()}, "mid.pt")


def resume(restore_optimizer, restore_data_rng):
    model, opt = make(seed=123)                                # a different random initialisation: everything has to come from the checkpoint
    ckpt = torch.load("mid.pt")
    model.load_state_dict(ckpt["model"])
    if restore_optimizer:
        opt.load_state_dict(ckpt["optimizer"])                 # Adam's first and second moments and its step count
    gen = torch.Generator().manual_seed(999)
    if restore_data_rng:
        gen.set_state(ckpt["data_rng"])                        # where the data had been read to
    train(model, opt, gen, 20)
    return max((a - reference[k]).abs().max().item() for k, a in model.state_dict().items())


for name, o, d in [("恢复模型、优化器状态和数据位置", True, True), ("没有恢复数据位置", True, False),
                   ("没有恢复优化器状态", False, True)]:
    print(f"{name}：续训 20 步后，与一口气训练 40 步的参数最大差 {resume(o, d):.1e}")
```

```text title="output"
恢复模型、优化器状态和数据位置：续训 20 步后，与一口气训练 40 步的参数最大差 0.0e+00
没有恢复数据位置：续训 20 步后，与一口气训练 40 步的参数最大差 7.5e-02
没有恢复优化器状态：续训 20 步后，与一口气训练 40 步的参数最大差 7.2e-02
```

- **The model parameters**, obviously.
- **The optimizer states**: Adam's first and second moments and its step count (needed for the bias correction). Restoring only the parameters and starting the optimizer from zero amounts to a cold start before the warm-up all over again, and the training trajectory is completely different; in large-model resumption this often shows up as a loss spike.
- **Where the data had been read to**: the random number generator's state, or the data loader's position. Without it, the resumed run rereads some data and misses other data.
- There is also **the learning-rate schedule's step** (this example uses a constant rate), mixed precision's loss-scaling factor, and each rank's data position under multi-GPU training. A large model's checkpoint is several terabytes and takes minutes to write, so real systems save asynchronously with distributed checkpoints (each rank storing its own shard, see [Training frameworks](../practice/frameworks-rl.md)) and test periodically that recovery actually works.

## Generating: temperature and top-k {#生成温度与-top-k}

Dial what temperature, top-k and top-p each do to the same distribution (the same tool as in the large-model handbook):

<div class="aig-widget" data-widget="softmax"></div>

Load the trained checkpoint and look at the model's prediction after an opening quotation, and at the effect of different sampling parameters:

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

```text title="output"
下一个 token 最可能是： 吾 3.9%  汝 3.3%  我 2.7%  今 1.9%  丞相 1.9%
温度 0.3：今丞相已兵，可自杀之。」 孔明从之，遂令人入。玄德怒曰：「吾若来，如何？」玄德曰：「吾且来，吾欲报我
温度 1.0：汝二人自此书兵栖，初更之首，多看倘之众，难行之；若拒断，亲自至四门。今日各处远借曹操妻小，则军师可杀众。」术
温度 1.0，top-k 20：今丞相已归，则不杀之者，则其当灭之：乃当归大事，非反也。今日今虽死，则可归荆州，必图我，何
温度 1.5：汝二人自此书兵栖队初更坡者之命，倘之众唤入难冻中箭。昨夜驰塞度不动四门，必然各处俄借安疾，则军师可为名，奈何疲术
```

- The model thinks the likeliest next characters are the pronouns and openers that this book's speeches commonly begin with, but each has a probability of only a few percent and the distribution is very flat.
- **Temperature** divides the logits by $T$ before the softmax: a small $T$ sharpens the distribution for more conservative and monotonous output (at 0.3 several sentences in a row start the same way); a large $T$ flattens it, and at 1.5 it is already babbling.
- **top-k** samples only among the $k$ most probable tokens, cutting off the low-probability tail and avoiding absurd characters while keeping some variety. These are exactly the sampling parameters in an inference engine (see [Sampling and the API](serving://engine/sampler-api/) in the inference-systems handbook).

!!! interview "How to explain it"
    To explain how you would train a language model from scratch and what to watch for: **the data** (cleaning, deduplication, the tokenizer, splitting the validation set by document), **the model** (pre-normalisation, RoPE, SwiGLU, a shared embedding; the residual projections scaled down by depth at initialisation), **the optimisation** (AdamW with $\beta_2 = 0.95$, weight decay on the matrices but not the normalisation layers; warm-up plus cosine or warmup-stable-decay; gradient clipping), **the monitoring** (the initial loss should be close to $\ln V$; the gap between training and validation loss, the gradient norm; reading generated samples regularly), and **the engineering** (a checkpoint holding the model, the optimizer, the scheduler and the data position so that resumption is exact; get the whole procedure working on a small model before scaling up).

## Exercises {#练习}

**1. What happens without warm-up?** Change `warmup` to 1, retrain, and compare the loss and the gradient norm over the first 200 steps.

??? success "The approach"
    Adam's first update amounts to $\eta \cdot \mathrm{sign}(g)$: every parameter takes a full-learning-rate step regardless of how small its gradient is. A randomly initialised network is most sensitive to exactly this, every direction taking a large step at once, so the loss shoots up or oscillates high for a while and the gradient norm spikes; a small model usually recovers and a large one may simply diverge. Warm-up lets Adam accumulate a few dozen steps of second-moment estimate before the steps get large.

**2. A larger learning rate.** Change `lr` from 3e-3 to 1e-2 and 3e-2. What happens to the training? How does the best learning rate relate to the model's size?

??? success "The approach"
    With too large a learning rate the loss falls faster but more erratically, the gradient norm often exceeds the clipping threshold, and the final validation loss is worse; larger still and it diverges (the loss becomes nan or stalls high). The general rule is that a larger model (especially a wider one) wants a smaller learning rate, which is the problem parameterisations like muP exist to solve: letting a learning rate tuned on a small model transfer directly to a large one. The practical approach is to sweep the learning rate on a small model and extrapolate by the empirical rule.

**3. What to do about overfitting?** The training set is only 400,000 tokens. List at least three ways to reduce overfitting and say whether each is common in large-model pretraining.

??? success "Answer"
    - **More data**: the most effective, and the main approach in large-model pretraining (each token is generally seen once).
    - **Fewer steps / early stopping**: stop at the lowest validation loss, common on a small dataset.
    - **A smaller model**: fewer parameters make the training set harder to memorise.
    - **Regularisation**: dropout, a larger weight decay. Large-model pretraining barely uses dropout (with enough data there is no overfitting to begin with and dropout only wastes compute); it is used when fine-tuning on a small dataset.
    - **Data augmentation**: not very common for text, but you can vary the split positions and mix in data from a nearby domain.

## Summary {#小结}

- [x] A small GPT: pre-RMSNorm, RoPE, SwiGLU, a shared embedding; the projections that write back into the residual stream scaled by $1/\sqrt{2L}$ at initialisation.
- [x] The training loop: sample random segments, forward to the loss, backward, clip the gradients, AdamW ($\beta_2 = 0.95$, no decay on the normalisation layers), warm-up plus a cosine learning rate; evaluate on fixed validation batches at intervals.
- [x] The initial loss should be close to $\ln V$; a widening gap between training and validation loss means overfitting, which is inevitable with little data.
- [x] Resuming has to restore the model, the optimizer states, the learning-rate schedule's step and the data position to match an uninterrupted run exactly.
- [x] Sampling: temperature controls how sharp the distribution is, and top-k cuts off the low-probability tail.
