# 13. Batching and padding

<p class="lead">Decode computes only one token per step yet reads 3 GB of weights — the compute units are nearly idle. Put several requests into one batch and one forward pass computes everyone's next token; one read of the weights yields B tokens and the throughput goes up. The price is that requests have different lengths, so they need <strong>padding</strong>: shorter ones get placeholder tokens on the left, and a mask marks which positions are fake. The <code>mask</code> argument left behind in chapter 10 is put to use here.</p>

!!! note "What this chapter runs"
    `ch13_batch.py`, about 30 seconds.

## The padding has to be hidden in three places {#填充位要瞒住三个地方}

After left padding, each row of the batch starts with some fake tokens. They must not affect the computation of the real tokens, which needs handling in three places (already written into chapter 10's code):

1. **Full attention**: the keys at fake positions may not be seen by anyone (`scores.masked_fill(~mask, smallest value)`). Note the use of `finfo.min` rather than `-inf`: when a fake position's own row is fully masked, `-inf` makes softmax produce NaN, which then pollutes everything through the residual stream.
2. **Linear attention**: the input at fake positions is zeroed (`x * mask`), so they neither enter the convolution window nor write into the memory table — the write is $\beta k (v - \cdots)$, which is zero when $k = v = 0$.
3. **Position numbering**: every row counts its own positions (`cumsum(mask) - 1`), and padding gets no number, otherwise RoPE's angles are wrong.

```python title="qwen/batch.py"
"""批处理：把几个长短不一的请求拼成一批，一次前向算所有人的下一个 token。

拼批要靠填充（padding）：短的请求左边补占位 token，mask 标出哪些位置是假的。
全注意力里假位置的 key 不许被看；线性注意力里假位置的输入清零，不写进记忆表；位置编号也要跳过假位置。
"""
import torch

from qwen.cache import State
from qwen.sample import sample


def pad_left(seqs, pad_id=0):
    """[[...], [...]] 长短不一的 token 列表 -> ids [B, T]（左边补 pad_id）和 mask [B, T]（True 是真 token）"""
    T = max(len(s) for s in seqs)
    ids = torch.tensor([[pad_id] * (T - len(s)) + s for s in seqs])
    mask = torch.tensor([[False] * (T - len(s)) + [True] * len(s) for s in seqs])
    return ids, mask


@torch.no_grad()
def generate_batch(model, seqs, max_new_tokens=100, stop=(), **kw):
    """seqs: 每个请求的提示词 token 列表 -> 每个请求生成的 token 列表（停止符之后的不算）"""
    ids, mask = pad_left(seqs)
    state = State(model, batch=len(seqs))
    logits = model(ids, state, mask)[:, -1]                                   # prefill: the whole batch in one pass
    outs, done = [[] for _ in seqs], [False] * len(seqs)
    for _ in range(max_new_tokens):
        nxt = [sample(logits[b], generated=outs[b], **kw) for b in range(len(seqs))]
        for b, t in enumerate(nxt):
            if not done[b]:
                outs[b].append(t)
                done[b] = t in stop
        if all(done):
            break
        logits = model(torch.tensor(nxt)[:, None], state)[:, -1]                 # decode: one token per row for the whole batch (finished rows ride along)
    return outs
```

In `generate_batch`, requests that finish early keep "riding along": their new tokens are still fed to the model, just no longer recorded. Production engines (continuous batching) move finished requests out and new ones in, and chapter 10's state design already allows that — each row's state is an independent slice.

## Batch == single {#批--单条}

```python title="ch13_batch.py" ci="loose"
"""第 13 章：4 个长短不一的请求拼成一批；和单条逐个生成比：结果一样、吞吐量更高"""
import os
import time

import torch

from qwen.batch import generate_batch, pad_left
from qwen.chat import Chat, build_prompt
from qwen.harness import max_diff
from qwen.model import Model
from qwen.sample import generate
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)
questions = ["用一句话介绍一下 Transformer。", "北京在哪个国家？", "写一句关于秋天的话。", "1+1 等于几？"]
seqs = [chat.tk.encode(build_prompt([{"role": "user", "content": q}])).ids for q in questions]
ids, mask = pad_left(seqs)
print(f"4 个请求的长度 {[len(s) for s in seqs]}，拼成 {tuple(ids.shape)}；mask 的第一行：{mask[0].int().tolist()}")

t = time.time()                                                      # 1. one request at a time
singles = [generate(model, torch.tensor([s]), max_new_tokens=60, stop=chat.stop, temperature=0) for s in seqs]
t_single = time.time() - t
t = time.time()                                                      # 2. as one batch
batched = generate_batch(model, seqs, max_new_tokens=60, stop=chat.stop, temperature=0)
t_batch = time.time() - t
n = sum(map(len, batched))
print(f"单条：{sum(map(len, singles))} 个 token {t_single:.1f} 秒（{sum(map(len, singles)) / t_single:.1f} tok/s）；"
      f"批处理：{n} 个 token {t_batch:.1f} 秒（{n / t_batch:.1f} tok/s）")
for q, a, b in zip(questions, singles, batched):
    print(f"  {'一致' if a == b else '不同'}  {q} -> {chat.tk.decode(b)!r}")

logits_batch = model(ids, mask=mask)                                 # 3. each row's logits in the batch equal the ones computed alone
for b, s in enumerate(seqs):
    print(f"  第 {b} 行（左边填了 {ids.shape[1] - len(s)} 个）：logits 最大差 {max_diff(logits_batch[b, -len(s):], model(torch.tensor([s]))[0]):.1e}")
```

```text title="output"
4 个请求的长度 [17, 17, 18, 19]，拼成 (4, 19)；mask 的第一行：[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
单条：99 个 token 5.6 秒（17.8 tok/s）；批处理：99 个 token 3.0 秒（32.7 tok/s）
  一致  用一句话介绍一下 Transformer。 -> 'Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。'
  一致  北京在哪个国家？ -> '北京是中国的首都，位于**中国**境内。'
  一致  写一句关于秋天的话。 -> '秋风卷起落叶，铺就了一条金黄的铺路，把整个世界都染成了温暖的金色。'
  一致  1+1 等于几？ -> '1+1 等于 **2**。\n\n这是最基础的算术运算，表示两个 1 相加，结果是 2。'
  第 0 行（左边填了 2 个）：logits 最大差 5.6e-05
  第 1 行（左边填了 2 个）：logits 最大差 5.1e-05
  第 2 行（左边填了 1 个）：logits 最大差 4.5e-05
  第 3 行（左边填了 0 个）：logits 最大差 4.9e-05
```

The four answers match single-request generation token for token, and the throughput roughly doubles — on a CPU, reading the weights is not the only bottleneck; on a GPU, batches of dozens or hundreds scale throughput almost linearly until the compute saturates.

!!! interview "How to explain it"
    To explain batching, start with the motivation: decode is memory-bound, and a batch of requests shares one read of the weights. Then the three padding traps (the attention mask, zeroing the recurrent layers' input, position numbering) and implementation details such as `-inf` producing NaN. Finally contrast static batching with continuous batching: the former waits for the whole batch to finish, the latter swaps requests in and out at any step; the schedulers of inference engines (vLLM / SGLang) do the latter.

## Summary {#小结}

- [x] Left padding + a mask; the padding must be hidden from attention's keys, linear attention's input and RoPE's positions.
- [x] Four requests in one batch match single-request generation token for token, at about 2× the throughput (CPU).

**File created in this chapter**: `qwen/batch.py`.
