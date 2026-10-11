# （十三）批处理与填充

<p class="lead">decode 每步只算一个 token，却要把 3 GB 权重读一遍——算力几乎闲着。把几个请求拼成一批，一次前向算所有人的下一个 token，读一遍权重出 B 个 token，吞吐量就上去了。代价是请求长短不一，要<strong>填充</strong>：短的左边补占位 token，用 mask 标出哪些位置是假的。第 10 章留下的 <code>mask</code> 参数这一章派上用场。</p>

!!! note "这一章跑什么"
    `ch13_batch.py`，约 30 秒。

## 填充位要瞒住三个地方

左填充之后，一批里每一行的开头是若干个假 token。它们不能影响真 token 的计算，有三处要处理（第 10 章的代码里已经写好）：

1. **全注意力**：假位置的 key 谁都不许看（`scores.masked_fill(~mask, 最小值)`）。注意用 `finfo.min` 而不是 `-inf`：假位置自己那一行全被挡住时，`-inf` 会让 softmax 算出 NaN，然后顺着残差流污染一切。
2. **线性注意力**：假位置的输入清零（`x * mask`），它们就既不进卷积窗口也不写记忆表——写入量是 $\beta k (v - \cdots)$，$k = v = 0$ 时为零。
3. **位置编号**：每一行各数各的位置（`cumsum(mask) - 1`），填充位不占号，否则 RoPE 的角度就错了。

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
    logits = model(ids, state, mask)[:, -1]                                   # prefill：一次算整批
    outs, done = [[] for _ in seqs], [False] * len(seqs)
    for _ in range(max_new_tokens):
        nxt = [sample(logits[b], generated=outs[b], **kw) for b in range(len(seqs))]
        for b, t in enumerate(nxt):
            if not done[b]:
                outs[b].append(t)
                done[b] = t in stop
        if all(done):
            break
        logits = model(torch.tensor(nxt)[:, None], state)[:, -1]                 # decode：整批各进一个 token（做完的也陪跑）
    return outs
```

`generate_batch` 里先完成的请求继续"陪跑"：它们的新 token 照样喂进模型，只是不再记录。工业界的引擎（连续批处理）会把做完的请求移出、把新请求塞进来，第 10 章的状态设计已经允许这么做——每一行的状态是独立的切片。

## 批 == 单条

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

t = time.time()                                                      # 1. 单条逐个生成
singles = [generate(model, torch.tensor([s]), max_new_tokens=60, stop=chat.stop, temperature=0) for s in seqs]
t_single = time.time() - t
t = time.time()                                                      # 2. 拼成一批
batched = generate_batch(model, seqs, max_new_tokens=60, stop=chat.stop, temperature=0)
t_batch = time.time() - t
n = sum(map(len, batched))
print(f"单条：{sum(map(len, singles))} 个 token {t_single:.1f} 秒（{sum(map(len, singles)) / t_single:.1f} tok/s）；"
      f"批处理：{n} 个 token {t_batch:.1f} 秒（{n / t_batch:.1f} tok/s）")
for q, a, b in zip(questions, singles, batched):
    print(f"  {'一致' if a == b else '不同'}  {q} -> {chat.tk.decode(b)!r}")

logits_batch = model(ids, mask=mask)                                 # 3. 批里每一行的 logits 和单独算的一样
for b, s in enumerate(seqs):
    print(f"  第 {b} 行（左边填了 {ids.shape[1] - len(s)} 个）：logits 最大差 {max_diff(logits_batch[b, -len(s):], model(torch.tensor([s]))[0]):.1e}")
```

```text title="输出"
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

四个请求的回答和单条生成逐 token 一致，吞吐量差不多翻倍——CPU 上权重读取不是唯一瓶颈，GPU 上批到几十、几百，吞吐量几乎线性涨，直到算力饱和。

!!! interview "怎么讲清楚"
    讲批处理先讲动机：decode 是访存瓶颈，一批请求共享一次权重读取。再讲填充的三个坑（注意力 mask、递推层清零、位置编号），以及 `-inf` 会产生 NaN 这种实现细节。最后对比静态批和连续批处理：前者等一批全做完，后者随时换人；推理引擎（vLLM / SGLang）的调度器做的就是后者。

## 小结

- [x] 左填充 + mask；填充位要瞒住注意力的 key、线性注意力的输入、RoPE 的位置。
- [x] 4 个请求一批，结果和单条逐 token 一致，吞吐量约 2 倍（CPU）。

**这一章新建的文件**：`qwen/batch.py`。
