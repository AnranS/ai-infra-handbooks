# 2. The embedding and the output layer

<p class="lead">The model's entrance and exit use the same table: on the way in, take one row per token id to get a 1024-dimensional vector; on the way out, dot the 1024-dimensional vector with every row of the table to get 248320 scores over the vocabulary. This is the simplest chapter of the book, but it replaces the block with the most parameters (254M, a third of the model), and it is the first taste of the "write one part, run one small experiment" rhythm.</p>

!!! note "What this chapter runs"
    `ch02_embedding.py`, about 10 seconds.

@@files@@

## Look up on the way in, dot on the way out {#查表进门点积出门}

```python title="qwen/embedding.py"
"""嵌入层与输出层：进门查表把 token id 变成向量，出门用同一张表把向量变回词表上的分数。"""
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))   # one row per token

    def forward(self, ids):
        return self.weight[ids]                                     # a lookup is taking rows: [B, T] -> [B, T, dim]


class LMHead(nn.Module):
    """tie_word_embeddings=True：输出层直接用嵌入层那张表，不另存一份（省下 254M 参数）"""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight                                        # the same Parameter object

    def forward(self, x):
        return x @ self.weight.T                                    # [B, T, dim] @ [dim, vocab] -> each position's scores over the vocabulary
```

`Embedding.forward` is one advanced indexing operation, `weight[ids]`, taking the shape from `[B, T]` to `[B, T, 1024]`. `LMHead` is one matrix multiplication, projecting 1024 dimensions back to 248320 — on a small model this is one of the most expensive single operators (1024 × 248320 ≈ 254M multiply-adds, while a whole feed-forward layer has 11M parameters).

**What tying the weights means**: output score = hidden state · embedding vector. For the model to give "park" a high score, it has to push the last layer's output towards the embedding of the token "park". So the embedding space is both the input space and the output space, which is also why there must be an RMSNorm before the last layer (chapter 3): it brings the scale in line with the embedding table.

## Run it: who is close to whom in embedding space {#跑起来嵌入空间里谁和谁近}

```python title="ch02_embedding.py"
"""第 2 章：嵌入层和输出层。看看嵌入空间里谁和谁近，再让一个只有进门和出门的"模型"写几个字"""
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from qwen.embedding import Embedding, LMHead
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # tokenizer.json is read directly with the tokenizers library
emb = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # same names as in the file, so the weights load directly
head = LMHead(emb.weight)                                                     # shares the same Parameter

ids = torch.tensor([tk.encode("今天天气不错，我们去公园").ids])
x = emb(ids)
print(f"进门：{tuple(ids.shape)} 个 token id -> {tuple(x.shape)} 的向量；出门：-> {tuple(head(x).shape)} 的分数")
print(f"嵌入表 {emb.weight.shape[0]} × {emb.weight.shape[1]}，{emb.weight.numel() / 1e6:.0f}M 参数，占全模型 {emb.weight.numel() / sum(v.numel() for v in W.values()):.0%}")

E = F.normalize(emb.weight, dim=-1)                                          # 1. who is close to whom in embedding space: the most similar tokens by cosine similarity
for word in ("今天", "公园", "北京", "快乐", "128"):
    i = tk.encode(word).ids[0]
    near = (E @ E[i]).topk(6).indices[1:]
    print(f"  {word!r:8} 最像的：{[tk.decode([j]) for j in near]}")
a, b, c = (E[tk.encode(w).ids[0]] for w in ("今天", "明天", "公园"))          # 2. similarity is a number: high for related words, low for unrelated ones
print(f"  cos(今天, 明天) = {a @ b:.2f}，cos(今天, 公园) = {a @ c:.2f}")

out = ids                                                                    # 3. a "model" with only an entrance and an exit: continue for 5 tokens
for _ in range(5):
    out = torch.cat([out, head(emb(out))[0, -1].argmax().view(1, 1)], dim=1)
print("没有中间 24 层的模型接着写：", repr(tk.decode(out[0, ids.shape[1]:].tolist())), "—— x = E[token] 和谁的点积最大？它自己，于是只会复读")
```

```text title="输出"
进门：(1, 6) 个 token id -> (1, 6, 1024) 的向量；出门：-> (1, 6, 248320) 的分数
嵌入表 248320 × 1024，254M 参数，占全模型 34%
  '今天'     最像的：['今日', '今天的', ' today', 'Today', ' Today']
  '公园'     最像的：['公園', '公园里', ' park', ' parks', ' 공원']
  '北京'     最像的：['北京的', '上海', '北京市', ' Beijing', '在北京']
  '快乐'     最像的：['快乐的', '快樂', '的快乐', '欢乐', '幸福']
  '128'    最像的：['2', '3', '4', '5', '0']
  cos(今天, 明天) = 0.46，cos(今天, 公园) = 0.12
没有中间 24 层的模型接着写： '公园公园公园公园公园' —— x = E[token] 和谁的点积最大？它自己，于是只会复读
```

The embedding table already carries some "meaning": the tokens closest to "今天" (today) are "今日" and today, the closest to "北京" (Beijing) is "上海" (Shanghai), and the neighbours of "128" are all numbers; measured by cosine similarity, today and tomorrow score 0.46, today and park only 0.12. But it knows nothing about context — "公园" (park) is always the same vector. The "model" at the end, with only an entrance and an exit, can only repeat itself: which token's embedding has the largest dot product with $x = E[\text{token}]$? Its own. What the following chapters do is fill in what goes between these two layers, so that the vector of "park" becomes something else because "we go to" came before it.

!!! interview "How to explain it"
    When explaining tied embeddings, cover both directions: it saves parameters (a third of a small model), and it binds the "input space" and the "output space" into one, so a logit is the similarity between the hidden state and each token's embedding. Then add the cost: the big models (Qwen3.5 from 9B up) do not tie, because the embedding table is only a few percent of them and training the two separately is freer.

## Summary {#小结}

- [x] Embedding = take a row by id; output layer = dot with every row; the two share one 248320 × 1024 table.
- [x] Related words sit close together in embedding space (today / tomorrow 0.46, today / park 0.12); a model with only an entrance and an exit can only repeat itself.

**File created in this chapter**: `qwen/embedding.py`.
