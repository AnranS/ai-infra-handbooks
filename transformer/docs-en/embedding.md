# 2. The embedding and the output layer

<p class="lead">The model's entrance and exit use the same table: on the way in, take one row per token id to get a 1024-dimensional vector; on the way out, dot the 1024-dimensional vector with every row of the table to get 248320 scores over the vocabulary. This is the simplest chapter of the book, but it replaces the block with the most parameters (254M, a third of the model), and it is the first taste of the "replace one part, check once" routine.</p>

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

## Check against the HF modules of the same name {#和-hf-的同名模块对答案}

```python title="ch02_embedding.py"
"""第 2 章：嵌入层和输出层，和 HF 的同名模块对答案；一个只有进门和出门的"模型"能写出什么"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen.embedding import Embedding, LMHead
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # the largest element-wise difference between two tensors
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # the reference: only its modules of the same name are used for checking
tok = AutoTokenizer.from_pretrained(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]

emb = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # same names as in the file, so the weights load directly
head = LMHead(emb.weight)                                                     # shares the same Parameter
x = emb(ids)
print(f"嵌入：{tuple(ids.shape)} -> {tuple(x.shape)}，和 HF 的 embed_tokens 最大差 {diff(x, hf.model.embed_tokens(ids)):.1e}")
logits = head(x)
print(f"输出层：{tuple(x.shape)} -> {tuple(logits.shape)}，和 HF 的 lm_head 最大差 {diff(logits, hf.lm_head(x)):.1e}")

for t, row in zip(ids[0], logits[0].topk(4).indices):                        # only the entrance and the exit: i.e. "which token's embedding is most similar"
    print(f"  {tok.decode(t)!r:12} 最像的：{[tok.decode(i) for i in row]}")
out = ids                                                                    # a "model" with nothing in between can still generate — it only repeats itself
for _ in range(5):
    out = torch.cat([out, head(emb(out))[0, -1].argmax().view(1, 1)], dim=1)
print("没有中间 24 层的模型接着写：", repr(tok.decode(out[0, ids.shape[1]:])))
print(f"嵌入表 {emb.weight.shape[0]} × {emb.weight.shape[1]}，{emb.weight.numel() / 1e6:.0f}M 参数，占全模型 {emb.weight.numel() / sum(v.numel() for v in W.values()):.0%}")
```

```text title="输出"
嵌入：(1, 6) -> (1, 6, 1024)，和 HF 的 embed_tokens 最大差 0.0e+00
输出层：(1, 6, 1024) -> (1, 6, 248320)，和 HF 的 lm_head 最大差 0.0e+00
  '今天'         最像的：['今天', '今天的', '今日', ' today']
  '天气'         最像的：['天气', '天氣', '的天气', 'weather']
  '不错'         最像的：['不错', '不錯', '不错的', '也不错']
  '，'          最像的：['，', ',', '。', ' ']
  '我们去'        最像的：['我们去', '你去', '我们为', '我去']
  '公园'         最像的：['公园', '公園', '公园里', ' parks']
没有中间 24 层的模型接着写： '公园公园公园公园公园'
嵌入表 248320 × 1024，254M 参数，占全模型 34%
```

The maximum difference is 0: a table lookup and a matrix multiplication leave no room for floating-point disagreement. The little experiment in the middle shows that the embedding table already carries some "meaning" — the tokens closest to "天气" (weather) mostly have to do with weather — but it knows nothing about context. The "model" at the end, with only an entrance and an exit, can only repeat itself: which token's embedding has the largest dot product with $x = E[\text{token}]$? Its own. The next fourteen chapters fill in what goes between these two layers.

!!! interview "How to explain it"
    When explaining tied embeddings, cover both directions: it saves parameters (a third of a small model), and it binds the "input space" and the "output space" into one, so a logit is the similarity between the hidden state and each token's embedding. Then add the cost: the big models (Qwen3.5 from 9B up) do not tie, because the embedding table is only a few percent of them and training the two separately is freer.

## Summary {#小结}

- [x] Embedding = take a row by id; output layer = dot with every row; the two share one 248320 × 1024 table.
- [x] Differences from HF's `embed_tokens` and `lm_head`: 0; a model with only an entrance and an exit can only repeat itself.

**File created in this chapter**: `qwen/embedding.py`.
