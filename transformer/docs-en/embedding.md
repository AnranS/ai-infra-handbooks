# 2. The embedding and the output layer

<p class="lead">The model's entrance and exit use the same table: on the way in, take one row per token id to get a 1024-dimensional vector; on the way out, dot the 1024-dimensional vector with every row of the table to get 248320 scores over the vocabulary. This is the simplest chapter of the book, but it replaces the block with the most parameters (254M, a third of the model), and it is the first taste of the "replace one part, check once" routine.</p>

!!! note "What this chapter runs"
    `ch02_embedding.py`, about 10 seconds.

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

## Plug it in and check {#换上去对答案}

```python title="ch02_embedding.py"
"""第 2 章：换掉嵌入层和输出层。顺便看看"只用嵌入表"能预测什么"""
import os

import torch

from qwen.embedding import Embedding, LMHead
from qwen.harness import load_hf, logits, max_diff, status
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf, tok = load_hf(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # replace the entrance: same names as in the file, weights load directly
hf.lm_head = LMHead(hf.model.embed_tokens.weight)                                              # replace the exit: shares the same Parameter
print(f"换掉嵌入层和输出层后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

emb = hf.model.embed_tokens                                      # without the 24 layers in between: embed -> lm_head directly, i.e. "which token's embedding is most similar"
near = (emb(ids) @ emb.weight.T)[0].topk(4).indices
for t, row in zip(ids[0], near):
    print(f"  {tok.decode(t)!r:12} 最像的：{[tok.decode(i) for i in row]}")
print(f"嵌入表 {emb.weight.shape[0]} × {emb.weight.shape[1]}，{emb.weight.numel() / 1e6:.0f}M 参数，占全模型 {emb.weight.numel() / sum(p.numel() for p in hf.parameters()):.0%}")
```

```text title="output"
换掉嵌入层和输出层后，logits 最大差 0.0e+00
已替换 2/9：embed_tokens✓  input_layernorm✗  post_attention_layernorm✗  mlp✗  rotary_emb✗  self_attn✗  linear_attn✗  norm✗  lm_head✓
  '今天'         最像的：['今天', '今天的', '今日', ' today']
  '天气'         最像的：['天气', '天氣', '的天气', 'weather']
  '不错'         最像的：['不错', '不錯', '不错的', '也不错']
  '，'          最像的：['，', ',', '。', ' ']
  '我们去'        最像的：['我们去', '你去', '我们为', '我去']
  '公园'         最像的：['公园', '公園', '公园里', ' parks']
嵌入表 248320 × 1024，254M 参数，占全模型 34%
```

The maximum difference is 0: a table lookup and a matrix multiplication leave no room for floating-point disagreement. The little experiment afterwards shows that the embedding table already carries some "meaning" — the tokens closest to "天气" (weather) mostly have to do with weather — but it knows nothing about context; the real prediction comes from the 24 layers in between.

!!! interview "How to explain it"
    When explaining tied embeddings, cover both directions: it saves parameters (a third of a small model), and it binds the "input space" and the "output space" into one, so a logit is the similarity between the hidden state and each token's embedding. Then add the cost: the big models (Qwen3.5 from 9B up) do not tie, because the embedding table is only a few percent of them and training the two separately is freer.

## Summary {#小结}

- [x] Embedding = take a row by id; output layer = dot with every row; the two share one 248320 × 1024 table.
- [x] After the replacement the maximum logit difference is 0; 2 of the skeleton's 9 slots are ours.

**File created in this chapter**: `qwen/embedding.py`.
