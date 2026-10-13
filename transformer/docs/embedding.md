# （二）嵌入层与输出层

<p class="lead">模型的进门和出门用的是同一张表：进门时按 token id 取一行，得到 1024 维的向量；出门时把 1024 维的向量和这张表的每一行做点积，得到词表上 248320 个分数。这是全书最简单的一章，但它换掉的是参数最多的那块（254M，占 1/3），也是第一次体验"写一块、跑一个小实验"的节奏。</p>

!!! note "这一章跑什么"
    `ch02_embedding.py`，约 10 秒。

@@files@@

## 查表进门，点积出门

```python title="qwen/embedding.py"
"""嵌入层与输出层：进门查表把 token id 变成向量，出门用同一张表把向量变回词表上的分数。"""
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))   # 一行一个 token

    def forward(self, ids):
        return self.weight[ids]                                     # 查表就是取行：[B, T] -> [B, T, dim]


class LMHead(nn.Module):
    """tie_word_embeddings=True：输出层直接用嵌入层那张表，不另存一份（省下 254M 参数）"""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight                                        # 同一个 Parameter 对象

    def forward(self, x):
        return x @ self.weight.T                                    # [B, T, dim] @ [dim, vocab] -> 每个位置在词表上的分数
```

`Embedding.forward` 就是一次高级索引：`weight[ids]`，形状从 `[B, T]` 变成 `[B, T, 1024]`。`LMHead` 是一次矩阵乘，把 1024 维投回 248320 维——这一步在小模型上是最贵的单个算子之一（1024 × 248320 ≈ 254M 次乘加，一层前馈网络才 11M 参数）。

**共用权重（tie）意味着什么**：输出分数 = 隐状态 · 嵌入向量。模型要想让"公园"得高分，就得把最后一层的输出推向"公园"这个 token 的嵌入向量。所以嵌入空间既是输入空间也是输出空间，这也是为什么最后一层前面必须有 RMSNorm（第 3 章）——把尺度统一到嵌入表的尺度上。

## 跑起来：嵌入空间里谁和谁近

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
tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # tokenizer.json 用 tokenizers 库直接读
emb = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 名字和文件里一样，权重直接灌
head = LMHead(emb.weight)                                                     # 共用同一个 Parameter

ids = torch.tensor([tk.encode("今天天气不错，我们去公园").ids])
x = emb(ids)
print(f"进门：{tuple(ids.shape)} 个 token id -> {tuple(x.shape)} 的向量；出门：-> {tuple(head(x).shape)} 的分数")
print(f"嵌入表 {emb.weight.shape[0]} × {emb.weight.shape[1]}，{emb.weight.numel() / 1e6:.0f}M 参数，占全模型 {emb.weight.numel() / sum(v.numel() for v in W.values()):.0%}")

E = F.normalize(emb.weight, dim=-1)                                          # 1. 嵌入空间里谁和谁近：按余弦相似度找最像的 token
for word in ("今天", "公园", "北京", "快乐", "128"):
    i = tk.encode(word).ids[0]
    near = (E @ E[i]).topk(6).indices[1:]
    print(f"  {word!r:8} 最像的：{[tk.decode([j]) for j in near]}")
a, b, c = (E[tk.encode(w).ids[0]] for w in ("今天", "明天", "公园"))          # 2. 相似度是个数：同类词高，不相干的词低
print(f"  cos(今天, 明天) = {a @ b:.2f}，cos(今天, 公园) = {a @ c:.2f}")

out = ids                                                                    # 3. 只有进门和出门的"模型"：接着写 5 个 token
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

嵌入表本身已经有一点"意思"：和"今天"最像的是"今日"和 today，和"北京"最像的是"上海"，"128" 的邻居全是数字；用余弦相似度量，今天和明天是 0.46，今天和公园只有 0.12。但它不知道上下文——"公园"永远是同一个向量。最后那个只有进门和出门的"模型"只会复读：$x = E[\text{token}]$ 和谁的点积最大？就是它自己。后面的章节做的事，就是往这两层之间填东西，让"公园"的向量因为前面是"我们去"而变成别的东西。

!!! interview "怎么讲清楚"
    讲 tie embedding 时把两个方向说清：省参数（小模型里是三分之一）；以及它把"输入空间"和"输出空间"绑成了同一个空间，所以 logit 就是隐状态和各个 token 嵌入的相似度。再补一句代价：大模型（9B 以上的 Qwen3.5）不绑，因为此时嵌入表只占百分之几，分开训练更自由。

## 小结

- [x] 嵌入 = 按 id 取行；输出层 = 和每一行做点积；两者共用一张 248320 × 1024 的表。
- [x] 嵌入空间里相近的词靠得近（今天 / 明天 0.46，今天 / 公园 0.12）；只有进门和出门的模型只会复读。

**这一章新建的文件**：`qwen/embedding.py`。
