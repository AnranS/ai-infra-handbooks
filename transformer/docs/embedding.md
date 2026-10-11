# （二）嵌入层与输出层

<p class="lead">模型的进门和出门用的是同一张表：进门时按 token id 取一行，得到 1024 维的向量；出门时把 1024 维的向量和这张表的每一行做点积，得到词表上 248320 个分数。这是全书最简单的一章，但它换掉的是参数最多的那块（254M，占 1/3），也是第一次体验"换一块、对一次"的流程。</p>

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

## 和 HF 的同名模块对答案

```python title="ch02_embedding.py"
"""第 2 章：嵌入层和输出层，和 HF 的同名模块对答案；一个只有进门和出门的"模型"能写出什么"""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen.embedding import Embedding, LMHead
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # 两个张量逐元素的最大差
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 参考实现：只拿它的同名模块对答案
tok = AutoTokenizer.from_pretrained(path)
ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]

emb = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 名字和文件里一样，权重直接灌
head = LMHead(emb.weight)                                                     # 共用同一个 Parameter
x = emb(ids)
print(f"嵌入：{tuple(ids.shape)} -> {tuple(x.shape)}，和 HF 的 embed_tokens 最大差 {diff(x, hf.model.embed_tokens(ids)):.1e}")
logits = head(x)
print(f"输出层：{tuple(x.shape)} -> {tuple(logits.shape)}，和 HF 的 lm_head 最大差 {diff(logits, hf.lm_head(x)):.1e}")

for t, row in zip(ids[0], logits[0].topk(4).indices):                        # 只有进门和出门：等于问"和它的嵌入最像的 token 是谁"
    print(f"  {tok.decode(t)!r:12} 最像的：{[tok.decode(i) for i in row]}")
out = ids                                                                    # 中间什么都没有的"模型"也能生成——只会复读
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

最大差是 0：查表和矩阵乘没有任何浮点误差可言。中间的小实验说明嵌入表本身已经有一点"意思"——和"天气"最像的几个 token 多半也和天气有关——但它不知道上下文。最后那个只有进门和出门的"模型"只会复读：$x = E[\text{token}]$ 和谁的点积最大？就是它自己。后面十四章做的事，就是往这两层之间填东西。

!!! interview "怎么讲清楚"
    讲 tie embedding 时把两个方向说清：省参数（小模型里是三分之一）；以及它把"输入空间"和"输出空间"绑成了同一个空间，所以 logit 就是隐状态和各个 token 嵌入的相似度。再补一句代价：大模型（9B 以上的 Qwen3.5）不绑，因为此时嵌入表只占百分之几，分开训练更自由。

## 小结

- [x] 嵌入 = 按 id 取行；输出层 = 和每一行做点积；两者共用一张 248320 × 1024 的表。
- [x] 和 HF 的 `embed_tokens`、`lm_head` 最大差 0；只有进门和出门的模型只会复读。

**这一章新建的文件**：`qwen/embedding.py`。
