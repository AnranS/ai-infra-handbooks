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
