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

hf.model.embed_tokens = load(Embedding(cfg.vocab_size, cfg.hidden_size), W, "embed_tokens.")   # 换进门：名字和文件里一样，权重直接灌
hf.lm_head = LMHead(hf.model.embed_tokens.weight)                                              # 换出门：共用同一个 Parameter
print(f"换掉嵌入层和输出层后，logits 最大差 {max_diff(ref, logits(hf, ids)):.1e}")
print(status(hf))

emb = hf.model.embed_tokens                                      # 没有中间 24 层会怎样：直接 embed -> lm_head，等于问"和它的嵌入最像的 token 是谁"
near = (emb(ids) @ emb.weight.T)[0].topk(4).indices
for t, row in zip(ids[0], near):
    print(f"  {tok.decode(t)!r:12} 最像的：{[tok.decode(i) for i in row]}")
print(f"嵌入表 {emb.weight.shape[0]} × {emb.weight.shape[1]}，{emb.weight.numel() / 1e6:.0f}M 参数，占全模型 {emb.weight.numel() / sum(p.numel() for p in hf.parameters()):.0%}")
