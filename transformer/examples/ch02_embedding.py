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
