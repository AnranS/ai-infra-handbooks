"""第 9 章：整模型和 HF 对 logits；只用 tokenizers 库和自己的模型生成第 0 步那句话"""
import os
import time

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()
path = find_model()
model = Model.from_pretrained(path)
print(f"自己的模型：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M 参数，{len(model.layers)} 层")

hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 1. 最后一次和 HF 对：整模型的 logits
tok = AutoTokenizer.from_pretrained(path)
for text in ("今天天气不错，我们去公园", "。".join(["今天天气不错，我们去公园"] * 40)):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = hf(ids).logits
    print(f"{ids.shape[1]:3d} 个 token：logits 最大差 {diff(model(ids), ref):.1e}")
del hf

tk = Tokenizer.from_file(str(path / "tokenizer.json"))              # 2. 不要 transformers 了：tokenizer.json 用 tokenizers 库直接读
prompt = "<|im_start|>user\n用一句话介绍一下 Transformer。<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # 第 12 章手写这个模板
ids = torch.tensor([tk.encode(prompt).ids])
print(f"提示词 {ids.shape[1]} 个 token，和第 0 步一样")
STOP = {tk.token_to_id("<|im_end|>"), tk.token_to_id("<|endoftext|>")}

t0 = time.time()                                                    # 3. 第 0 步的循环，换成自己的模型
out = ids
for _ in range(40):
    nxt = model(out)[0, -1].argmax()
    out = torch.cat([out, nxt.view(1, 1)], dim=1)
    if nxt.item() in STOP:
        break
new = out[0, ids.shape[1]:].tolist()
print(f"生成 {len(new)} 个 token，{time.time() - t0:.1f} 秒（每步都重算整段，第 10 章解决）")
print(tk.decode(new))
