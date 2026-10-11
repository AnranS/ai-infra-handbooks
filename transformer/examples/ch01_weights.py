"""第 1 章：读配置和权重，和 HF 加载的逐个对；记下替换前的 logits"""
import os

import torch

from qwen.harness import load_hf, logits, max_diff, status
from qwen.weights import Config, find_model, load_weights, sub

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
print(f"{cfg.num_hidden_layers} 层，hidden {cfg.hidden_size}，词表 {cfg.vocab_size}，RoPE theta {cfg.rope_theta:.0e}，只旋转 {cfg.partial_rotary_factor:.0%} 的维度")

W = load_weights(path)
print(f"语言模型 {len(W)} 个张量，{sum(v.numel() for v in W.values()) / 1e6:.1f}M 参数（文件 bf16 -> 内存 fp32）")
print("lm_head.weight 在文件里吗：", "lm_head.weight" in W, "；tie_word_embeddings =", cfg.tie_word_embeddings)
print("第 0 层（线性注意力）的张量：", sorted(sub(W, "layers.0.").keys()))
print("第 3 层（全注意力）的张量：  ", sorted(sub(W, "layers.3.").keys()))

hf, tok = load_hf(path)                                                   # 和 HF 加载的逐个对
sd = hf.state_dict()
mismatch = [k for k, v in W.items() if not torch.equal(v, sd["model." + k])]
print(f"和 HF 的 state_dict 对比：{len(W) - len(mismatch)} 个相同，{len(mismatch)} 个不同；HF 多出来的键：",
      sorted(set(k[6:] for k in sd if k.startswith("model.")) - set(W)) + [k for k in sd if not k.startswith("model.")])

ids = tok("今天天气不错，我们去公园", return_tensors="pt")["input_ids"]
ref = logits(hf, ids)
print(f"替换前的 logits：形状 {tuple(ref.shape)}，下一个 token 的 top-3：", [tok.decode(i) for i in ref[0, -1].topk(3).indices])
print(status(hf))
