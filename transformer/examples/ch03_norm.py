"""第 3 章：三种归一化，和 HF 的同名模块对答案；看看零中心权重长什么样"""
import os

import torch
from transformers import AutoModelForCausalLM

from qwen.norm import GatedRMSNorm, RMSNorm, l2norm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
diff = lambda a, b: (a - b).abs().max().item()                   # 两个张量逐元素的最大差
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # 参考实现：只拿它的同名模块对答案
torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5

norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
print(f"RMSNorm 和 HF 的 layers[0].input_layernorm 最大差 {diff(norm(x), hf.model.layers[0].input_layernorm(x)):.1e}")
final = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
print(f"最后的 RMSNorm 和 HF 的 norm 最大差 {diff(final(x), hf.model.norm(x)):.1e}")
gated = load(GatedRMSNorm(cfg.linear_value_head_dim, cfg.rms_norm_eps), W, "layers.0.linear_attn.norm.")
h, z = torch.randn(96, cfg.linear_value_head_dim), torch.randn(96, cfg.linear_value_head_dim)
print(f"带门的 RMSNorm 和 HF 的 linear_attn.norm 最大差 {diff(gated(h, z), hf.model.layers[0].linear_attn.norm(h, z)):.1e}")
print("l2norm 之后向量的长度：", [f"{v:.3f}" for v in l2norm(x).norm(dim=-1).flatten()[:3]])

w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 它把尺度调到嵌入表的尺度")
y = RMSNorm(cfg.hidden_size)(x)                                   # 权重为 0 时，归一化之后每个向量的均方根都是 1
print("归一化前每个向量的均方根：", [f"{v:.2f}" for v in x.pow(2).mean(-1).sqrt().flatten()[:3]],
      "归一化后：", [f"{v:.2f}" for v in y.pow(2).mean(-1).sqrt().flatten()[:3]])
print(f"全模型 {cfg.num_hidden_layers * 2 + 1} 处 RMSNorm、{cfg.layer_types.count('full_attention') * 2} 处按头的 QK-Norm、"
      f"{cfg.layer_types.count('linear_attention')} 处带门的 RMSNorm，都是这两个类")
