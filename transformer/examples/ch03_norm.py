"""第 3 章：三种归一化。看归一化前后的尺度、真实权重把尺度调到了哪里、为什么要存 w-1"""
import os

import torch

from qwen.norm import GatedRMSNorm, RMSNorm, l2norm
from qwen.weights import Config, find_model, load, load_weights

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
cfg = Config.load(path)
W = load_weights(path)
torch.manual_seed(0)
x = torch.randn(2, 3, cfg.hidden_size) * 5
rms = lambda t: [f"{v:.2f}" for v in t.float().pow(2).mean(-1).sqrt().flatten()[:3]]

print("归一化前每个向量的均方根：", rms(x), "；权重为 0（即比例为 1）时归一化后：", rms(RMSNorm(cfg.hidden_size)(x)))   # 1. 它做的事
norm = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "layers.0.input_layernorm.")
final = load(RMSNorm(cfg.hidden_size, cfg.rms_norm_eps), W, "norm.")
print("第 0 层输入归一化之后：", rms(norm(x)), "；最后一层归一化之后：", rms(final(x)), "（它把尺度放大到嵌入表的尺度）")
w = W["layers.0.input_layernorm.weight"]
print(f"第 0 层输入归一化的权重：均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}] —— 存的是 w-1，所以在 0 附近")
w = W["norm.weight"]
print(f"最后一层的归一化权重：  均值 {w.mean():+.3f}，范围 [{w.min():+.2f}, {w.max():+.2f}]")

one = torch.tensor(1.0 + 1e-3)                                              # 2. 为什么存 w-1：bf16 在 1 附近粗、在 0 附近细
print(f"bf16 存 1 + 0.001 得到 {one.bfloat16().item():.4f}（更新被吞掉）；存 0.001 得到 {torch.tensor(1e-3).bfloat16().item():.6f}（留住了）")
print(f"bf16 输入 -> 输出 dtype {RMSNorm(cfg.hidden_size)(x.bfloat16()).dtype}，统计量在 fp32 里算")

gated = load(GatedRMSNorm(cfg.linear_value_head_dim, cfg.rms_norm_eps), W, "layers.0.linear_attn.norm.")   # 3. 带门的版本：门关上输出就没了
h = torch.randn(4, cfg.linear_value_head_dim)
for g in (-6.0, 0.0, 6.0):
    print(f"  门 = {g:+.0f}：输出的均方根 {gated(h, torch.full_like(h, g)).pow(2).mean(-1).sqrt().mean():.3f}（silu({g:+.0f}) = {torch.nn.functional.silu(torch.tensor(g)):.3f}）")
print("l2norm 之后向量的长度：", [f"{v:.3f}" for v in l2norm(x).norm(dim=-1).flatten()[:3]])
print(f"全模型 {cfg.num_hidden_layers * 2 + 1} 处 RMSNorm、{cfg.layer_types.count('full_attention') * 2} 处按头的 QK-Norm、{cfg.layer_types.count('linear_attention')} 处带门的 RMSNorm，都是这两个类")
