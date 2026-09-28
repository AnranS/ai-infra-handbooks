"""第 3 章：从 config.json 到可以计算的模型。"""

import time

import torch
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, create_model, load_weight
from minisgl.utils import cached_load_hf_config, torch_dtype

set_tp_info(0, 1)
set_rope_device(torch.device("cpu"))
config = ModelConfig.from_hf(cached_load_hf_config("models/Qwen3-0.6B"))
print(f"层数 {config.num_layers}，q 头 {config.num_qo_heads}，kv 头 {config.num_kv_heads}，"
      f"head_dim {config.head_dim}，hidden {config.hidden_size}，词表 {config.vocab_size}")
print(f"RoPE base {config.rotary_config.base:.0f}，共享词嵌入 {config.tie_word_embeddings}，"
      f"结构 {config.architectures[0]}")

t = time.perf_counter()
with torch.device("meta"), torch_dtype(torch.bfloat16):
    model = create_model(config)
print(f"在 meta 设备上建模型：{(time.perf_counter() - t) * 1e3:.0f} ms，"
      f"权重 {len(model.state_dict())} 个张量")

t = time.perf_counter()
weights = {}
for i, (name, tensor) in enumerate(load_weight("models/Qwen3-0.6B", torch.device("cpu"))):
    if i < 6:
        print(f"  yield {name:<48} {tuple(tensor.shape)}")
    weights[name] = tensor
print(f"流式读取 {len(weights)} 个张量：{time.perf_counter() - t:.2f} s")
model.load_state_dict(weights)  # lm_head.weight 被共享词嵌入的输出层丢掉
w = model.model.layers.op_list[0].self_attn.qkv_proj.weight
print("加载后 qkv_proj:", tuple(w.shape), w.dtype, w.device)
