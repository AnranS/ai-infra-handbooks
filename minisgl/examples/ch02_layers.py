"""第 2 章：BaseOP 如何收集权重名；参考算子与 Hugging Face 实现的数值差异。"""

import torch
from minisgl.distributed import set_tp_info
from minisgl.kernel import torch_ops
from minisgl.layers.rotary import get_rope, set_rope_device
from minisgl.models import ModelConfig
from minisgl.models.decoder import DecoderLayer
from minisgl.utils import cached_load_hf_config, torch_dtype

set_tp_info(0, 1)
set_rope_device(torch.device("cpu"))  # RoPE 表是真实数据，不能放在 meta 设备上
config = ModelConfig.from_hf(cached_load_hf_config("models/Qwen3-0.6B"))
with torch.device("meta"), torch_dtype(torch.bfloat16):  # 只建结构，不分配内存
    layer = DecoderLayer(config, layer_id=0, has_qk_norm=True, use_moe=False)
for name, t in layer.state_dict(prefix="model.layers.0").items():
    print(f"{name:<50} {str(tuple(t.shape)):<14} {t.dtype} {t.device}")

# 与 HF 的 RMSNorm、RoPE 比较
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

torch.manual_seed(0)
x, w = torch.randn(4, 1024), torch.randn(1024)
ref = Qwen3RMSNorm(1024, eps=1e-6)
ref.weight.data.copy_(w)
print("RMSNorm 最大误差:", (torch_ops.rmsnorm(x, w, 1e-6) - ref(x)).abs().max().item())

rope = get_rope(128, 128, 4096, 1e6)
pos = torch.tensor([0, 7, 4095])
q, k = torch.randn(3, 16 * 128), torch.randn(3, 8 * 128)
inv = 1.0 / 1e6 ** (torch.arange(0, 128, 2).float() / 128)
emb = torch.cat([pos.float()[:, None] * inv] * 2, dim=-1)
q_ref, k_ref = apply_rotary_pos_emb(q.view(3, 16, 128).transpose(0, 1)[None],
                                    k.view(3, 8, 128).transpose(0, 1)[None],
                                    emb.cos()[None], emb.sin()[None])
rope.forward(pos, q, k)
print("RoPE 最大误差:", (q.view(3, 16, 128).transpose(0, 1) - q_ref[0]).abs().max().item())
