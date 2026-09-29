"""检查二：加载器产出的键与模型的 state_dict 完全一致（形状也一致），并且跳过了视觉编码器和 MTP 层"""
import torch
from common import MODEL, report

from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, create_model, load_weight
from minisgl.utils import cached_load_hf_config, torch_dtype

set_tp_info(0, 1)
set_rope_device(torch.device("cpu"))
config = ModelConfig.from_hf(cached_load_hf_config(MODEL))
with torch.device("meta"), torch_dtype(torch.float32):
    model = create_model(config)
expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
loaded = {k: tuple(v.shape) for k, v in load_weight(MODEL, torch.device("cpu"))}
loaded.pop("lm_head.weight", None)                              # 共享词嵌入
problems = []
extra = sorted(set(loaded) - set(expected))
missing = sorted(set(expected) - set(loaded))
if extra:
    problems.append(f"加载了模型里没有的键 {len(extra)} 个，例如 {extra[:3]}")
if missing:
    problems.append(f"模型里有、却没有加载的键 {len(missing)} 个，例如 {missing[:3]}")
if any(k.startswith(("model.visual", "visual", "mtp")) for k in loaded):
    problems.append("不应加载视觉编码器或 MTP 层的权重")
bad = [k for k in set(loaded) & set(expected) if loaded[k] != expected[k]]
if bad:
    problems.append(f"形状不一致：{[(k, loaded[k], expected[k]) for k in bad[:3]]}")
n_linear = sum(1 for k in expected if ".linear_attn." in k)
if n_linear == 0:
    problems.append("state_dict 里没有 linear_attn 的权重：线性注意力层的名字应与 checkpoint 对上")
report("weights", not problems, "；".join(problems) or f"{len(expected)} 个张量")
