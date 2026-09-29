"""检查一：ModelConfig 能解析 Qwen3.5 的配置（语言部分在 text_config 里），原有模型不受影响"""
from common import MODEL, QWEN3, report

from minisgl.models import ModelConfig
from minisgl.utils import cached_load_hf_config

c = ModelConfig.from_hf(cached_load_hf_config(MODEL))
problems = []
want = {
    "num_layers": 24, "num_qo_heads": 8, "num_kv_heads": 2, "head_dim": 256, "hidden_size": 1024,
    "vocab_size": 248320, "intermediate_size": 3584, "tie_word_embeddings": True,
    "linear_num_key_heads": 16, "linear_num_value_heads": 16, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
}
for name, value in want.items():
    got = getattr(c, name, None)
    if got != value:
        problems.append(f"{name} = {got}，应为 {value}")
if (c.rotary_config.rotary_dim, c.rotary_config.base) != (64, 1e7):
    problems.append(f"RoPE：rotary_dim = {c.rotary_config.rotary_dim}、base = {c.rotary_config.base}，应为 64、1e7")
if getattr(c, "layer_types", None) != (["linear_attention"] * 3 + ["full_attention"]) * 6:
    problems.append("layer_types 应为每 4 层 3 个 linear_attention + 1 个 full_attention")
if getattr(c, "full_attention_layers", None) != [3, 7, 11, 15, 19, 23]:
    problems.append(f"full_attention_layers = {getattr(c, 'full_attention_layers', None)}")
if getattr(c, "linear_attention_layers", None) != [i for i in range(24) if i % 4 != 3]:
    problems.append("linear_attention_layers 不对")
if getattr(c, "is_hybrid", None) is not True:
    problems.append("is_hybrid 应为 True")

q3 = ModelConfig.from_hf(cached_load_hf_config(QWEN3))
if getattr(q3, "is_hybrid", None) is not False or getattr(q3, "full_attention_layers", None) != list(range(28)):
    problems.append("Qwen3-0.6B：is_hybrid 应为 False，full_attention_layers 应为全部 28 层")
if q3.rotary_config.rotary_dim != 128:
    problems.append("Qwen3-0.6B 的 rotary_dim 应仍为 128")

report("config", not problems, "；".join(problems))
