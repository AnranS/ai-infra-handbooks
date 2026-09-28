import torch
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.models import ModelConfig, create_model, load_weight, shard_tensor
from minisgl.utils import cached_load_hf_config, torch_dtype

from conftest import QWEN3, QWEN25


def test_model_config_from_hf():
    c = ModelConfig.from_hf(cached_load_hf_config(QWEN3))
    assert (c.num_layers, c.num_qo_heads, c.num_kv_heads, c.head_dim) == (28, 16, 8, 128)
    assert c.rotary_config.base == 1e6 and c.tie_word_embeddings and not c.attention_bias
    c2 = ModelConfig.from_hf(cached_load_hf_config(QWEN25))
    assert c2.attention_bias and c2.head_dim == 64 and c2.num_kv_heads == 2


def test_weight_loader_produces_exactly_the_model_keys():
    set_tp_info(0, 1)
    set_rope_device(torch.device("cpu"))
    config = ModelConfig.from_hf(cached_load_hf_config(QWEN3))
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    loaded = {k: tuple(v.shape) for k, v in load_weight(QWEN3, torch.device("cpu"))}
    # Qwen3-0.6B 共享词嵌入，但 checkpoint 里仍然存了一份 lm_head.weight，加载时由 ParallelLMHead 丢掉
    assert loaded.pop("lm_head.weight") == (151936, 1024)
    assert loaded == expected  # q/k/v 合并成 qkv_proj，gate/up 合并成 gate_up_proj
    assert expected["model.layers.0.self_attn.qkv_proj.weight"] == ((16 + 2 * 8) * 128, 1024)


def test_shard_tensor_replicates_kv_heads_when_tp_exceeds_kv_heads():
    w = torch.arange(4 * 3).view(4, 3)  # 2 个 KV 头，每头 2 行
    parts = [shard_tensor("x.k_proj.weight", w, r, 4, num_kv_heads=2) for r in range(4)]
    assert [p[:, 0].tolist() for p in parts] == [[0, 3], [0, 3], [6, 9], [6, 9]]
    rows = [shard_tensor("x.o_proj.weight", w, r, 3, num_kv_heads=2) for r in range(3)]
    assert [r.shape for r in rows] == [torch.Size([4, 1])] * 3  # 行并行按输入维切
