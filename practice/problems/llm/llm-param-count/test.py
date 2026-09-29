from checker import check
from solution import count_params

QWEN3_0_6B = {"vocab_size": 151936, "hidden_size": 1024, "num_hidden_layers": 28, "intermediate_size": 3072,
              "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": True,
              "qk_norm": True}
QWEN3_30B_A3B = {"vocab_size": 151936, "hidden_size": 2048, "num_hidden_layers": 48, "intermediate_size": 6144,
                 "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "tie_word_embeddings": False,
                 "qk_norm": True, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768}
LLAMA2_7B = {"vocab_size": 32000, "hidden_size": 4096, "num_hidden_layers": 32, "intermediate_size": 11008,
             "num_attention_heads": 32}


def test_example():
    check(count_params(QWEN3_0_6B), (596049920, 596049920), "Qwen3-0.6B")
    check(count_params(QWEN3_30B_A3B), (30532122624, 3353032704), "Qwen3-30B-A3B")


def test_llama2_defaults():
    """没有 head_dim、num_key_value_heads、tie_word_embeddings 字段时用默认值"""
    check(count_params(LLAMA2_7B), (6738415616, 6738415616), "LLaMA-2-7B")


def test_moe_top_all_equals_total():
    c = dict(QWEN3_30B_A3B, num_experts_per_tok=128)
    total, active = count_params(c)
    check(active, total, "每个 token 用全部专家时，激活参数量 = 总参数量")


def test_small_handmade():
    c = {"vocab_size": 10, "hidden_size": 4, "num_hidden_layers": 2, "intermediate_size": 8, "num_attention_heads": 2,
         "num_key_value_heads": 1, "num_experts": 3, "num_experts_per_tok": 2, "moe_intermediate_size": 5,
         "qk_norm": True, "tie_word_embeddings": True}
    hd = 2
    attn = 4 * 2 * hd + 2 * 4 * 1 * hd + 2 * hd * 4
    layer = attn + 2 * 4 + 2 * hd + 4 * 3 + 3 * (3 * 4 * 5)
    total = 10 * 4 + 2 * layer + 4
    check(count_params(c), (total, total - 2 * 1 * 3 * 4 * 5), "手算的小模型")
