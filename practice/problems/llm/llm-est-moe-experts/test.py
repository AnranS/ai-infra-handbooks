from checker import check_close
from solution import decode_expert_bytes, expected_distinct_experts, moe_params

# DeepSeek-V3：每层 MLA 注意力的参数（q_a、q_b、kv_a、kv_b、o 五个矩阵）
MLA = 7168 * 1536 + 1536 * 128 * 192 + 7168 * 576 + 512 * 128 * 256 + 128 * 128 * 7168
DSV3 = {"d": 7168, "n_layers": 61, "n_dense_layers": 3, "dense_ffn": 18432, "moe_ffn": 2048, "n_experts": 256,
        "top_k": 8, "n_shared": 1, "attn_params": MLA, "vocab": 129280}
# Qwen3-235B-A22B：94 层都是 MoE，128 个专家选 8 个，没有共享专家
QWEN3 = {"d": 4096, "n_layers": 94, "moe_ffn": 1536, "n_experts": 128, "top_k": 8,
         "attn_params": 4096 * 64 * 128 * 2 + 4096 * 4 * 128 * 2, "vocab": 151936}


def test_example():
    total, active = moe_params(DSV3)
    check_close(total, 671e9, rtol=0.01, what="DeepSeek-V3 总参数")
    check_close(active, 37e9, rtol=0.03, what="DeepSeek-V3 激活参数")
    check_close(expected_distinct_experts(256, 8, 1), 8, what="1 个 token")
    check_close(expected_distinct_experts(256, 8, 64), 222.4, rtol=2e-3, what="64 个 token")


def test_other_model():
    total, active = moe_params(QWEN3)
    check_close(total, 235e9, rtol=0.01, what="Qwen3-235B 总参数")
    check_close(active, 22e9, rtol=0.02, what="Qwen3-235B 激活参数")


def test_distinct_limits():
    check_close(expected_distinct_experts(128, 8, 0), 0, atol=1e-12, what="0 个 token")
    check_close(expected_distinct_experts(128, 8, 10_000), 128, rtol=1e-9, what="token 很多时趋近全部专家")
    check_close(expected_distinct_experts(64, 64, 1), 64, what="top_k = 专家数")


def test_decode_bytes():
    expert = 3 * 7168 * 2048
    check_close(decode_expert_bytes(DSV3, 1), 58 * 9 * expert, rtol=1e-9, what="batch=1：只读 8 个路由专家 + 1 个共享专家")
    b64 = decode_expert_bytes(DSV3, 64)
    check_close(b64, 58 * (expected_distinct_experts(256, 8, 64) + 1) * expert, rtol=1e-9, what="batch=64")
    ratio = b64 / decode_expert_bytes(DSV3, 1)
    assert 20 < ratio < 30, f"batch 从 1 到 64，要读的专家权重增加约 25 倍，算出来 {ratio:.1f}"
    check_close(decode_expert_bytes(DSV3, 1, bytes_per_param=2), 2 * decode_expert_bytes(DSV3, 1), what="BF16 权重")
