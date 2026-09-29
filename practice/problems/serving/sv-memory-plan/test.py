from checker import check, check_close, raises
from solution import plan

LLAMA_8B = {"params": 8.03e9, "n_layers": 32, "n_kv_heads": 8, "head_dim": 128}
LLAMA_70B = {"params": 70.6e9, "n_layers": 80, "n_kv_heads": 8, "head_dim": 128}
GB = 2 ** 30


def test_example():
    p = plan(LLAMA_8B, 80, "bf16", "bf16", 8192)
    check_close(p["weight_gb"], 8.03e9 * 2 / GB, what="权重 GB")
    check(p["kv_bytes_per_token"], 131072, "每 token 的 KV 字节数")
    kv = int((80 * GB * 0.9 - 8.03e9 * 2 - 2 * GB) // 131072)
    check((p["kv_tokens"], p["max_seqs"], p["fits"]), (kv, kv // 8192, True), "KV 容量")


def test_fp8_kv_doubles_capacity():
    a = plan(LLAMA_8B, 80, "bf16", "bf16", 8192)
    b = plan(LLAMA_8B, 80, "bf16", "fp8", 8192)
    check(b["kv_bytes_per_token"], a["kv_bytes_per_token"] // 2, "FP8 KV 每 token 字节数减半")
    assert b["kv_tokens"] >= 2 * a["kv_tokens"] - 1, "KV 容量翻倍"


def test_int4_weights():
    p = plan(LLAMA_70B, 48, "int4", "fp8", 4096)
    wb = 70.6e9 * 0.5 + -(-70.6e9 // 128) * 2
    check_close(p["weight_gb"], wb / GB, what="int4 权重（含 scale）")
    check(p["fits"], True, "70B int4 放进 48 GB")


def test_does_not_fit():
    p = plan(LLAMA_70B, 80, "bf16", "bf16", 4096)
    check((p["kv_tokens"], p["max_seqs"], p["fits"]), (0, 0, False), "70B bf16 放不进一张 80 GB")
    with raises(ValueError, "未知 dtype"):
        plan(LLAMA_8B, 80, "fp4", "bf16", 1024)
