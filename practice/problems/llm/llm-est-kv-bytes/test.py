import math

from checker import check, check_close
from solution import kv_bytes, kv_bytes_per_token, max_context

LLAMA3_70B = {"attn": "gqa", "n_layers": 80, "n_kv_heads": 8, "head_dim": 128}
DSV3 = {"attn": "mla", "n_layers": 61, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
LLAMA2_7B = {"attn": "mha", "n_layers": 32, "n_heads": 32, "head_dim": 128}
HYBRID = {"attn": "gqa", "n_layers": 48, "n_kv_heads": 8, "head_dim": 128, "window": 4096, "n_global_layers": 8}


def test_example():
    check(kv_bytes_per_token(LLAMA3_70B), 327680, "Llama-3-70B")
    check(kv_bytes_per_token(DSV3), 70272, "DeepSeek-V3（MLA）")
    check(max_context(LLAMA3_70B, 40), 131072, "40 GiB 放下的最长序列")


def test_kinds_and_dtype():
    check(kv_bytes_per_token(LLAMA2_7B), 524288, "MHA：每 token 512 KiB")
    check(kv_bytes_per_token(LLAMA3_70B, dtype_bytes=1), 163840, "FP8 KV 减半")
    check_close(kv_bytes_per_token(LLAMA3_70B) / kv_bytes_per_token(DSV3), 4.663, rtol=1e-3, what="GQA / MLA")


def test_full_sequence():
    check(kv_bytes(LLAMA3_70B, 8192), 8192 * 327680, "没有滑动窗口时就是线性增长")
    check(max_context(DSV3, 40), 611191, "MLA：40 GiB 能放 61 万 token")


def test_sliding_window():
    per = 2 * 8 * 128 * 2
    check(kv_bytes(HYBRID, 1000), per * 48 * 1000, "短于窗口时所有层都存完整序列")
    check(kv_bytes(HYBRID, 131072), per * (8 * 131072 + 40 * 4096), "128K：局部层只存 4096 个 token")
    full = dict(HYBRID)
    del full["window"]
    ratio = kv_bytes(full, 131072) / kv_bytes(HYBRID, 131072)
    assert 5.0 < ratio < 5.4, f"128K 时全局注意力的 KV 约是混合结构的 5.2 倍，算出来 {ratio:.2f}"
    n = max_context(HYBRID, 8)
    assert kv_bytes(HYBRID, n) <= 8 * 2 ** 30 < kv_bytes(HYBRID, n + 1), "max_context 要恰好是放得下的最长序列"
    local_only = dict(HYBRID, n_global_layers=0)
    check(max_context(local_only, 8), math.inf, "全部是滑动窗口时长度不受 KV 限制")
    check(max_context(HYBRID, 0.1), int(0.1 * 2 ** 30 // (per * 48)), "预算连一个窗口都放不下时")
