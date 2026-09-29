from checker import check, check_close
from solution import bound, decode_attn_intensity, tp_intensity

H100 = (989, 3350)


def test_example():
    check_close(decode_attn_intensity("mha", 32), 1.0, what="MHA")
    check_close(decode_attn_intensity("gqa", 64, n_kv_heads=8), 8.0, what="GQA 组大小为 8")
    check_close(decode_attn_intensity("mla", 128), 128 * (2 * 576 + 2 * 512) / (576 * 2), rtol=1e-12, what="MLA 128 头")
    check(bound(decode_attn_intensity("mla", 128), *H100), "memory", "MLA q_len=1 在 H100 上")


def test_q_len_and_dtype():
    check_close(decode_attn_intensity("mla", 128, q_len=2), 2 * decode_attn_intensity("mla", 128), what="q_len=2")
    check(bound(decode_attn_intensity("mla", 128, q_len=2), *H100), "compute", "MTP 一次验证 2 个 token")
    check_close(decode_attn_intensity("gqa", 64, n_kv_heads=8, dtype_bytes=1), 16.0, what="FP8 KV")
    check_close(decode_attn_intensity("gqa", 32, n_kv_heads=8, head_dim=64), 4.0, what="与头维度无关")


def test_tp():
    check_close(tp_intensity("mla", 128, 8), decode_attn_intensity("mla", 16), what="MLA、TP=8：每卡 16 头，KV 不切")
    check_close(tp_intensity("gqa", 64, 8, n_kv_heads=8), 8.0, what="GQA、TP=8：每卡 8 个 query 头配 1 个 KV 头")
    check_close(tp_intensity("gqa", 64, 16, n_kv_heads=8), 4.0, what="TP=16 超过 KV 头数：KV 头被复制")
    check_close(tp_intensity("mha", 32, 4), 1.0, what="MHA 切分后仍是 1")


def test_ridge():
    check(bound(295.3, *H100), "compute", "略高于屋脊点")
    check(bound(295.0, *H100), "memory", "略低于屋脊点")
