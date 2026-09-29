from checker import check, raises
from solution import fused_qkv_rows, train_sources


def test_example():
    check(fused_qkv_rows(1, 4, heads=8, kv_heads=2, d=8), [("q", 16, 32), ("k", 0, 8), ("v", 0, 8)], "TP=4、2 个 KV 头的 rank 1")
    check(train_sources(3, infer_tp=4, train_tp=2, heads=8, kv_heads=2), [1], "推理 rank 3 的数据只在训练 rank 1 上")


def test_rows():
    check(fused_qkv_rows(0, 2, heads=8, kv_heads=2, d=8), [("q", 0, 32), ("k", 0, 8), ("v", 0, 8)], "KV 头数等于 TP")
    check(fused_qkv_rows(3, 4, heads=32, kv_heads=8, d=128), [("q", 3072, 4096), ("k", 768, 1024), ("v", 768, 1024)], "KV 头多于 TP")
    check(fused_qkv_rows(7, 8, heads=64, kv_heads=8, d=128), [("q", 7168, 8192), ("k", 896, 1024), ("v", 896, 1024)], "Llama-3-70B，TP=8")
    kv = [fused_qkv_rows(r, 8, heads=16, kv_heads=2, d=4)[1] for r in range(8)]
    check(kv, [("k", 0, 4)] * 4 + [("k", 4, 8)] * 4, "每个 KV 头复制到 4 个 rank")
    q = [fused_qkv_rows(r, 8, heads=16, kv_heads=2, d=4)[0] for r in range(8)]
    check(q, [("q", 8 * r, 8 * r + 8) for r in range(8)], "Q 头不复制，连续均分")


def test_invalid():
    with raises(ValueError):
        fused_qkv_rows(0, 3, heads=8, kv_heads=2, d=8)
    with raises(ValueError):
        fused_qkv_rows(0, 4, heads=12, kv_heads=6, d=8)
    with raises(ValueError):
        fused_qkv_rows(0, 4, heads=8, kv_heads=3, d=8)


def test_sources():
    check([train_sources(r, 4, 2, 8, 2) for r in range(4)], [[0], [0], [1], [1]], "推理 TP 是训练 TP 的两倍")
    check([train_sources(r, 2, 4, 32, 8) for r in range(2)], [[0, 1], [2, 3]], "推理 TP 比训练小：要从两个训练 rank 取")
    check([train_sources(r, 8, 8, 64, 8) for r in range(8)], [[r] for r in range(8)], "TP 相同：一一对应")
    with raises(ValueError):
        train_sources(0, 4, 4, 8, 2)
