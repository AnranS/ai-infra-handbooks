from checker import check, check_close
from solution import choose, kv_bytes, recompute_ms, swap_ms

M70 = {"layers": 80, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "params": 70e9}
M8 = {"layers": 32, "kv_heads": 8, "head_dim": 128, "dtype_bytes": 2, "params": 8e9}


def test_example():
    check(kv_bytes(32768, 80, 8, 128, 2), 10 * 2**30, "32K token 的 KV 是 10 GiB")
    decision, s, r = choose(32768, M70, 50, 600)
    check(decision, "swap", "长上下文换回来更快")
    check_close(s, 214.7483648, rtol=1e-6, what="换回的毫秒数")
    check_close(r, 7645.866666666667, rtol=1e-6, what="重算的毫秒数")


def test_parts():
    check_close(swap_ms(50e9, 50), 1000.0, what="50 GB 走 50 GB/s 是 1 秒")
    check_close(swap_ms(0, 50, overhead_ms=0.5), 0.5, what="固定开销")
    check_close(recompute_ms(1000, 1e9, 1), 2000.0, what="1B 参数、1000 token、1 TFLOPS")


def test_independent_of_length():
    for tokens in (128, 4096, 131072):
        check(choose(tokens, M8, 25, 400)[0], "swap", f"没有固定开销时和长度无关（{tokens} token）")


def test_overhead_flips_short():
    check(choose(16, M8, 25, 400, overhead_ms=2.0)[0], "recompute", "很短的上下文：固定开销让重算更快")
    check(choose(8192, M8, 25, 400, overhead_ms=2.0)[0], "swap", "长上下文：固定开销可以忽略")


def test_slow_link_big_gqa():
    mha = {"layers": 32, "kv_heads": 32, "head_dim": 128, "dtype_bytes": 2, "params": 7e9}
    check(choose(4096, mha, 2, 900)[0], "recompute", "没有 GQA、链路很慢（2 GB/s）、算力很强时重算更快")
