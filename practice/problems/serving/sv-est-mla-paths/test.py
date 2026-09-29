from checker import check, check_close
from solution import choose, crossover_q, path_flops

V3 = {"H": 128, "nope": 128, "rope": 64, "dv": 128, "dc": 512}


def test_example():
    check(choose(1, 8192, V3), "absorb", "decode")
    check(choose(8192, 8192, V3), "expand", "prefill")
    check(round(crossover_q(V3)), 171, "分界点")


def test_flops():
    e, a = path_flops(1, 8192, V3)
    check_close(e, 8192 * 2 * 512 * 128 * 256 + 8192 * 81920, what="decode 走展开")
    check_close(a, 2 * 128 * 128 * 512 * 2 + 8192 * 278528, what="decode 走吸收")
    e, a = path_flops(4, 4, V3)
    check_close(e, 4 * 33554432 + 10 * 81920, what="4 个 token 的 prefill：10 对")


def test_extend():
    check(choose(64, 32768 + 64, V3), "absorb", "长前缀后追加 64 个 token")
    check(choose(512, 32768 + 512, V3), "expand", "长前缀后追加 512 个 token")
    small = {"H": 16, "nope": 64, "rope": 32, "dv": 64, "dc": 256}
    check(choose(1, 4096, small), "absorb", "小模型的 decode 也走吸收")
    assert crossover_q(small) > 0
