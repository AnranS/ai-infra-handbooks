from checker import check, check_close
from solution import best_k, expected_advance, layer_time, mtp_throughput


def close_all(got, want, what, **kw):
    """逐个比较（不依赖 numpy）"""
    got, want = list(got), list(want)
    check(len(got), len(want), f"{what}：个数")
    for i, (g, w) in enumerate(zip(got, want)):
        check_close(g, w, what=f"{what}（第 {i + 1} 个）", **kw)

A = [0.85, 0.75, 0.65]


def test_example():
    check_close(mtp_throughput(16, 4096, 1, A) / mtp_throughput(16, 4096, 0, A), 1.8202, rtol=1e-3, what="16 个请求、4K")
    check(best_k(64, 4096, A), 0, "64 个请求、4K：不用 MTP")


def test_advance():
    check_close(expected_advance(A), 1 + 0.85 + 0.85 * 0.75 + 0.85 * 0.75 * 0.65, what="3 个草稿")
    check_close(expected_advance([]), 1.0, what="不用草稿")
    check_close(expected_advance([0.5, 0.5]), 1.75, what="1 + 0.5 + 0.25")


def test_layer():
    check_close(layer_time(16, 32, 4096), 0.00011776043940298508, rtol=1e-9, what="16 个请求、验证 32 个 token")
    check_close(layer_time(64, 128, 4096), 0.00044498944, rtol=1e-9, what="64 个请求：all-to-all 成了瓶颈")
    check_close(layer_time(64, 128, 4096), 128 * 8 * ((7168 + 224) + 7168 * 2) / 50e9, rtol=1e-12,
                what="这时一层的时间就是 all-to-all 的时间")


def test_table():
    table = {(4096, 16): (2227.361610545793, [1.8202, 1.7157, 1.4969]),
             (4096, 64): (4715.529105000121, [0.9175, 0.8202, 0.7167]),
             (16384, 16): (1414.9804072013553, [1.804, 1.7273, 1.5645]),
             (16384, 64): (2301.799084171586, [1.2987, 1.1597, 1.0127])}
    for (ctx, b), (base, ratios) in table.items():
        got = mtp_throughput(b, ctx, 0, A)
        check_close(got, base, rtol=1e-9, what=f"上下文 {ctx}、每卡 {b} 个请求、不用 MTP 的吞吐")
        close_all([mtp_throughput(b, ctx, k, A) / got for k in (1, 2, 3)], ratios, rtol=1e-3,
                    what=f"上下文 {ctx}、每卡 {b} 个请求时 1～3 个草稿的加速比")


def test_best_k():
    check([best_k(b, ctx, A) for ctx, b in ((4096, 16), (4096, 64), (16384, 16), (16384, 64))], [1, 0, 1, 1],
          "四种配置的最佳草稿数")
    check(best_k(16, 16384, [0.99, 0.99, 0.99]), 3, "上下文长、接受率很高时多用草稿")
    check(best_k(16, 4096, [0.99, 0.99, 0.99]), 2, "上下文短时，第 3 个草稿抵不过 MTP 模块多跑的一次")
    check(best_k(16, 4096, A, k_max=0), 0, "不允许用草稿")
