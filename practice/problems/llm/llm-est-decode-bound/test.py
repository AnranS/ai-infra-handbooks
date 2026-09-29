from checker import check_close
from solution import decode_step_ms, decode_tokens_per_s, prefill_ms

W70, KV70 = 141e9, 327680


def test_example():
    base = decode_step_ms(W70, KV70, 32, 4096, 3350, tp=8)
    check_close(base, (141e9 + 32 * 4096 * KV70) / 8 / 3350e9 * 1e3, rtol=1e-12, what="权重 + KV")
    check_close(base, 6.86, rtol=0.01, what="约 6.9 ms")
    check_close(decode_step_ms(W70, KV70, 32, 4096, 3350, tp=8, n_allreduce=160, allreduce_us=10), base + 1.6,
                rtol=1e-9, what="加上 160 次 all-reduce")


def test_batch_scaling():
    t1 = decode_tokens_per_s(W70, KV70, 1, 4096, 3350, tp=8)
    t64 = decode_tokens_per_s(W70, KV70, 64, 4096, 3350, tp=8)
    assert 30 < t64 / t1 < 64, f"batch 1 → 64 吞吐大幅上升但不到 64 倍（KV 读取在增长），算出来 {t64 / t1:.1f} 倍"
    long_ctx = 64 * 32768 * KV70
    assert long_ctx > W70, "64 条 32K 序列的 KV 已经比 70B 的权重还大"
    check_close(decode_tokens_per_s(W70, KV70, 16, 1000, 3350), 16 / decode_step_ms(W70, KV70, 16, 1000, 3350) * 1e3,
                rtol=1e-12, what="tokens/s = batch / 单步时间")


def test_prefill():
    t = prefill_ms(70.6e9, 8192, 80, 8192, 989, mfu=0.5, tp=8)
    flops = 2 * 70.6e9 * 8192 + 2 * 80 * 8192 * 8192 ** 2
    check_close(t, flops / (8 * 989e12 * 0.5) * 1e3, rtol=1e-12, what="70B、8K prompt、8 卡")
    share = 2 * 80 * 8192 * 8192 ** 2 / flops
    assert 0.05 < share < 0.1, f"8K prompt 时因果注意力约占 prefill FLOPs 的 7.5%，算出来 {share:.1%}"
    check_close(prefill_ms(8e9, 1000, 32, 4096, 989, mfu=1.0), (16e12 + 2 * 32 * 4096 * 1e6) / 989e12 * 1e3,
                rtol=1e-12, what="mfu=1 的理论下限")
