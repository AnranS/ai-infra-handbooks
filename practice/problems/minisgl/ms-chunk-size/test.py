from checker import check, check_close
from solution import best_chunk, simulate


def test_example():
    ttft, gap = simulate(L=8000, C=2048, n_decode=64, a=5.0, b=0.01)
    blocks = [2048, 2048, 2048, 1856]
    want_ttft = sum(5 + 0.01 * c for c in blocks) + 3 * (5 + 0.64)
    check_close(ttft, want_ttft, what="TTFT")
    check_close(gap, 5 + 20.48 + 5 + 0.64, what="decode 的最长间隔")


def test_no_chunking():
    ttft, gap = simulate(1000, 1000, 10, 2.0, 0.1)
    check_close((ttft, gap), (102.0, 102.0 + 3.0), what="块大小 >= L：一次做完")


def test_best_chunk():
    cands = [256, 512, 1024, 2048, 4096, 8192]
    c = best_chunk(8000, 64, 5.0, 0.01, tpot_slo=40.0, candidates=cands)
    check(c, 2048, "SLO 40ms 下最好的块大小")
    check(best_chunk(8000, 64, 5.0, 0.01, tpot_slo=200.0, candidates=cands), 8192, "SLO 很宽：一次做完 TTFT 最小")
    check(best_chunk(8000, 64, 5.0, 0.01, tpot_slo=10.0, candidates=cands), None, "SLO 太严：都不满足")


def test_tradeoff_monotone():
    prev_ttft, prev_gap = None, None
    for C in [128, 256, 512, 1024, 2048]:
        ttft, gap = simulate(4096, C, 32, 4.0, 0.02)
        if prev_ttft is not None:
            assert ttft < prev_ttft and gap > prev_gap, "块越大：TTFT 越小、decode 间隔越大"
        prev_ttft, prev_gap = ttft, gap
