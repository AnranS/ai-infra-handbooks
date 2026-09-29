from checker import check_close
from solution import exposed_ms, kv_transfer_ms, ttft_ms


def test_example():
    t = kv_transfer_ms(8192, 327680, 400, n_links=8)
    check_close(t, 8192 * 327680 / (8 * 50e9 * 0.8) * 1e3, rtol=1e-12, what="8 × 400G")
    check_close(t, 8.39, rtol=0.01, what="约 8.4 ms")
    check_close(exposed_ms(314.7, 8.39, 80), 8.39 / 80, rtol=1e-9, what="传得快：只暴露最后一层")
    check_close(exposed_ms(314.7, 1074, 80), 314.7 / 80 + 1074 - 314.7, rtol=1e-9, what="传得慢")


def test_boundary():
    check_close(exposed_ms(100, 100, 10), 10, rtol=1e-9, what="每层计算和传输一样快")
    check_close(exposed_ms(100, 0, 10), 0, atol=1e-12, what="不传输")
    assert exposed_ms(100, 50, 10) < exposed_ms(100, 150, 10) < 150, "传输变慢，暴露时间变长，但仍小于总传输时间"


def test_ttft():
    check_close(ttft_ms(314.7, 8.39, 80, layerwise=False), 314.7 + 8.39, what="不逐层传输")
    check_close(ttft_ms(314.7, 8.39, 80), 314.7 + 8.39 / 80, what="逐层传输")
    mla = kv_transfer_ms(8192, 70272, 400, n_links=8)
    check_close(kv_transfer_ms(8192, 327680, 400, n_links=8) / mla, 327680 / 70272, what="MLA 的 KV 少得多")
    check_close(kv_transfer_ms(1000, 1e6, 100, efficiency=1.0), 80.0, what="1 GB 走 100G 链路")
