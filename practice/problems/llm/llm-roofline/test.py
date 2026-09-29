from checker import check, check_close
from solution import bound, decode_step_ms, gemm_intensity, roofline_time

H100 = (989, 3350)


def test_example():
    check(bound(4096, 4096, 4096, *H100), "compute", "4096³ GEMM")
    check(bound(1, 4096, 4096, *H100), "memory", "batch=1 的 GEMV")
    check_close(decode_step_ms(7e9, 1, *H100), 14e9 / 3350e9 * 1e3, rtol=1e-9, what="7B 模型 decode 一步")


def test_intensity():
    check_close(gemm_intensity(4096, 4096, 4096), 2 * 4096 ** 3 / (3 * 4096 ** 2 * 2), what="方阵的算术强度")
    check_close(gemm_intensity(1, 4096, 4096), 2 * 4096 * 4096 / ((4096 + 4096 * 4096 + 4096) * 2), what="GEMV")
    check_close(gemm_intensity(128, 128, 128, bytes_per_elem=4), 2 * 128 ** 3 / (3 * 128 ** 2 * 4), what="fp32")


def test_roofline_time_units():
    check_close(roofline_time(1e12, 0, 1, 1), 1.0, what="1 TFLOP / 1 TFLOPS")
    check_close(roofline_time(0, 1e9, 1, 1), 1.0, what="1 GB / 1 GB/s")
    check_close(roofline_time(2e12, 5e9, 1, 1), 5.0, what="取两者的最大值")


def test_ridge_point():
    ridge = 989e12 / 3350e9
    M = 1
    while gemm_intensity(M, 8192, 8192) < ridge:
        check(bound(M, 8192, 8192, *H100), "memory", f"M={M}")
        M += 1
    check(bound(M, 8192, 8192, *H100), "compute", f"M={M}（越过屋脊点）")
    assert 250 < M < 400, f"8192×8192 权重的 GEMM 在 M≈300 时越过屋脊点，算出来是 M={M}"


def test_decode_batching():
    t1 = decode_step_ms(7e9, 1, *H100)
    t64 = decode_step_ms(7e9, 64, *H100)
    check_close(t64, t1, rtol=1e-9, what="batch=64 仍然受限于带宽，时间不变")
    t1024 = decode_step_ms(7e9, 1024, *H100)
    check_close(t1024, 2 * 7e9 * 1024 / 989e12 * 1e3, rtol=1e-9, what="batch=1024 变成算力瓶颈")
