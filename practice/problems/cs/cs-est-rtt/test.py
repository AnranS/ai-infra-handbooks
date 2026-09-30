from checker import check, check_close
from solution import first_byte_ms, slow_start_rtts, stream_total_ms, throughput_mbps, window_for_mbps


def test_example():
    check_close(first_byte_ms(30, 150), 240.0, rtol=1e-9, what="新建连接")
    check_close(first_byte_ms(30, 150, reuse=True), 180.0, rtol=1e-9, what="复用连接")
    check_close(throughput_mbps(256, 30), 8.74, rtol=0.01, what="256 KB 窗口、30 ms RTT")
    check(window_for_mbps(1000, 1), 977, "1 GB/s、RTT 1 ms 需要的窗口")
    check(slow_start_rtts(1024), 7, "1 MB 的响应要 7 个 RTT")


def test_no_tls():
    check_close(first_byte_ms(30, 150, tls=False), 210.0, rtol=1e-9, what="没有 TLS 少一个 RTT")
    check_close(first_byte_ms(0.5, 200, reuse=True), 200.5, rtol=1e-9, what="同机房复用连接")


def test_stream_total():
    check_close(stream_total_ms(30, 150, 200, 20, reuse=True), 180 + 199 * 20, rtol=1e-9,
                what="首字之后还有 199 个 token")
    check_close(stream_total_ms(30, 150, 1, 20, reuse=True), 180.0, rtol=1e-9, what="只有一个 token")
    reuse = stream_total_ms(30, 150, 200, 20, reuse=True)
    fresh = stream_total_ms(30, 150, 200, 20)
    check_close(fresh - reuse, 60.0, rtol=1e-9, what="握手的代价只有 60 ms，对总时长影响小")


def test_bdp():
    check_close(throughput_mbps(64, 0.1), 655.36, rtol=0.01, what="同机房、64 KB 窗口")
    check(window_for_mbps(throughput_mbps(256, 30), 30) in (256, 257), True, "两个函数互逆")
    check(window_for_mbps(3125, 10) > 30000, True, "25 Gb/s、RTT 10 ms 需要 30 MB 以上的窗口")


def test_slow_start():
    check([slow_start_rtts(kb) for kb in (14, 100, 1024, 10240)], [1, 4, 7, 10], "每个 RTT 翻倍")
    check(slow_start_rtts(1, 14), 1, "一个窗口就发完了")
    check(slow_start_rtts(1024, 64) < slow_start_rtts(1024, 14), True, "初始窗口越大越快")
