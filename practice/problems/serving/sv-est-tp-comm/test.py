from checker import check_close
from solution import allreduce_ms, comm_share, tp_comm_ms


def test_example():
    prefill = tp_comm_ms(8192, 8192, 80, 8, 450)
    check_close(prefill, 160 * 1.75 * 8192 * 8192 * 2 / 450e9 * 1e3, rtol=1e-12, what="prefill 8K")
    check_close(prefill, 83.5, rtol=0.01, what="约 83 ms")
    decode = tp_comm_ms(64, 8192, 80, 8, 450, latency_us=2)
    bw_part = tp_comm_ms(64, 8192, 80, 8, 450)
    check_close(bw_part, 0.652, rtol=0.01, what="decode 的带宽项")
    check_close(decode - bw_part, 160 * 14 * 2 / 1e3, rtol=1e-9, what="decode 的延迟项：160 次 × 14 步 × 2 微秒")


def test_allreduce():
    check_close(allreduce_ms(1e9, 1, 100), 0.0, atol=1e-12, what="单卡不需要通信")
    check_close(allreduce_ms(1e9, 2, 100), 10.0, what="2 卡：发送 1 倍数据")
    check_close(allreduce_ms(1e9, 8, 100), 17.5, what="8 卡：发送 1.75 倍数据")
    check_close(allreduce_ms(0, 4, 100, latency_us=5), 0.03, what="只有延迟：6 步 × 5 微秒")


def test_pcie_vs_nvlink():
    ratio = tp_comm_ms(8192, 8192, 80, 8, 25) / tp_comm_ms(8192, 8192, 80, 8, 450)
    check_close(ratio, 18, what="PCIe 比 NVLink 慢 18 倍")
    check_close(comm_share(83.5, 290), 83.5 / 373.5, what="通信占比")
