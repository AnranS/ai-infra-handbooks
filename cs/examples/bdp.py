# 带宽延迟积：一条 TCP 连接的吞吐 ≈ 窗口大小 ÷ RTT。窗口不够大时，带宽再高也跑不满
def throughput_gbps(window_kb, rtt_ms):
    return window_kb * 1024 * 8 / (rtt_ms * 1e-3) / 1e9


print("窗口 \\ RTT" + "".join(f"{r:>10} ms" for r in (0.1, 1, 10, 50)))
for kb in (64, 256, 1024, 4096):
    row = "".join(f"{throughput_gbps(kb, r):10.2f} Gb/s" for r in (0.1, 1, 10, 50))
    print(f"{kb:6d} KB" + row)
print()
print("要跑满一条链路，窗口至少要有带宽 × RTT：")
for gbps, rtt in [(10, 0.1), (10, 1), (25, 10), (100, 50)]:
    need = gbps * 1e9 / 8 * (rtt * 1e-3)
    print(f"  {gbps:3d} Gb/s、RTT {rtt:4.1f} ms：{need / 1024:8.0f} KB")
print()
print("慢启动：初始窗口 10 个 MSS（约 14 KB），每个 RTT 翻倍，要几个 RTT 才能发完一个响应？")
for kb in (14, 100, 1024, 10240):
    sent, win, rtts = 0, 14, 0
    while sent < kb:
        sent += win
        win *= 2
        rtts += 1
    print(f"  {kb:6d} KB 的响应：{rtts} 个 RTT")
