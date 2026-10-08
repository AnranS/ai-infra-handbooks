# 一张卡上跑多个模型实例：MIG（硬件切分）和 MPS（共享）各自的账
# MIG 把 SM、L2、显存控制器都按份切开；MPS 让多个进程的 kernel 同时上一张完整的卡
SM, L2, MEM, BW = 132, 50, 80, 3350        # H100：SM 数、L2（MB）、显存（GB）、带宽（GB/s）
print("MIG 切分（H100 的 7 种档位里挑 3 种）：")
for name, slices, sms, mem in [("1g.10gb", 1, 16, 10), ("2g.20gb", 2, 32, 20), ("3g.40gb", 3, 60, 40)]:
    print(f"  {name:8s} {sms:3d} 个 SM（{sms / SM:4.0%}） 显存 {mem} GB  带宽约 {BW * slices / 7 / 1000:.2f} TB/s  "
          f"一张卡能切 {7 // slices} 个")
print()
print("同一个 7B 模型（BF16 权重 14 GB，decode 读一遍权重）：")
for name, sms, mem, share in [("整卡", SM, MEM, 1.0), ("3g.40gb", 60, 40, 3 / 7), ("2g.20gb", 32, 20, 2 / 7)]:
    bw = BW * share
    rest = f"还剩 {mem - 14:.0f} GB 放 KV" if mem > 14 else "放不下权重"
    print(f"  {name:8s} 带宽 {bw / 1000:.2f} TB/s  decode 一步下限 {14 / bw * 1000:5.1f} ms  {rest}")
