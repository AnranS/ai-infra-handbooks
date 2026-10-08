# Little 定律：要维持带宽 B，在途（已发出、还没返回）的数据量 = B × 延迟。访存延迟按 600 ns 估算（量级）
LATENCY = 600e-9
gpus = [("A100", 2039e9, 108), ("H100 SXM", 3350e9, 132), ("B200", 8000e9, 148)]
print("GPU        整卡在途   每个 SM   每个线程（每 SM 2048 个线程全满）")
for name, bw, sms in gpus:
    total = bw * LATENCY
    per_sm = total / sms
    print(f"{name:9s} {total / 1e6:6.2f} MB {per_sm / 1024:7.1f} KB {per_sm / 2048:9.1f} 字节")
