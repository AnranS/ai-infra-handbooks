import torch

print("各格式能表示的最小正规数与机器精度：")
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"  {str(dt):<15} 最大 {fi.max:.3g}，最小正规数 {fi.tiny:.3g}，eps {fi.eps:.3g}")

g = torch.tensor([1e-8, 1e-6, 2e-5])                     # 很小的梯度（深层网络里很常见）
print("fp16 直接存：", g.half().tolist())
scale = 2.0 ** 16
print("先乘 2^16 再存 fp16、用时再除回：", (g * scale).half().float().div(scale).tolist())
print("bf16 直接存：", g.bfloat16().float().tolist())

w16, w32 = torch.tensor(1.0, dtype=torch.bfloat16), torch.tensor(1.0)
for _ in range(1000):                                    # 学习率 × 梯度 = 1e-3 的小更新，做 1000 次
    w16 += 1e-3
    w32 += 1e-3
print(f"bf16 权重直接更新 1000 次：{w16.item():.4f}；fp32 主权重：{w32.item():.4f}")
