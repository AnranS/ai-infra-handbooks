# 每一代数据中心卡（SXM 形态）的算力、带宽各涨了多少，屋脊点（算力÷带宽）怎么变。
# 屋脊点的含义：每从显存读 1 字节，要做多少次运算才能把算力喂饱
gens = [
    # 名称, 年份, FP16/BF16 稠密 TFLOPS, 该代最低精度的稠密 TFLOPS, 精度名, 带宽 GB/s, 显存 GB
    ("V100", 2017, 125, 125, "FP16", 900, 32),
    ("A100", 2020, 312, 312, "BF16", 2039, 80),
    ("H100", 2022, 989, 1979, "FP8", 3350, 80),
    ("B200", 2024, 2250, 9000, "FP4", 8000, 180),
]
print("GPU   年份  BF16算力 较上代    带宽   较上代  BF16屋脊点   最低精度算力  屋脊点")
prev = None
for name, year, bf16, low, dtype, bw, mem in gens:
    up = (f"{bf16 / prev[0]:5.1f}x", f"{bw / prev[1]:5.1f}x") if prev else ("    -", "    -")
    print(f"{name:5s} {year} {bf16:7.0f} {up[0]} {bw / 1000:6.2f} TB/s {up[1]} "
          f"{bf16 * 1e12 / (bw * 1e9):9.0f} {low:11.0f} {dtype:4s} {low * 1e12 / (bw * 1e9):7.0f}")
    prev = (bf16, bw)
print()
print("同一个 8B 模型（BF16 权重 16 GB，KV 每 token 128 KB）在各代上的理论下限：")
print("GPU   decode 一步(batch=1)  prefill 4K(BF16，算力用满一半)  能放的 KV")
for name, year, bf16, low, dtype, bw, mem in gens:
    decode = 16e9 / (bw * 1e9) * 1e3                       # 读一遍权重
    prefill = 2 * 8e9 * 4096 / (bf16 * 1e12 * 0.5) * 1e3
    kv = (mem * 1e9 - 16e9) / (128 * 1024) / 1e3
    print(f"{name:5s} {decode:12.1f} ms {prefill:26.0f} ms {kv:9.0f}K token")
