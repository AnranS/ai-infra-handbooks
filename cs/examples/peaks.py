# 峰值 = SM 数 × 每个 SM 每周期的运算量 × 频率。用规格表里的峰值反推频率，检验这个拆法
gpus = [
    # 名称, SM 数, 每 SM 的 FP32 单元, 规格 FP32 TFLOPS, 每 SM 每周期 Tensor Core 稠密 FP16 乘加, 规格 FP16/BF16 TFLOPS
    ("V100", 80, 64, 15.7, 512, 125),
    ("A100", 108, 64, 19.5, 1024, 312),
    ("H100 SXM", 132, 128, 67, 2048, 989.4),
    ("B200", 148, None, None, 4096, 2250),
]
print("GPU       SM数 FP32单元/SM 反推频率  Tensor FLOP/周期/SM 反推频率")
for name, sm, lanes, fp32, fma, tc in gpus:
    f_fp32 = f"{fp32 * 1e12 / (sm * lanes * 2) / 1e9:.2f} GHz" if fp32 else "-"
    f_tc = f"{tc * 1e12 / (sm * fma * 2) / 1e9:.2f} GHz"
    print(f"{name:9s} {sm:4d} {lanes or '-':>11} {f_fp32:>9} {fma * 2:>20} {f_tc:>9}")
