# 注意力里每个分数 s = q·k 要做一次 exp，前后是两次矩阵乘：QK^T 和 PV，每个分数各 2d 次 FLOP。
# Tensor Core 满速时，每个 SM 每周期要做多少次 exp？特殊函数单元（SFU）每 SM 每周期只能做 16 次
cases = [
    # GPU/精度, Tensor Core 每 SM 每周期 FLOP, SFU 每 SM 每周期 exp 次数
    ("H100 BF16", 4096, 16),
    ("H100 FP8", 8192, 16),
    ("B200 BF16", 8192, 16),
    ("B200 BF16，SFU 翻倍", 8192, 32),
]
dims = [(64, 64), (128, 128), (576, 512)]            # (q·k 的维度, v 的维度)；最后一个是 MLA 的 decode
print("SFU 要达到的利用率（超过 100% 就是 exp 拖了 Tensor Core 的后腿）")
print("  d=64  d=128    MLA  GPU 与精度")
for name, tc, sfu in cases:
    need = [tc / (2 * dqk + 2 * dv) / sfu for dqk, dv in dims]   # 每周期需要的 exp 次数 ÷ SFU 的能力
    print(f"{need[0]:>6.0%}{need[1]:>7.0%}{need[2]:>7.0%}  {name}")
