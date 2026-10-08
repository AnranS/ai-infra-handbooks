# 估算：Tensor Core 满速时，每个 SM 每周期要从共享内存读多少字节的 A、B 操作数（16 位），
# 以及一块累加器（FP32）有多大。共享内存按每 SM 每周期 128 字节（32 个 bank × 4 字节）算
SMEM = 128
cases = [
    # 方案, 每 SM 每周期乘加数, 一个 SM 算的 M, N（每个 K 步）, 这个 SM 从自己的共享内存读的 A 行数、B 列数
    ("Ampere mma.sync（每个 warp 算 64x64）", 1024, 64, 64, 64, 64),
    ("Hopper wgmma m64n64k16", 2048, 64, 64, 64, 64),
    ("Hopper wgmma m64n256k16", 2048, 64, 256, 64, 256),
    ("Blackwell tcgen05 m128n256k16", 4096, 128, 256, 128, 256),
    ("Blackwell 双 SM m256n256k16", 4096, 128, 256, 128, 128),   # 每个 SM 只放一半的 B
]
print("字节/乘加  共享内存 B/周期  占带宽  累加器  方案")
for name, fma, m, n, a_rows, b_cols in cases:
    per_fma = 2 * (a_rows + b_cols) / (m * n)       # 每个 K 步读 (a_rows + b_cols) × 2 字节，做 m × n 次乘加
    need = fma * per_fma
    print(f"{per_fma:8.4f} {need:12.0f} {need / SMEM:11.0%} {m * n * 4 // 1024:5d} KB  {name}")
