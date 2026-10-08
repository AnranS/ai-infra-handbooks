# decode 要多大的 batch 才能从"带宽受限"变成"算力受限"？
# 一步里每个参数：读 w 字节，做 2 × batch 次运算（batch 个 token 共用这一份权重）
# 所以算术强度 = 2 × batch / w（FLOP/字节），它要达到屋脊点 算力÷带宽
gens = [("A100", 312, 2039), ("H100", 989, 3350), ("H100 FP8", 1979, 3350),
        ("B200", 2250, 8000), ("B200 FP4", 9000, 8000)]
weights = [("BF16 权重", 2), ("FP8 权重", 1), ("4 比特权重", 0.5)]
print("GPU       屋脊点   " + "  ".join(f"{n}" for n, _ in weights))
for name, tf, bw in gens:
    ridge = tf * 1e12 / (bw * 1e9)
    row = "".join(f"{ridge * w / 2:11.0f}" for _, w in weights)
    print(f"{name:9s} {ridge:6.0f}{row}")
print()
print("权重之外，decode 还要读 KV。KV 是每个请求自己的，加大 batch 不会让它被复用：")
print("每个 KV 元素被组里的 g 个查询头各用一次乘加，所以注意力部分的算术强度 = g（BF16 KV，FLOP/字节）")
print("每组查询头数  算术强度  和 H100 BF16 屋脊点 295 相比  注意力方案")
for name, g in [("MHA（每个头一份 KV）", 1), ("GQA 8 组", 8), ("MLA（吸收后，128 个头共用）", 128)]:
    print(f"{g:11d} {g:9d} {295 / g:20.0f} 倍之差  {name}")
