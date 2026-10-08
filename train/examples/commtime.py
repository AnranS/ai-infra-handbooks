import math

ALPHA = 5e-6            # 每一步通信的固定开销（握手、同步），秒
BETA = 200e9            # 单向链路带宽，字节 / 秒（按 NVLink 一个方向 200 GB/s 估）


def ring(nbytes, n):    # 环形 all-reduce：2(n-1) 步，每步只发 1/n
    return 2 * (n - 1) * (ALPHA + nbytes / n / BETA)


def tree(nbytes, n):    # 树形：2*log2(n) 步，每步发整份
    return 2 * math.ceil(math.log2(n)) * (ALPHA + nbytes / BETA)


print("一次 all-reduce 要多久（α-β 模型，α = 5 μs，β = 200 GB/s）")
print(f"{'卡数':>4} {'消息大小':>10} {'环形':>12} {'树形':>12}  更快的")
for n in (4, 8, 16, 64):
    for name, nbytes in (("4 KB", 4 << 10), ("4 MB", 4 << 20), ("1 GB", 1 << 30)):
        t_ring, t_tree = ring(nbytes, n), tree(nbytes, n)
        print(f"{n:>4} {name:>10} {t_ring * 1e6:>9.1f} μs {t_tree * 1e6:>9.1f} μs  "
              f"{'环形' if t_ring < t_tree else '树形'}")

print()
print("通信占一步训练的多少（7B 模型，bf16 梯度 14 GB，一步前反向按 300 ms 算）")
grad = 14 * (1 << 30)
for n in (8, 16, 64, 256):
    t = ring(grad, n)
    print(f"  {n:>3} 卡数据并行：all-reduce {t * 1e3:>6.1f} ms，占一步的 {t / (0.3 + t):>5.1%}")
