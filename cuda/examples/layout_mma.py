from layout_core import Layout

# CUTLASS 的 cute/atom/mma_traits_sm80.hpp：SM80_16x8x16_F32F16F16F32_TN（mma.sync.m16n8k16）的 TV 布局，
# 把 (线程号, 值编号) 映射到块内的列优先下标
A = Layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))    # A：16×16 的 M×K
B = Layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))            # B：8×16 的 N×K，下标 = n + 8k
C = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))            # C：16×8 的 M×N


def rc(idx):
    return idx % 16, idx // 16                       # 列优先下标 → (行, 列)


def ptx_a(lane, i):                                  # PTX 文档的表（Tensor Core 一章抄过）：g = lane / 4，t = lane % 4
    g, t = lane // 4, lane % 4
    return g + 8 * (i // 2 % 2), 2 * t + i % 2 + 8 * (i // 4)


def ptx_b(lane, i):                                  # B 按 (k, n) 给出
    g, t = lane // 4, lane % 4
    return 2 * t + i % 2 + 8 * (i // 2), g


def ptx_c(lane, i):
    g, t = lane // 4, lane % 4
    return g + 8 * (i // 2), 2 * t + i % 2


for lane in (0, 1, 4):
    print(f"lane {lane}：A 的 8 个值在", [rc(A((lane, i))) for i in range(8)])
print("A 与 PTX 的表一致：", all(rc(A((l, i))) == ptx_a(l, i) for l in range(32) for i in range(8)))
print("B 与 PTX 的表一致：", all((B((l, i)) // 8, B((l, i)) % 8) == ptx_b(l, i) for l in range(32) for i in range(4)))
print("C 与 PTX 的表一致：", all(rc(C((l, i))) == ptx_c(l, i) for l in range(32) for i in range(4)))
owner = {rc(C((l, i))): l for l in range(32) for i in range(4)}
print("C 的每个元素归哪个线程（行 0～3）：")
for r in range(4):
    print("  " + " ".join(f"{owner[(r, c)]:2d}" for c in range(8)))
