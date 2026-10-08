from layout_algebra import zipped_divide
from layout_core import Layout

tile = Layout((16, 8))                               # 16×8 的 fp32 块，列优先：同一列的 16 个元素相邻
print("tile =", tile)


def blocked(val_shape):
    """每个线程拿一个 val_shape 的小块：按小块划分，"块号"那一维就是线程号"""
    d = zipped_divide(tile, tuple(Layout(n) for n in val_shape))
    print(f"每个线程一个 {val_shape[0]}×{val_shape[1]} 的小块：", d)
    return lambda t, v: d((v, t))


def local_partition(thr):
    """CuTe 的 local_partition：按线程布局的形状划分，线程在块里的坐标固定"块内"那一维，剩下的"块号"是它的各个值"""
    d = zipped_divide(tile, tuple(Layout(n) for n in thr.shape))
    print(f"按线程布局 {thr} 划分：", d)
    pos = [thr(j) for j in range(thr.size())]
    return lambda t, v: d((pos.index(t), v))


def request(part, values):
    """一条访存指令：32 个线程各读自己编号为 values 的值，数一数落在几个 32 字节扇区里"""
    addr = sorted(part(t, v) for t in range(32) for v in values)
    sectors = len({a * 4 // 32 for a in addr})
    print(f"  读编号 {values} 的值：地址 {addr[:6]}…，{sectors} 个扇区，利用率 {len(addr) * 4 / (sectors * 32):.0%}")


p = blocked((2, 2))
print("  线程 0、1、8 的值：", [[p(t, v) for v in range(4)] for t in (0, 1, 8)])
request(p, [0])                                      # 每个线程读 4 字节
request(p, [0, 1])                                   # 前两个值在内存里相邻：一条 8 字节的向量指令读完
q = local_partition(Layout((16, 2)))
print("  线程 0、1、16 的值：", [[q(t, v) for v in range(4)] for t in (0, 1, 16)])
request(q, [0])
