from collections import OrderedDict


class Cache:
    """组相联缓存，组内按 LRU 替换。size、line 以字节计。"""

    def __init__(self, size, ways, line=64):
        self.line, self.ways = line, ways
        self.nsets = size // (ways * line)
        self.sets = [OrderedDict() for _ in range(self.nsets)]
        self.hits = self.misses = 0

    def access(self, addr):
        tag = addr // self.line                     # 第几条缓存行
        s = self.sets[tag % self.nsets]             # 行号取模决定放进哪一组
        if tag in s:
            s.move_to_end(tag)
            self.hits += 1
        else:
            self.misses += 1
            if len(s) == self.ways:
                s.popitem(last=False)               # 组满了：踢掉最久没用的
            s[tag] = True


def walk(rows, cols, row_bytes, by_column):
    c = Cache(48 * 1024, ways=12)                   # 和本机 L1d 一样：48 KB、12 路、64 组
    order = ((r, k) for k in range(cols) for r in range(rows)) if by_column else \
            ((r, k) for r in range(rows) for k in range(cols))
    for r, k in order:
        c.access(r * row_bytes + k * 4)             # float 矩阵
    return c.hits / (c.hits + c.misses)


print(f"512x512 按行遍历：        命中率 {walk(512, 512, 512 * 4, False):.1%}")
print(f"512x512 按列遍历：        命中率 {walk(512, 512, 512 * 4, True):.1%}")
print(f"每行补 16 个 float 再按列：命中率 {walk(512, 512, 528 * 4, True):.1%}")
