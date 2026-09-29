class GuardCache:
    def __init__(self, cache_limit=8, mark_dynamic=()):
        self.cache_limit = cache_limit
        self.marked = set(mark_dynamic)
        self.auto = set()                               # 自动标成动态的 (输入, 维)
        self.first = None                               # 第一次编译时的形状
        self.entries = []

    @staticmethod
    def _match(guard, shapes):
        if len(guard) != len(shapes):
            return False
        for g, s in zip(guard, shapes):
            if len(g) != len(s):
                return False
            for gd, sd in zip(g, s):
                if gd == "dyn":
                    if sd < 2:                          # 动态维度要求大小 ≥ 2：0 和 1 是特化的
                        return False
                elif gd != sd:
                    return False
        return True

    def call(self, shapes):
        shapes = tuple(tuple(s) for s in shapes)
        for guard in reversed(self.entries):            # 从新到旧检查
            if self._match(guard, shapes):
                return "hit"
        if len(self.entries) >= self.cache_limit:
            return "eager"
        if self.first is None:
            self.first = shapes
        else:                                           # 和第一次编译时比，变了的维度从此都是动态
            for i, (s, f) in enumerate(zip(shapes, self.first)):
                if len(s) == len(f):
                    self.auto |= {(i, d) for d, (a, b) in enumerate(zip(s, f)) if a != b}
        dyn = self.marked | self.auto
        guard = tuple(tuple("dyn" if (i, d) in dyn and size >= 2 else size for d, size in enumerate(s))
                      for i, s in enumerate(shapes))
        self.entries.append(guard)
        return "compile"

    def num_compiles(self):
        return len(self.entries)

    def guards(self):
        return list(self.entries)
