class GuardCache:
    def __init__(self, cache_limit=8, mark_dynamic=()):
        self.entries = []

    def call(self, shapes):
        if shapes in self.entries:                     # 只会按具体形状缓存
            return "hit"
        self.entries.append(shapes)
        return "compile"

    def num_compiles(self):
        return len(self.entries)

    def guards(self):
        return list(self.entries)
