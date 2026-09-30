from collections import OrderedDict


class PageCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pages = OrderedDict()              # 页 -> 是否是脏页
        self.hits = self.disk_reads = self.disk_writes = 0

    def _insert(self, page, dirty):
        if len(self.pages) == self.capacity:
            old, old_dirty = self.pages.popitem(last=False)
            if old_dirty:
                self.disk_writes += 1
        self.pages[page] = dirty

    def read(self, page):
        if page in self.pages:
            self.hits += 1
            self.pages.move_to_end(page)
            return
        self.disk_reads += 1
        self._insert(page, False)

    def write(self, page):
        if page in self.pages:
            self.pages.move_to_end(page)
            self.pages[page] = True
        else:
            self._insert(page, True)

    def fsync(self):
        for page in self.pages:                 # 把缓存里的页都写一遍
            self.disk_writes += 1
            self.pages[page] = False

    def stats(self):
        return {"hits": self.hits, "disk_reads": self.disk_reads, "disk_writes": self.disk_writes,
                "dirty": sum(self.pages.values())}
