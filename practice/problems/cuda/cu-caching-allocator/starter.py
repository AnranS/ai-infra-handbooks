class Block:
    def __init__(self, addr, size, stream):
        self.addr, self.size, self.stream = addr, size, stream


class CachingAllocator:
    """不缓存的分配器：每次都向驱动要一段，释放就还回去（也不取整）"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.next_addr = 0
        self.live = {}

    def malloc(self, size, stream=0):
        if sum(b.size for b in self.live.values()) + size > self.capacity:
            raise MemoryError("out of memory")
        b = Block(self.next_addr, size, stream)
        self.next_addr += size
        self.live[b.addr] = b
        return b

    def free(self, block):
        del self.live[block.addr]

    def empty_cache(self):
        pass

    def memory_allocated(self):
        return sum(b.size for b in self.live.values())

    def memory_reserved(self):
        return self.memory_allocated()

    def snapshot(self):
        return [(b.addr, b.size, b.stream, [(b.addr, b.size, True)]) for b in sorted(self.live.values(), key=lambda b: b.addr)]
