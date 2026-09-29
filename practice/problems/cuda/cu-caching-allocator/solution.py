MB = 1 << 20
MIN_BLOCK, SMALL_SIZE, SMALL_BUFFER = 512, 1 * MB, 2 * MB
LARGE_BUFFER, MIN_LARGE_ALLOC, ROUND_LARGE = 20 * MB, 10 * MB, 2 * MB


class Block:
    def __init__(self, addr, size, stream, small, segment):
        self.addr, self.size, self.stream, self.small, self.segment = addr, size, stream, small, segment
        self.allocated = False
        self.prev = self.next = None                  # 同一段里按地址相邻的块


class Segment:
    def __init__(self, base, size, stream, small):
        self.base, self.size, self.stream, self.small = base, size, stream, small
        self.head = Block(base, size, stream, small, self)


class CachingAllocator:
    def __init__(self, capacity):
        self.capacity = capacity
        self.next_addr = 0
        self.segments = []
        self.free_blocks = set()
        self.allocated = 0

    @staticmethod
    def _round(size):
        return max(MIN_BLOCK, -(-size // MIN_BLOCK) * MIN_BLOCK)

    @staticmethod
    def _segment_size(size):
        if size <= SMALL_SIZE:
            return SMALL_BUFFER
        if size < MIN_LARGE_ALLOC:
            return LARGE_BUFFER
        return -(-size // ROUND_LARGE) * ROUND_LARGE

    def _best_fit(self, size, stream, small):
        cands = [b for b in self.free_blocks if b.small == small and b.stream == stream and b.size >= size]
        return min(cands, key=lambda b: (b.size, b.addr)) if cands else None

    def malloc(self, size, stream=0):
        size = self._round(size)
        small = size <= SMALL_SIZE
        block = self._best_fit(size, stream, small)
        if block is None:
            seg_size = self._segment_size(size)
            if self.memory_reserved() + seg_size > self.capacity:
                self.empty_cache()                    # 先把缓存着的整段还给驱动，再试一次
                if self.memory_reserved() + seg_size > self.capacity:
                    raise MemoryError(f"申请 {size} 字节失败：已分配 {self.allocated}，已保留 {self.memory_reserved()}，"
                                      f"上限 {self.capacity}")
            seg = Segment(self.next_addr, seg_size, stream, small)
            self.next_addr += seg_size
            self.segments.append(seg)
            block = seg.head
        else:
            self.free_blocks.remove(block)
        rest = block.size - size
        if (rest >= MIN_BLOCK) if small else (rest > SMALL_SIZE):      # 切下剩余部分，成为地址更高的空闲块
            tail = Block(block.addr + size, rest, stream, small, block.segment)
            tail.prev, tail.next = block, block.next
            if block.next:
                block.next.prev = tail
            block.next = tail
            block.size = size
            self.free_blocks.add(tail)
        block.allocated = True
        self.allocated += block.size
        return block

    def free(self, block):
        if not getattr(block, "allocated", False):
            raise ValueError("释放了一个没有分配的块（重复释放？）")
        block.allocated = False
        self.allocated -= block.size
        if block.prev and not block.prev.allocated:   # 和前一个空闲块合并：前一个块吸收这一块
            p = block.prev
            self.free_blocks.remove(p)
            p.size += block.size
            p.next = block.next
            if block.next:
                block.next.prev = p
            block = p
        if block.next and not block.next.allocated:   # 和后一个空闲块合并
            n = block.next
            self.free_blocks.remove(n)
            block.size += n.size
            block.next = n.next
            if n.next:
                n.next.prev = block
        self.free_blocks.add(block)

    def empty_cache(self):
        keep = []
        for seg in self.segments:
            h = seg.head
            if not h.allocated and h.size == seg.size:  # 整段空闲：还给驱动
                self.free_blocks.remove(h)
            else:
                keep.append(seg)
        self.segments = keep

    def memory_allocated(self):
        return self.allocated

    def memory_reserved(self):
        return sum(s.size for s in self.segments)

    def snapshot(self):
        out = []
        for seg in sorted(self.segments, key=lambda s: s.base):
            blocks, b = [], seg.head
            while b:
                blocks.append((b.addr, b.size, b.allocated))
                b = b.next
            out.append((seg.base, seg.size, seg.stream, blocks))
        return out
