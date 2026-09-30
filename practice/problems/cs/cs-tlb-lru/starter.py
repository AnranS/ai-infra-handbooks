from collections import deque


def split_va(addr):
    offset = addr & 0xFFF
    return ((addr >> 39) & 0x1FF, (addr >> 30) & 0x1FF, (addr >> 21) & 0x1FF, (addr >> 12) & 0x1FF, offset)


def tlb_stats(addrs, page_size, entries):
    tlb, order = set(), deque()             # 先进先出：最早放进来的先被淘汰
    hits = misses = 0
    for a in addrs:
        vpn = a // page_size
        if vpn in tlb:
            hits += 1
            continue
        misses += 1
        if len(tlb) == entries:
            tlb.discard(order.popleft())
        tlb.add(vpn)
        order.append(vpn)
    return hits, misses
