from collections import OrderedDict


def split_va(addr):
    offset = addr & 0xFFF
    return ((addr >> 39) & 0x1FF, (addr >> 30) & 0x1FF, (addr >> 21) & 0x1FF, (addr >> 12) & 0x1FF, offset)


def tlb_stats(addrs, page_size, entries):
    tlb = OrderedDict()
    hits = misses = 0
    for a in addrs:
        vpn = a // page_size
        if vpn in tlb:
            hits += 1
            tlb.move_to_end(vpn)                 # 最近用过，放到队尾
            continue
        misses += 1
        if len(tlb) == entries:
            tlb.popitem(last=False)              # 淘汰最久没用过的
        tlb[vpn] = True
    return hits, misses
