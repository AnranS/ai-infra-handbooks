from collections import OrderedDict


def cache_stats(addrs, size, ways, line=64):
    nsets = size // (ways * line)
    sets = [OrderedDict() for _ in range(nsets)]
    hits = misses = 0
    for a in addrs:
        tag = a // line
        s = sets[tag % nsets]
        if tag % nsets in s:                            # 用组号当键：同一组里的不同行被当成了同一行
            s.move_to_end(tag % nsets)
            hits += 1
        else:
            misses += 1
            if len(s) == ways:
                s.popitem(last=False)
            s[tag % nsets] = True
    return hits, misses


def pad_floats(rows, cols, size, ways, line=64):
    nsets = size // (ways * line)
    step = line // 4
    for pad in range(0, nsets * step + 1, step):
        if ((cols + pad) * 4 // line) % nsets:          # 只看步长不为 0，没数真正覆盖了几个组
            return pad
    return 0
