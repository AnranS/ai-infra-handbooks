from collections import OrderedDict


def cache_stats(addrs, size, ways, line=64):
    nsets = size // (ways * line)
    sets = [OrderedDict() for _ in range(nsets)]
    hits = misses = 0
    for a in addrs:
        tag = a // line
        s = sets[tag % nsets]
        if tag in s:
            s.move_to_end(tag)
            hits += 1
        else:
            misses += 1
            if len(s) == ways:
                s.popitem(last=False)
            s[tag] = True
    return hits, misses


def pad_floats(rows, cols, size, ways, line=64):
    nsets = size // (ways * line)
    want = min(rows, nsets)
    step = line // 4                                    # 一条缓存行有几个 float32
    for pad in range(0, nsets * step + 1, step):
        row_bytes = (cols + pad) * 4
        seen = {(r * row_bytes // line) % nsets for r in range(rows)}
        if len(seen) >= want:
            return pad
    return 0
