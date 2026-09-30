def partition_labels(s):
    out = []
    start = 0
    for i, ch in enumerate(s):
        if ch not in s[i + 1:]:                # 这个字母以后不再出现就切：切得太碎
            out.append(i - start + 1)
            start = i + 1
    return out


def merge_ranges(pairs):
    out = []
    for start, end in sorted(pairs):
        if out and start < out[-1][1]:         # 用 <：端点相接的没有合并
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
