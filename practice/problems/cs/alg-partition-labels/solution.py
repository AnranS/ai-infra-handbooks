def partition_labels(s):
    last = {ch: i for i, ch in enumerate(s)}   # 每个字母最后出现的位置
    out = []
    start = end = 0
    for i, ch in enumerate(s):
        end = max(end, last[ch])
        if i == end:                           # 这一段里的字母都不会再出现了
            out.append(end - start + 1)
            start = i + 1
    return out


def merge_ranges(pairs):
    out = []
    for start, end in sorted(pairs):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
