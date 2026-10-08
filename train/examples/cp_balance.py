def work(query_chunks, n_chunks):
    """因果注意力里，一个 query 块需要和多少个 KV 块做计算（对角线上的块算半个）"""
    return sum(q + 0.5 for q in query_chunks)


P = 4
contiguous = {r: [2 * r, 2 * r + 1] for r in range(P)}          # 按顺序切：rank r 拿第 2r、2r+1 块（共 2P 块）
zigzag = {r: [r, 2 * P - 1 - r] for r in range(P)}               # 之字形：rank r 拿第 r 块和倒数第 r+1 块
for name, split in (("顺序切分", contiguous), ("之字形切分", zigzag)):
    print(name, [work(split[r], 2 * P) for r in range(P)])
