def prefix_hits(requests: list[list[int]]) -> list[int]:
    root: dict = {}
    result = []
    for req in requests:
        node, depth = root, 0
        for tok in req:                 # 能走多深就是命中长度
            nxt = node.get(tok)
            if nxt is None:
                break
            node, depth = nxt, depth + 1
        result.append(depth)
        for tok in req[depth:]:         # 把剩下的部分插进字典树
            node = node.setdefault(tok, {})
    return result
