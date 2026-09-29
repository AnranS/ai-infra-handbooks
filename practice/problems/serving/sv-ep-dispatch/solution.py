import numpy as np


def ep_moe(x, topk_ids, topk_w, experts, n):
    E = len(experts)
    per = E // n
    send_counts = [[0] * n for _ in range(n)]
    inbox = [[] for _ in range(n)]            # (src_rank, token, slot, expert, 向量)
    for r in range(n):
        pairs = []
        for i in range(len(x[r])):
            for j, e in enumerate(topk_ids[r][i]):
                pairs.append((int(e) // per, i, j, int(e)))
        pairs.sort()
        for dst, i, j, e in pairs:
            send_counts[r][dst] += 1
            inbox[dst].append((r, i, j, e, x[r][i]))
    results = {}                               # (src_rank, token, slot) -> 专家输出
    for s in range(n):
        items = inbox[s]
        if not items:
            continue
        ex = np.array([it[3] for it in items])
        vecs = np.stack([it[4] for it in items])
        for e in range(s * per, (s + 1) * per):
            rows = np.nonzero(ex == e)[0]
            if len(rows):
                ys = experts[e](vecs[rows])
                for row, y in zip(rows, ys):
                    src, i, j = items[row][:3]
                    results[src, i, j] = y
    outputs = []
    for r in range(n):
        out = np.zeros_like(x[r], dtype=np.float64)
        for i in range(len(x[r])):
            for j in range(len(topk_ids[r][i])):
                out[i] += topk_w[r][i][j] * results[r, i, j]
        outputs.append(out)
    return outputs, send_counts
