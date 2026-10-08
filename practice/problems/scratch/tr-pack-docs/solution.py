import numpy as np


def pack(docs, seq_len, eos):
    stream = []
    for d in docs:
        stream += list(d) + [eos]                      # 每篇文档后面接一个 eos
    n = (len(stream) - 1) // seq_len                   # 能切出多少个完整的 seq_len + 1
    return [stream[i * seq_len:i * seq_len + seq_len + 1] for i in range(n)]


def positions(tokens, eos):
    out, p = [], 0
    for t in tokens:
        out.append(p)
        p = 0 if t == eos else p + 1                   # eos 之后的 token 重新从 0 开始
    return out


def cu_seqlens(tokens, eos):
    bounds = [0] + [i + 1 for i, t in enumerate(tokens) if t == eos]
    if bounds[-1] != len(tokens):
        bounds.append(len(tokens))
    return bounds


def doc_mask(tokens, eos):
    L = len(tokens)
    seg = np.zeros(L, dtype=int)
    for s, (a, b) in enumerate(zip(cu_seqlens(tokens, eos), cu_seqlens(tokens, eos)[1:])):
        seg[a:b] = s
    return np.tril(np.ones((L, L), dtype=bool)) & (seg[:, None] == seg[None, :])
