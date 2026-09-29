import numpy as np


def pack(docs, seq_len, eos):
    stream = [t for d in docs for t in d]              # 忘了在文档之间加 eos
    return [stream[i:i + seq_len + 1] for i in range(0, len(stream) - seq_len, seq_len)]


def positions(tokens, eos):
    return list(range(len(tokens)))                    # 没有在 eos 之后归零


def cu_seqlens(tokens, eos):
    pass


def doc_mask(tokens, eos):
    L = len(tokens)
    return np.tril(np.ones((L, L), dtype=bool))       # 只有因果，没有文档边界
