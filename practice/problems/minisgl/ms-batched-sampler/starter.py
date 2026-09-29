import numpy as np


def sample_batch(logits, temperature, top_k, top_p, u):
    logits = np.asarray(logits, dtype=np.float64)
    out = np.argmax(logits, axis=1).astype(np.int64)      # 贪心的行已经对了
    # TODO：非贪心的行：temperature → softmax → top-k → top-p → 逆变换采样（尽量整批向量化）
    return out
