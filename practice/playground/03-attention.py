"""从零写因果多头注意力（numpy）"""
import numpy as np

rng = np.random.default_rng(0)
T, D, H = 6, 32, 4                                   # 序列长度、隐藏维度、头数
hd = D // H
x = rng.standard_normal((T, D))
Wq, Wk, Wv, Wo = (rng.standard_normal((D, D)) / np.sqrt(D) for _ in range(4))


def attention(x):
    q = (x @ Wq).reshape(T, H, hd).transpose(1, 0, 2)        # [H, T, hd]
    k = (x @ Wk).reshape(T, H, hd).transpose(1, 0, 2)
    v = (x @ Wv).reshape(T, H, hd).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)           # [H, T, T]
    scores = np.where(np.tril(np.ones((T, T), bool)), scores, -np.inf)   # 因果掩码：只能看自己和之前的位置
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    out = (p @ v).transpose(1, 0, 2).reshape(T, D)
    return out @ Wo, p


out, p = attention(x)
print("输出形状", out.shape, "；注意力权重形状", p.shape)
print("第 0 个头的注意力权重（每行之和为 1，右上角为 0）：\n", np.round(p[0], 2))
# 试试：把 x 的最后一行改掉，前面几行的输出会变吗？（因果掩码保证不会）
