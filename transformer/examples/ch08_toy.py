"""第 8 章（一）：3 个 token 手算分块：先逐 token 递推，再把同一件事写成一个下三角方程组，一次解出来，对答案"""
import torch

r = lambda t: [round(x, 3) for x in t.tolist()]
K = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])           # 3 个 token 的 key（已经是单位长度）
V = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])           # value
beta = torch.tensor([1.0, 1.0, 0.5])                              # 写入强度
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])           # query
print("1. 逐 token 递推（先不管衰减，表从空的开始）")
S = torch.zeros(2, 2)
for t in range(3):
    pred = S.T @ K[t]                                             # 表里现在认为 k_t 对应什么
    u = beta[t] * (V[t] - pred)                                   # 这个 token 真正写进去的量
    S = S + torch.outer(K[t], u)
    print(f"   token {t + 1}：表里预测 {r(pred)}，真正写入 u{t + 1} = {r(u)}，输出 {r(S.T @ Q[t])}")
print(f"   最后的表：{[r(row) for row in S]}")
print("2. 分块：三个 u 一次解出来")
A = torch.tril(beta[:, None] * (K @ K.T), -1)                     # A[t, j] = β_t · (k_t · k_j)，只留 j < t
print(f"   A = {[r(row) for row in A]}")
U = torch.linalg.solve_triangular(torch.eye(3) + A, beta[:, None] * V, upper=False)   # 解 (I + A) U = β ⊙ V
print(f"   解 (I + A) U = β ⊙ V 得 U = {[r(row) for row in U]}")
print(f"   输出 = (Q Kᵀ 只留下三角，含对角线) · U = {[r(row) for row in torch.tril(Q @ K.T) @ U]}")
print(f"   最后的表 = Kᵀ · U = {[r(row) for row in K.T @ U]}")
