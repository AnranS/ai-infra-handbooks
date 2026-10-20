"""第 7 章（一）：在两维的记忆表上手算：写 = 外积相加，读 = Sᵀq；线性注意力 → DeltaNet → Gated DeltaNet 一步步改；最后看表能记多少"""
import torch

r = lambda t, n=3: [round(x, n) for x in t.tolist()]
k1, k2 = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
S = torch.outer(k1, torch.tensor([3.0, 1.0])) + torch.outer(k2, torch.tensor([-1.0, 2.0]))   # 1. 写：k vᵀ 加进表里
print(f"1. 写入 k1 → [3, 1]、k2 → [-1, 2] 之后，表 S = {[r(row) for row in S]}")
for q in (k1, k2, torch.tensor([0.6, 0.8])):                         #    读：o = Sᵀq
    print(f"   用 q = {r(q)} 去读：Sᵀq = {r(S.T @ q)}")

v_new = torch.tensor([5.0, 0.0])                                      # 2. 现在 k1 要改成 [5, 0]
linear = S + torch.outer(k1, v_new)                                   #    线性注意力：直接加上去
delta = S + torch.outer(k1, v_new - S.T @ k1)                         #    DeltaNet：先用 k1 查表，只写差值（β = 1）
alpha = 0.5
gated = alpha * S + torch.outer(k1, v_new - (alpha * S).T @ k1)       #    Gated DeltaNet：整张表先 × α，再写差值
for name, M in (("线性注意力", linear), ("DeltaNet", delta), ("Gated DeltaNet（α = 0.5）", gated)):
    print(f"2. {name}：读 k1 得 {r(M.T @ k1)}，读 k2 得 {r(M.T @ k2)}")

for a in (1.0, 0.9, 0.5):                                             # 3. 衰减：不再写的旧内容每一步乘一次 α
    print(f"3. α = {a}：10 步后旧内容剩 {a ** 10:.3f}，50 步后剩 {a ** 50:.3g}")

torch.manual_seed(0)                                                  # 4. 容量：一张 128 × 128 的表塞进 n 对 (k, v)，再用每个 k 读回来
dk = 128


def fill(K, V):
    plain, dlt = torch.zeros(dk, dk), torch.zeros(dk, dk)
    for k, v in zip(K, V):
        plain += torch.outer(k, v)
        dlt += torch.outer(k, v - dlt.T @ k)
    err = lambda M, s: ((K[s] @ M - V[s]).norm(dim=-1) / V[s].norm(dim=-1)).mean().item()   # 读回来的值和 v 差多少（相对误差）
    return err(plain, slice(None)), err(dlt, slice(None)), err(dlt, slice(-16, None)), err(dlt, slice(0, 16))


Q, _ = torch.linalg.qr(torch.randn(dk, dk))
p, d, _, _ = fill(Q, torch.randn(dk, dk))
print(f"4. 128 个互相垂直的 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}")
for n in (256, 1024):
    p, d, last, first = fill(torch.nn.functional.normalize(torch.randn(n, dk), dim=-1), torch.randn(n, dk))
    print(f"   {n} 个随机 key：线性注意力误差 {p:.0%}，delta 规则 {d:.0%}（其中最后写的 16 个 {last:.0%}，最早写的 16 个 {first:.0%}）")
