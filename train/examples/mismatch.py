import torch

torch.manual_seed(0)
V = 200
logits = torch.randn(V) * 2
pi = logits.softmax(-1)                                        # 训练端的策略
noise = 0.2 * torch.randn(V)
noise[logits.argsort()[-8:-5]] -= 3                            # 少数几个 token 的概率被推理端严重低估（比如某个 kernel 的精度问题）
mu = (logits + noise).softmax(-1)                              # 推理引擎实际采样用的分布
A = torch.randn(V)                                             # 每个动作的优势
true_grad = (pi * A)[:, None] * (torch.eye(V) - pi[None, :])  # 精确的策略梯度 E_π[A ∇log π]，对 logits 求导
true_grad = true_grad.sum(0)

w = pi / mu
print(f"重要性比 π/μ：中位数 {w.median():.2f}，最大 {w.max():.1f}，最小 {w.min():.2f}")


def estimate(kind, n=256, C=2.0, gen=None):
    a = torch.multinomial(mu, n, replacement=True, generator=gen)   # 只能拿到推理端采的样本
    score = torch.eye(V)[a] - pi                               # ∇ log π(a) 对 logits 的梯度
    ratio = w[a]
    weight = {"不修正": torch.ones(n), "重要性采样": ratio, "TIS（截断到 2）": ratio.clamp(max=C),
              "MIS（比率超过 2 的丢掉）": ratio * (ratio <= C)}[kind]
    return (weight * A[a])[:, None].mul(score).mean(0)


gen = torch.Generator().manual_seed(1)
for kind in ("不修正", "重要性采样", "TIS（截断到 2）", "MIS（比率超过 2 的丢掉）"):
    est = torch.stack([estimate(kind, gen=gen) for _ in range(400)])
    bias = (est.mean(0) - true_grad).norm() / true_grad.norm()
    noise = (est - est.mean(0)).norm(dim=1).mean() / true_grad.norm()
    print(f"{kind:18s} 偏差 {bias:5.1%}，单个 batch 的噪声 {noise:5.1%}")
