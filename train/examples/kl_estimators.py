import torch

torch.manual_seed(0)
V = 1000
p_ref = torch.randn(V).softmax(-1)                            # 参考模型在某个位置的分布
q = (p_ref.log() + 0.3 * torch.randn(V)).softmax(-1)          # 当前策略：离参考模型不远
exact = (q * (q / p_ref).log()).sum()                         # KL(q ‖ p_ref) 的精确值

x = torch.multinomial(q, 100_000, replacement=True)           # 从当前策略采样的 token（rollout）
r = p_ref[x] / q[x]
estimators = {"k1 = -log r": -r.log(), "k2 = (log r)² / 2": r.log() ** 2 / 2, "k3 = (r - 1) - log r": (r - 1) - r.log()}
print(f"精确的 KL：{exact:.4f}")
for name, v in estimators.items():
    print(f"{name:22s} 均值 {v.mean():.4f}，标准差 {v.std():.4f}，负值的比例 {(v < 0).float().mean():.0%}")
