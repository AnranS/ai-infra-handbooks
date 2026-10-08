import torch

# 一个 prompt 采样 4 个回答（GRPO 的组），奖励 1 表示答对
rewards = torch.tensor([1.0, 0.0, 0.0, 1.0])
lengths = torch.tensor([200, 50, 2000, 400])                  # 回答长度差别很大

adv = rewards - rewards.mean()
adv_std = adv / (rewards.std() + 1e-6)
print("组内优势（减均值）：", adv.tolist(), " 再除以标准差：", [round(a, 2) for a in adv_std.tolist()])

# 每个 token 的梯度权重 = 优势 × 这个 token 在损失里的系数
seq_mean = adv_std / lengths / len(lengths)                   # GRPO：先在每个回答内部求平均，再对回答求平均
tok_mean = adv_std / lengths.sum()                            # DAPO：所有 token 放在一起求平均
print("回答   奖励  长度   每个 token 的权重（序列平均）  （token 平均）  整条回答的总权重（序列平均）  （token 平均）")
for i in range(4):
    print(f"{i:4d} {rewards[i]:5.0f} {lengths[i]:5d}   {seq_mean[i]:+26.2e}  {tok_mean[i]:+13.2e}"
          f"  {seq_mean[i] * lengths[i]:+27.3f}  {tok_mean[i] * lengths[i]:+13.3f}")

# 全对或全错的组：优势全是 0，这一组对梯度没有任何贡献
for group in ([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]):
    g = torch.tensor(group)
    print(f"奖励 {group}：优势 {(g - g.mean()).tolist()}")
