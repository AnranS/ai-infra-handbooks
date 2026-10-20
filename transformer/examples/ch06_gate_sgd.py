"""第 6 章（三）：门是怎么学出来的——一个能手算的最小例子，看 W_gate 从初始化一步步变成"该开就开、该关就关"。

两种 token：x(问号) = [1, 0]，x(数字) = [0, 1]；门只有 1 维，W_gate 就是两个数 w = [w_问号, w_数字]。
这一层的输出 y = 残差 r + 门 g × 头的输出 o，目标 t = 1，损失 (y − t)² / 2：
  问号：r = 0（残差流里还没有答案），o = 1.0（头找到了答案）     → 门应该开
  数字：r = 1（残差流里已经对了），  o = 0.8（头在这里只会添乱） → 门应该关
"""
import torch

x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
r, o, t = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.8]), torch.tensor([1.0, 1.0])
w = torch.tensor([0.01, -0.02], requires_grad=True)        # 初始化：很小的随机数，门都在 0.5 附近
for step in range(201):
    g = torch.sigmoid(x @ w)                               # 门
    loss = ((r + g * o - t) ** 2 / 2).sum()
    w.grad = None
    loss.backward()                                        # 梯度 = (y − t) × o × g(1 − g) × x
    if step in (0, 1, 2, 5, 20, 50, 200):
        print(f"第 {step:3d} 步：w = [{w[0].item():+.3f}, {w[1].item():+.3f}]，门(问号) {g[0].item():.3f}，门(数字) {g[1].item():.3f}，"
              f"损失 {loss.item():.4f}，梯度 [{w.grad[0].item():+.4f}, {w.grad[1].item():+.4f}]")
    with torch.no_grad():
        w -= 1.0 * w.grad                                  # 学习率 1
