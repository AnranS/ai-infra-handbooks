"""形状的功夫：view 和 reshape 的区别、permute 之后为什么不连续、广播规则、einsum"""
import torch

x = torch.arange(24).reshape(2, 3, 4)
print("x:", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

print("\n—— view / reshape / permute ——")
v = x.view(6, 4)                                              # 不复制数据，只改元数据
print("view(6, 4) 和 x 共享存储：", v.data_ptr() == x.data_ptr())
p = x.permute(0, 2, 1)                                        # 交换后两维，只改 stride
print("permute(0,2,1):", tuple(p.shape), "stride", p.stride(), "连续", p.is_contiguous())
try:
    p.view(2, 12)
except RuntimeError as e:
    print("非连续的张量不能 view：", str(e).split(".")[0])
print("reshape 会在需要时自己复制一份：", tuple(p.reshape(2, 12).shape),
      "共享存储", p.reshape(2, 12).data_ptr() == x.data_ptr())
print("-1 让 PyTorch 自己算那一维：", tuple(x.view(-1, 4).shape))
print("squeeze / unsqueeze 增删长度为 1 的维：",
      tuple(x.unsqueeze(0).shape), "->", tuple(x.unsqueeze(0).squeeze(0).shape))

print("\n—— 广播：从右往左对齐，维度相等或其中一个是 1 ——")
a = torch.ones(3, 1, 5)
b = torch.ones(4, 5)
print(f"{tuple(a.shape)} + {tuple(b.shape)} = {tuple((a + b).shape)}")
print("广播不复制数据，靠 stride=0 实现：", a.expand(3, 4, 5).stride())
try:
    torch.ones(3, 4) + torch.ones(3, 5)
except RuntimeError as e:
    print("对不上就报错：", str(e).split(".")[0])
scores = torch.zeros(2, 3)
mask = torch.tensor([True, False, True])
print("最常见的用法是给某一维加 1：", (scores + mask.unsqueeze(0)).tolist())

print("\n—— 静默的 bug：形状对得上，但不是你想要的 ——")
pred = torch.tensor([1.0, 2.0, 3.0])                          # 形状 (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])                  # 形状 (3, 1)
print(f"(3,) - (3,1) 会广播成 {tuple((pred - target).shape)}，均方误差算出来是 "
      f"{((pred - target) ** 2).mean():.4f}，而不是 0")
print("写损失函数时养成习惯：两个张量的形状打印出来对一遍，或者用 assert")

print("\n—— einsum：把下标写出来，省得数 permute ——")
q = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(1))   # (batch, 序列, 维度)
k = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(2))
by_hand = q @ k.transpose(1, 2)
by_einsum = torch.einsum("bqd,bkd->bqk", q, k)
print("注意力分数两种写法一致：", torch.allclose(by_hand, by_einsum), tuple(by_einsum.shape))
print("多头的版本一行就写完：", tuple(torch.einsum("bhqd,bhkd->bhqk",
                                              q.view(2, 2, 2, 8), k.view(2, 2, 2, 8)).shape))
