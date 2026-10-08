"""索引、归约与掩码：切片是视图、高级索引是拷贝，以及 gather / scatter / masked_fill 怎么用"""
import torch

x = torch.arange(12).reshape(3, 4)
print("x =\n", x)

print("\n—— 切片是视图，高级索引是拷贝 ——")
s = x[1:, 2:]
s[0, 0] = -1
print("改切片会改到原张量：\n", x)
a = x[[0, 2]]                                                 # 整数列表 / 张量索引 = 高级索引
a[0, 0] = -999
print("改高级索引的结果不影响原张量：x[0,0] =", x[0, 0].item())
print("布尔索引同样是拷贝，而且会把结果拉平：", x[x > 8].tolist())

print("\n—— 按下标取：gather ——")
logits = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
labels = torch.tensor([1, 0])
picked = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
print("每一行取 label 那一列（交叉熵里天天用）：", picked.round(decimals=3).tolist())
print("等价的高级索引写法：", logits[torch.arange(2), labels].round(decimals=3).tolist())

print("\n—— 按下标写：scatter_ ——")
onehot = torch.zeros(2, 3).scatter_(1, labels.unsqueeze(1), 1.0)
print("做 one-hot：\n", onehot)
print("带下划线的方法是**原地**修改，会改掉输入，autograd 里要小心")

print("\n—— 掩码 ——")
scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [0.0, 3.0, 4.0]])
causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))       # 下三角为 True
masked = scores.masked_fill(~causal, float("-inf"))
print("因果掩码之后的分数：\n", masked)
print("softmax 之后被屏蔽的位置正好是 0：\n", masked.softmax(-1).round(decimals=3))
print("用 -inf 而不是很大的负数：softmax 之后严格为 0，而且不会在 fp16 下溢出成 nan")
print("torch.where 按条件二选一：", torch.where(scores > 3, scores, torch.zeros_like(scores)).tolist())

y = torch.arange(12).reshape(3, 4)                            # 重新来一份干净的，前面那个被改过

print("\n—— 归约：dim 和 keepdim ——")
print("y.sum() 把所有元素加起来：", y.sum().item())
print("y.sum(dim=0) 沿着第 0 维加，形状", tuple(y.sum(dim=0).shape), "：", y.sum(dim=0).tolist())
print("keepdim=True 保留那一维，方便广播：", tuple(y.sum(dim=1, keepdim=True).shape))
print("减去每行最大值（softmax 的标准做法）：", (y - y.max(dim=1, keepdim=True).values)[0].tolist())
print("max 返回 (值, 下标) 两个张量，argmax 只返回下标：", y.max(dim=1).indices.tolist(), y.argmax(dim=1).tolist())
print("topk 取前 k 大：", logits.topk(2, dim=1).indices.tolist())
print("cumsum 做前缀和（变长序列的偏移量常这么算）：", torch.tensor([2, 3, 1]).cumsum(0).tolist())
