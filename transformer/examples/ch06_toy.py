"""第 6 章（一）：三个 token、两维向量，手算一遍注意力；再看 softmax 为什么不让一个头"交白卷" """
import math

import torch

words = ["今天", "天气", "不错"]
k = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])      # 每个 token 的 key：给别人查的"标签"
v = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])      # 每个 token 的 value：被取走的"内容"
q = torch.tensor([0.0, 4.0])                                # 「不错」的 query：它想找"被评价的是谁"

scores = k @ q / math.sqrt(2)                               # 1. 打分：和每个 key 做点积，再除以 √d（d = 2）
w = torch.softmax(scores, dim=-1)                           # 2. softmax：变成加起来正好是 1 的比例
out = w @ v                                                 # 3. 加权求和：按比例把 value 混在一起
r = lambda t, n=3: [round(x, n) for x in t.tolist()]
print(f"「不错」给三个 token 打的分：{r(scores, 2)}")
print(f"softmax 之后的比例：{r(w)}，加起来 {w.sum():.3f}")
print(f"输出 = 比例 × value 再相加：{r(out)}")

Q = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])      # 4. 三个 token 一起算：每行一个 query
mask = torch.ones(3, 3, dtype=torch.bool).triu(1)          # 因果掩码：上三角填 −inf，第 i 行只能看前 i 个
W = torch.softmax((Q @ k.T / math.sqrt(2)).masked_fill(mask, float("-inf")), dim=-1)
print("三行一起算、加上因果掩码（行 = query，列 = key）：")
for word, row in zip(words, W):
    print(f"  {word}  {r(row)}")

q0 = torch.tensor([0.0, 0.0])                               # 5. 这个头这次什么都不想看：query 全是 0
w0 = torch.softmax(k @ q0 / math.sqrt(2), dim=-1)
print(f"什么都不想看时的比例：{r(w0)} → 输出 {r(w0 @ v)}：还是交了一份平均值")
g = torch.sigmoid(torch.tensor(-6.0))                       # 6. 有门：输出再乘一个 sigmoid(−6)
print(f"乘上门 sigmoid(−6) = {g:.4f} → 输出 {r(w0 @ v * g, 4)}：这次真的没发言")
