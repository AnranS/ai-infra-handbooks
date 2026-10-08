import torch

bias = torch.tensor([1.0, 2.0, 3.0])
e = bias.expand(4, 3)              # 4 行都指向同一段内存
r = bias.repeat(4, 1)              # 真的复制成 4 行
print("expand：stride", e.stride(), "存储字节", e.untyped_storage().nbytes())
print("repeat：stride", r.stride(), "存储字节", r.untyped_storage().nbytes())

e[0, 0] = 5.0                      # 4 行共用这个元素：改一个等于改一列
print("写 e[0, 0] 之后 bias =", bias.tolist(), "e[3, 0] =", e[3, 0].item())
try:
    e.add_(1)                      # 原地运算会写同一个位置多次：直接报错
except RuntimeError as err:
    print("原地 add_ 失败：", str(err).split(":")[0])

win = torch.arange(6).unfold(0, 3, 1)   # 滑动窗口也是一个视图：相邻窗口的 stride 是 1
print("滑动窗口：stride", win.stride(), "内容", win.tolist())
