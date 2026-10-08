# （五）nn.Module：把模型搭起来

<p class="lead">`nn.Module` 做的事只有一件：**帮你把参数管起来**。注册了的参数能被优化器拿到、能跟着模型搬到 GPU 上、能被存进 `state_dict`；没注册的就什么都不是。这一章把注册规则、buffer、`state_dict` 和 `train/eval` 讲清楚，顺带演示两个新手常踩的坑。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `nn.Parameter` 和普通张量有什么区别？
    2. buffer 是什么？它和参数有什么不同？
    3. 把几个子模块放进一个普通的 Python list，会发生什么？
    4. `model.eval()` 会关掉梯度吗？
    5. `state_dict()` 里有哪些东西？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `nn.Parameter` 被赋值给 Module 的属性时会**自动注册**成参数：出现在 `parameters()` 里、会被优化器更新、会跟着 `.to(device)` 搬、会进 `state_dict`。普通张量什么都不会发生。
    2. 不需要训练但属于模型状态的张量，比如 BatchNorm 的滑动均值、RoPE 的 cos/sin 表。用 `register_buffer` 注册，它进 `state_dict`、跟着 `.to()` 搬，但不在 `parameters()` 里。
    3. 注册不上。参数拿不到、设备搬不动、存不进 `state_dict`。要用 `nn.ModuleList` 或 `nn.ModuleDict`。
    4. 不会。它只切换 dropout、batchnorm 这类"训练/推理两套行为"的层。关梯度是 `no_grad` 的事，两者必须分开理解。
    5. 所有注册过的参数和 buffer，键是按层级拼出来的字符串（`blocks.0.fc.weight`）。

## 注册、buffer 与 state_dict

```python title="module.py"
"""nn.Module：参数怎么注册、buffer 是什么、state_dict 长什么样、train 和 eval 有什么区别"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d_in, d_out, p=0.5):
        super().__init__()                                    # 忘了这一句，后面注册参数会直接报错
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)
        self.scale = nn.Parameter(torch.ones(d_out))          # 自己加的可训练参数
        self.register_buffer("calls", torch.zeros(1))         # 不训练但要随 state_dict 存取的状态

    def forward(self, x):
        self.calls += 1
        return self.drop(self.norm(self.fc(x)) * self.scale)


class Net(nn.Module):
    def __init__(self, d=4, n_block=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, d) for _ in range(n_block))   # 用 ModuleList，不要用 []
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)                                      # 残差
        return self.head(x)


net = Net()
print("—— 参数 ——")
total = sum(p.numel() for p in net.parameters())
print(f"参数总数 {total}，可训练的 {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
for name, p in list(net.named_parameters())[:4]:
    print(f"  {name:24s} {tuple(p.shape)}")
print("  ……")
print("buffer 不在 parameters() 里，但在 state_dict 里：", [n for n, _ in net.named_buffers()])

print("\n—— 放进 list 就注册不上了 ——")
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2) for _ in range(2)]     # 错误示范

print("Broken 的参数个数：", sum(p.numel() for p in Broken().parameters()), "——优化器什么也拿不到，模型也不会跟着 .to(device) 搬")

print("\n—— state_dict ——")
sd = net.state_dict()
print("键的数量：", len(sd), "，前三个：", list(sd)[:3])
print("保存和加载就是它：torch.save(net.state_dict(), ...) / net.load_state_dict(...)")
print("load_state_dict 的返回值会告诉你缺了哪些、多了哪些：", net.load_state_dict(sd, strict=True))

print("\n—— 初始化 ——")
def init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(m.bias)

net.apply(init)                                               # apply 会递归作用到每一个子模块
print("apply 之后第一层权重的标准差：", f"{net.blocks[0].fc.weight.std().item():.4f}")

print("\n—— train 和 eval ——")
x = torch.randn(3, 4)
net.train()
out_train = torch.stack([net(x) for _ in range(2)])
net.eval()
out_eval = torch.stack([net(x) for _ in range(2)])
print("train 模式下两次前向不同（dropout 在随机丢）：", not torch.allclose(out_train[0], out_train[1]))
print("eval 模式下两次前向相同：", torch.allclose(out_eval[0], out_eval[1]))
print("eval() 只影响 dropout、batchnorm 这类有「训练/推理两套行为」的层，它**不会**关掉梯度——那是 no_grad 的事")
print("buffer 记下来的调用次数：", net.blocks[0].calls.item())
```

```text title="输出"
—— 参数 ——
参数总数 74，可训练的 74
  blocks.0.scale           (4,)
  blocks.0.fc.weight       (4, 4)
  blocks.0.fc.bias         (4,)
  blocks.0.norm.weight     (4,)
  ……
buffer 不在 parameters() 里，但在 state_dict 里： ['blocks.0.calls', 'blocks.1.calls']

—— 放进 list 就注册不上了 ——
Broken 的参数个数： 0 ——优化器什么也拿不到，模型也不会跟着 .to(device) 搬

—— state_dict ——
键的数量： 14 ，前三个： ['blocks.0.scale', 'blocks.0.calls', 'blocks.0.fc.weight']
保存和加载就是它：torch.save(net.state_dict(), ...) / net.load_state_dict(...)
load_state_dict 的返回值会告诉你缺了哪些、多了哪些： <All keys matched successfully>

—— 初始化 ——
apply 之后第一层权重的标准差： 0.0236

—— train 和 eval ——
train 模式下两次前向不同（dropout 在随机丢）： True
eval 模式下两次前向相同： True
eval() 只影响 dropout、batchnorm 这类有「训练/推理两套行为」的层，它**不会**关掉梯度——那是 no_grad 的事
buffer 记下来的调用次数： 4.0
```

几条要记住的：

- **赋值即注册**。`self.fc = nn.Linear(...)` 这一行同时做了两件事：把子模块挂上去，并让它的参数成为本模块参数的一部分。所以 `super().__init__()` 必须第一行就调用，否则注册机制还没初始化，赋值会直接报错；
- **普通 list 装子模块 = 白写**。上面 `Broken` 的参数个数是 0。这个坑之所以危险，是因为代码能跑、loss 也在降（因为别的层在学），只是那几层永远不更新。看到"某一部分好像完全没学"，先查这里；
- **buffer 不是参数**：不会被优化器更新，但会被保存、会跟着设备走。自己写位置编码表、因果掩码时用它，别用普通属性（不跟着 `.to()` 走，GPU 上会直接报设备不匹配）；
- **`state_dict` 是扁平的字典**，键就是属性路径。加载时 `strict=True`（默认）要求两边完全一致，返回值会告诉你缺了哪些、多了哪些——改了模型结构之后加载旧权重，就靠读这个返回值定位；
- **`train()` / `eval()` 只管行为，不管梯度**。推理时两件事都要做：`model.eval()` 加 `with torch.no_grad()`。只做前者，dropout 关了但显存照样涨；只做后者，dropout 还在随机丢，同一个输入两次结果不一样。

!!! tip "apply 做初始化"
    `model.apply(fn)` 会把 `fn` 递归作用到每一个子模块上，配合 `isinstance` 判断就能"所有 Linear 这样初始化、所有 LayerNorm 那样初始化"。这是从零写模型时的标准做法，比在每个 `__init__` 里零散地初始化好维护。

!!! interview "怎么讲清楚"
    讲 `nn.Module`：它的核心职责是**参数管理**——`nn.Parameter` 赋值给属性就自动注册，于是优化器拿得到、`.to(device)` 搬得动、`state_dict` 存得下。接着讲三个容易错的点：子模块必须放 `nn.ModuleList` 而不是普通 list（否则参数个数是 0，那几层永远不更新）；不训练但要保存的状态用 `register_buffer`（BN 的滑动统计、RoPE 的 cos/sin 表）；`eval()` 只切换 dropout/BN 的行为，**不关梯度**，推理要 `eval()` 和 `no_grad()` 一起上。

## 练习

**1. 验证注册。** 把 `Broken` 里的 list 换成 `nn.ModuleList`，再数一遍参数个数；然后试试 `.to(torch.float64)`，看两种写法的层分别变没变。

??? success "参考思路"
    换成 `ModuleList` 之后参数就出现了。`.to()` 的差别更直观：普通 list 里的层 dtype 不变，于是前向时报 "expected scalar type Double but found Float"——很多"设备/类型不匹配"的报错根子都在这里。

**2. 共享权重。** 让输出层复用嵌入层的权重矩阵（权重绑定），然后数一下 `parameters()` 的个数，看它被算了一次还是两次。

??? success "参考答案"
    `self.head.weight = self.embed.weight` 之后，`parameters()` 会**去重**，只算一次（它内部按对象 id 去重）。但 `state_dict()` 里两个键都在，指向同一个张量——存出来的文件会大一点，加载回来仍然是共享的。

**3. 冻结与部分加载。** 加载一个结构改过的 checkpoint：用 `strict=False`，然后打印返回的 missing / unexpected，解释每一项为什么出现。

??? success "参考思路"
    `missing_keys` 是模型里有、文件里没有的（新加的层），`unexpected_keys` 反之（删掉的层）。微调时常见的做法是先 `strict=False` 加载骨干，再单独初始化新加的头，并把返回值打到日志里——否则哪天拼错一个名字，权重悄悄没加载上，你只会看到效果莫名其妙地差。

## 小结

- [x] `nn.Parameter` 赋值给属性就自动注册；`super().__init__()` 必须最先调用。
- [x] 子模块放 `nn.ModuleList` / `nn.ModuleDict`，放普通 list 等于没注册。
- [x] 不训练但要保存、要跟着设备走的状态用 `register_buffer`。
- [x] `state_dict` 是扁平字典；加载时读返回的 missing / unexpected 定位问题。
- [x] `eval()` 只切换 dropout/BN 的行为，不关梯度——推理要和 `no_grad` 一起用。
