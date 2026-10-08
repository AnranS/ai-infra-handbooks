# （六）数据与训练循环

<p class="lead">这一章把前面的东西拼起来：先用 <code>Dataset</code> 和 <code>DataLoader</code> 把变长样本拼成 batch，再写一个带评估、学习率调度和梯度裁剪的训练循环。主干只有五行，但每一行都有一个常见的写错方式。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 自定义 `Dataset` 至少要实现哪两个方法？
    2. 样本长度不一样时，默认的 collate 会怎样？该怎么办？
    3. 训练循环的五个步骤是什么？顺序能不能换？
    4. `zero_grad(set_to_none=True)` 和填零有什么区别？
    5. `scheduler.step()` 该放在哪里？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `__len__` 和 `__getitem__`。
    2. 默认 collate 会 `torch.stack`，形状不一致直接报错。要自己写 `collate_fn`，补齐到本 batch 的最长，并返回 mask。
    3. 前向算损失 → `zero_grad` → `backward` → （裁剪）→ `step`。`zero_grad` 必须在 `backward` 之前，`step` 必须在 `backward` 之后。
    4. `set_to_none=True` 把 `.grad` 置为 None 而不是写一遍 0：省一次显存写，而且"没收到梯度的参数"会保持 None，容易发现问题。
    5. 按 epoch 调度的放在 epoch 循环末尾；按步调度的（warmup）放在内层循环里。两种不能混。

## 变长样本怎么拼成 batch

```python title="data.py"
"""Dataset、DataLoader 和 collate_fn：变长样本怎么拼成一个 batch"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ToyText(Dataset):
    """一个最小的 Dataset：实现 __len__ 和 __getitem__ 就够了"""

    def __init__(self, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.lens = torch.randint(2, 7, (n,), generator=g).tolist()     # 每条样本长度不同
        self.g = g

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        return torch.arange(self.lens[i]) + 10 * i, self.lens[i] % 2    # (序列, 标签)


ds = ToyText()
print("样本长度：", ds.lens)
print("第 0 条：", ds[0])

print("\n—— 默认的 collate 要求形状一致 ——")
try:
    next(iter(DataLoader(ds, batch_size=3)))
except RuntimeError as e:
    print("直接用会报错：", str(e).split(".")[0])


def collate(batch):
    """自己拼：补齐到本 batch 里最长的那条，并给出 mask"""
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask = padded.new_zeros(padded.shape, dtype=torch.bool)
    for i, s in enumerate(seqs):
        mask[i, :len(s)] = True
    return padded, mask, torch.tensor(labels)


loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate)
for step, (x, mask, y) in enumerate(loader):
    print(f"batch {step}: x{tuple(x.shape)} 真实 token 数 {mask.sum().item():2d}/{mask.numel():2d}  标签 {y.tolist()}")
print("补齐到**本 batch** 的最长，而不是全局最长——所以按长度排序再分桶能少算很多 padding")

print("\n—— shuffle 与可复现 ——")
a = [b[2].tolist() for b in DataLoader(ds, batch_size=3, shuffle=True,
                                       generator=torch.Generator().manual_seed(7), collate_fn=collate)]
b = [b[2].tolist() for b in DataLoader(ds, batch_size=3, shuffle=True,
                                       generator=torch.Generator().manual_seed(7), collate_fn=collate)]
print("给 DataLoader 一个 generator，两次 shuffle 的顺序一样：", a == b, a)
print("num_workers>0 时每个 worker 还要单独设种子（worker_init_fn），否则各进程的数据增强会撞上")
print("drop_last=True 丢掉最后不满的一批；分布式训练里常开，省得各卡步数对不齐")
```

```text title="输出"
样本长度： [6, 6, 5, 2, 5, 6, 4, 5]
第 0 条： (tensor([0, 1, 2, 3, 4, 5]), 0)

—— 默认的 collate 要求形状一致 ——
直接用会报错： stack expects each tensor to be equal size, but got [6] at entry 0 and [5] at entry 2
batch 0: x(3, 6) 真实 token 数 17/18  标签 [0, 0, 1]
batch 1: x(3, 6) 真实 token 数 13/18  标签 [0, 1, 0]
batch 2: x(2, 5) 真实 token 数  9/10  标签 [0, 1]
补齐到**本 batch** 的最长，而不是全局最长——所以按长度排序再分桶能少算很多 padding

—— shuffle 与可复现 ——
给 DataLoader 一个 generator，两次 shuffle 的顺序一样： True [[0, 1, 0], [0, 1, 1], [0, 0]]
num_workers>0 时每个 worker 还要单独设种子（worker_init_fn），否则各进程的数据增强会撞上
drop_last=True 丢掉最后不满的一批；分布式训练里常开，省得各卡步数对不齐
```

- **`collate_fn` 是 batch 的组装说明书**。默认实现只会 `stack`，所以变长数据必须自己写。补齐到**本 batch** 的最长而不是全局最长——这是最省算力的做法，也是为什么长度相近的样本应该分到一桶（bucketing）；
- **mask 要一起返回**。补出来的位置不能参与注意力和损失，上一章的 `masked_fill` 就派上用场了；
- **shuffle 的随机性要能控制**。给 `DataLoader` 一个 `generator`，两次跑的顺序就一样。`num_workers > 0` 时每个 worker 还要单独设种子，否则多个进程的数据增强可能完全一样。

## 完整的训练循环

```python title="loop.py" ci="loose"
"""一个完整的训练循环：五行主干，外加评估、学习率调度和梯度裁剪"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# 造一个两类的玩具数据：两个高斯团
g = torch.Generator().manual_seed(0)
n = 512
x = torch.cat([torch.randn(n, 2, generator=g) + 1.5, torch.randn(n, 2, generator=g) - 1.5])
y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
perm = torch.randperm(len(x), generator=g)
x, y = x[perm], y[perm]
train = TensorDataset(x[:800], y[:800])
val = TensorDataset(x[800:], y[800:])
loader = DataLoader(train, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(1))

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
loss_fn = nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate():
    model.eval()                                              # 关掉 dropout / BN 的训练行为
    xb, yb = val.tensors
    logits = model(xb)
    acc = (logits.argmax(1) == yb).float().mean().item()
    model.train()                                             # 记得切回来
    return loss_fn(logits, yb).item(), acc


print("epoch   训练 loss   验证 loss   验证准确率   学习率")
for epoch in range(10):
    total = 0.0
    for xb, yb in loader:                                     # ——— 五行主干 ———
        loss = loss_fn(model(xb), yb)                         # 1. 前向 + 算损失
        opt.zero_grad(set_to_none=True)                       # 2. 清空上一步的梯度
        loss.backward()                                       # 3. 反向
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 4. 梯度裁剪（可选，但建议有）
        opt.step()                                            # 5. 更新参数
        total += loss.item() * len(xb)                        # .item() 会同步，放在循环里只是为了打日志
    sched.step()                                              # 调度器按 epoch 走，别放进内层循环
    vl, acc = evaluate()
    print(f"{epoch + 1:5d}   {total / len(train):9.4f}   {vl:9.4f}   {acc:10.1%}   {sched.get_last_lr()[0]:.2e}")

print("\n几条容易写错的：")
print("- zero_grad 要在 backward **之前**；忘了它，梯度会一直累加")
print("- set_to_none=True 把梯度置空而不是填零，省一次写显存，也能让「没收到梯度的参数」暴露出来")
print("- loss.item() 会强制同步 GPU，别在每一步都调；要累计就先留在张量里，最后再取")
print("- scheduler.step() 按 epoch 调用；按步调度的（warmup）要放进内层循环，两种别混")
```

```text title="输出"
epoch   训练 loss   验证 loss   验证准确率   学习率
    1      0.4155      0.0677        99.6%   9.76e-03
    2      0.0819      0.0258        99.6%   9.05e-03
    3      0.0749      0.0211        99.6%   7.94e-03
    4      0.0738      0.0214        99.6%   6.55e-03
    5      0.0727      0.0220        99.6%   5.00e-03
    6      0.0726      0.0231       100.0%   3.45e-03
    7      0.0717      0.0233       100.0%   2.06e-03
    8      0.0714      0.0234        99.6%   9.55e-04
    9      0.0712      0.0235        99.6%   2.45e-04
   10      0.0711      0.0236        99.6%   0.00e+00

几条容易写错的：
- zero_grad 要在 backward **之前**；忘了它，梯度会一直累加
- set_to_none=True 把梯度置空而不是填零，省一次写显存，也能让「没收到梯度的参数」暴露出来
- loss.item() 会强制同步 GPU，别在每一步都调；要累计就先留在张量里，最后再取
- scheduler.step() 按 epoch 调用；按步调度的（warmup）要放进内层循环，两种别混
```

主干五行，每一行都对应一类错误：

| 步骤 | 写错的样子 | 后果 |
| --- | --- | --- |
| 前向 + 损失 | 形状广播错了（见第二章） | 不报错，loss 降不下去 |
| `zero_grad` | 忘了，或者放在 `backward` 之后 | 梯度一直累加，等效学习率越来越大 |
| `backward` | 对非标量调用 | 报错，要么 `.sum()` 要么传 `gradient=` |
| 梯度裁剪 | 放在 `step` 之后 | 等于没裁 |
| `step` | 忘了 `zero_grad` 配套 | 同上 |

几个工程细节：

- **`loss.item()` 会强制同步**。在 GPU 上，它要等这一步真的算完才能取出数来。每一步都调用会把 CPU 和 GPU 的流水线打断；累计损失时先留在张量里，一个 epoch 结束再 `.item()`；
- **`set_to_none=True`** 现在是默认值。除了省一次写显存，它还有个调试上的好处：没收到梯度的参数 `.grad` 会一直是 None，而不是一个看起来很正常的全零张量；
- **评估要 `eval()` + `no_grad()`**，而且评估完记得切回 `train()`。上面用装饰器 `@torch.no_grad()` 包了整个函数，比到处写 `with` 干净；
- **调度器按 epoch 还是按步**，取决于你用的是哪一类。混用会让学习率曲线完全不是你以为的样子——打印出来看一眼是最省事的验证方式。

!!! interview "怎么讲清楚"
    把训练循环说成五步：**前向算损失 → zero_grad → backward →（裁剪）→ step**，并指出顺序约束（zero_grad 在 backward 前，step 在 backward 后，裁剪夹在中间）。然后讲三个工程细节：`loss.item()` 会同步 GPU，日志别每步都取；`set_to_none=True` 省一次写显存、也让"没收到梯度"暴露出来；评估要 `eval()` 和 `no_grad()` 一起上、评估完切回 `train()`。数据这边讲 `collate_fn`：变长样本补齐到**本 batch** 最长并返回 mask，长度相近的分桶能省掉大量 padding 计算。

## 练习

**1. 按长度分桶。** 把 `ToyText` 的样本按长度排序再分 batch，统计 padding 占比从多少降到多少。

??? success "参考思路"
    用 `BatchSampler` 或者先排序再切块。玩具数据上就能看到 padding 比例明显下降；真实的变长文本数据上，分桶经常能省掉三成以上的计算。代价是打乱程度变低，通常的折中是"桶内随机、桶间随机"。

**2. 梯度裁剪的阈值。** 把 `clip_grad_norm_` 的返回值（裁剪前的范数）打印出来，训练几个 epoch 看它的分布，再决定阈值该设多少。

??? success "参考思路"
    返回值是裁剪**前**的全局范数。通常的做法是先不裁（设一个很大的阈值）跑几百步，看范数稳定在什么量级，再把阈值设在那个量级的一两倍。范数突然飙升往往是 loss 突刺的前兆。

**3. 加上早停。** 记录验证 loss 最低时的 `state_dict`，训练结束后恢复它。

??? success "参考思路"
    注意要 `{k: v.clone() for k, v in model.state_dict().items()}`——`state_dict()` 返回的是对参数张量的**引用**，不 clone 的话后续训练会把它改掉，你存下来的"最佳权重"会跟着一起变。这是个很隐蔽的坑。

## 小结

- [x] `Dataset` 实现 `__len__` 和 `__getitem__`；变长数据要自己写 `collate_fn`，补齐到本 batch 最长并返回 mask。
- [x] 训练循环五步：前向 → zero_grad → backward → 裁剪 → step，顺序有硬约束。
- [x] `loss.item()` 会同步 GPU，日志别每步都取；`set_to_none=True` 让"没收到梯度"暴露出来。
- [x] 评估要 `eval()` + `no_grad()`，完了切回 `train()`；调度器按 epoch 还是按步别混。
