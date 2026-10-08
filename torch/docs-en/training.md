# 6. Data and the training loop

<p class="lead">This chapter puts the pieces together: first <code>Dataset</code> and <code>DataLoader</code> turn variable-length samples into batches, then a training loop with evaluation, a learning-rate schedule and gradient clipping. The core is five lines, and every one of them has a common way of being written wrong.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which two methods does a custom `Dataset` have to implement?
    2. What does the default collate do when the samples have different lengths? What should you do?
    3. What are the five steps of a training loop? Can the order change?
    4. How does `zero_grad(set_to_none=True)` differ from filling with zeros?
    5. Where does `scheduler.step()` belong?

??? success "Answers (try first, then expand to compare)"
    1. `__len__` and `__getitem__`.
    2. The default collate calls `torch.stack` and raises when the shapes differ. Write a `collate_fn` that pads to the longest sample in this batch and returns a mask.
    3. Forward and loss → `zero_grad` → `backward` → (clip) → `step`. `zero_grad` must come before `backward`, and `step` after it.
    4. `set_to_none=True` sets `.grad` to None instead of writing zeros: one less memory write, and parameters that received no gradient stay None, which makes problems visible.
    5. A per-epoch schedule steps at the end of the epoch loop; a per-step one (warmup) goes in the inner loop. Do not mix the two.

## Turning variable-length samples into batches {#变长样本怎么拼成-batch}

```python title="data.py"
"""Dataset、DataLoader 和 collate_fn：变长样本怎么拼成一个 batch"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class ToyText(Dataset):
    """一个最小的 Dataset：实现 __len__ 和 __getitem__ 就够了"""

    def __init__(self, n=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.lens = torch.randint(2, 7, (n,), generator=g).tolist()     # every sample has a different length
        self.g = g

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        return torch.arange(self.lens[i]) + 10 * i, self.lens[i] % 2    # (sequence, label)


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

```text title="output"
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

- **`collate_fn` is the assembly instruction for a batch**. The default only stacks, so variable-length data needs your own. Pad to the longest in **this batch** rather than globally — that is the cheapest thing to do, and it is why samples of similar length should share a bucket;
- **Return the mask too**. The padded positions must not take part in attention or the loss, which is where the previous chapter's `masked_fill` comes in;
- **Shuffling has to be controllable**. Give the `DataLoader` a `generator` and two runs produce the same order. With `num_workers > 0` each worker also needs its own seed, or several processes may apply identical augmentations.

## The complete loop {#完整的训练循环}

```python title="loop.py" ci="loose"
"""一个完整的训练循环：五行主干，外加评估、学习率调度和梯度裁剪"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)

# a two-class toy dataset: two Gaussian blobs
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
    model.eval()                                              # switch dropout and BN out of training behaviour
    xb, yb = val.tensors
    logits = model(xb)
    acc = (logits.argmax(1) == yb).float().mean().item()
    model.train()                                             # remember to switch back
    return loss_fn(logits, yb).item(), acc


print("epoch   训练 loss   验证 loss   验证准确率   学习率")
for epoch in range(10):
    total = 0.0
    for xb, yb in loader:                                     # --- the five-line core ---
        loss = loss_fn(model(xb), yb)                         # 1. forward pass and loss
        opt.zero_grad(set_to_none=True)                       # 2. clear the previous step's gradients
        loss.backward()                                       # 3. backward
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 4. gradient clipping (optional, but recommended)
        opt.step()                                            # 5. update the parameters
        total += loss.item() * len(xb)                        # .item() synchronises; it is in the loop only so the log has a number
    sched.step()                                              # this scheduler steps per epoch; do not put it in the inner loop
    vl, acc = evaluate()
    print(f"{epoch + 1:5d}   {total / len(train):9.4f}   {vl:9.4f}   {acc:10.1%}   {sched.get_last_lr()[0]:.2e}")

print("\n几条容易写错的：")
print("- zero_grad 要在 backward **之前**；忘了它，梯度会一直累加")
print("- set_to_none=True 把梯度置空而不是填零，省一次写显存，也能让「没收到梯度的参数」暴露出来")
print("- loss.item() 会强制同步 GPU，别在每一步都调；要累计就先留在张量里，最后再取")
print("- scheduler.step() 按 epoch 调用；按步调度的（warmup）要放进内层循环，两种别混")
```

```text title="output"
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

Five lines, each corresponding to a class of error:

| Step | What it looks like wrong | Consequence |
| --- | --- | --- |
| forward and loss | the shapes broadcast wrongly (chapter 2) | no error, the loss just will not fall |
| `zero_grad` | forgotten, or placed after `backward` | gradients keep accumulating; the effective learning rate grows |
| `backward` | called on a non-scalar | raises; use `.sum()` or pass `gradient=` |
| clipping | placed after `step` | no clipping at all |
| `step` | no matching `zero_grad` | as above |

A few engineering details:

- **`loss.item()` forces a synchronisation**. On a GPU it has to wait for the step to really finish before it can hand you a number. Calling it every step breaks the CPU/GPU pipeline; accumulate in a tensor and call `.item()` once at the end of the epoch;
- **`set_to_none=True`** is now the default. Besides saving a memory write, it helps debugging: a parameter that received no gradient keeps `.grad` as None instead of a perfectly innocent-looking tensor of zeros;
- **Evaluation needs `eval()` and `no_grad()`**, and remember to switch back to `train()` afterwards. Above, the `@torch.no_grad()` decorator wraps the whole function, which is tidier than scattering `with` blocks;
- **Per-epoch or per-step scheduling** depends on which kind you use. Mixing them gives a learning-rate curve nothing like the one you have in mind — printing it is the cheapest way to check.

!!! interview "How to explain it"
    Describe the loop as five steps: **forward and loss → zero_grad → backward → (clip) → step**, and point out the ordering constraints (zero_grad before backward, step after, clipping in between). Then three engineering details: `loss.item()` synchronises the GPU, so do not call it every step for logging; `set_to_none=True` saves a memory write and makes "received no gradient" visible; evaluation needs `eval()` and `no_grad()` together and must switch back to `train()`. On the data side, talk about `collate_fn`: pad variable-length samples to the longest **in this batch** and return a mask, and bucketing by length saves a lot of padding compute.

## Exercises {#练习}

**1. Bucket by length.** Sort `ToyText`'s samples by length before batching, and measure how the padding fraction changes.

??? success "An approach"
    Use a `BatchSampler`, or sort and then chunk. Even on toy data the padding fraction drops visibly; on real variable-length text, bucketing often saves more than thirty percent of the compute. The cost is less shuffling, and the usual compromise is "random within a bucket, random across buckets".

**2. The clipping threshold.** Print `clip_grad_norm_`'s return value (the norm before clipping), watch its distribution over a few epochs, and then decide on a threshold.

??? success "An approach"
    The return value is the global norm **before** clipping. The usual approach is to run a few hundred steps with a very large threshold (effectively no clipping), see what magnitude the norm settles at, and set the threshold at one or two times that. A sudden spike in the norm is often the precursor of a loss spike.

**3. Add early stopping.** Record the `state_dict` at the lowest validation loss and restore it at the end.

??? success "An approach"
    Note that you need `{k: v.clone() for k, v in model.state_dict().items()}` — `state_dict()` returns **references** to the parameter tensors, so without the clone the continued training modifies them and your "best weights" change along with everything else. This one is well hidden.

## Summary {#小结}

- [x] A `Dataset` implements `__len__` and `__getitem__`; variable-length data needs a `collate_fn` that pads to the batch's longest and returns a mask.
- [x] The loop is five steps — forward → zero_grad → backward → clip → step — with hard ordering constraints.
- [x] `loss.item()` synchronises the GPU, so do not call it every step; `set_to_none=True` makes "received no gradient" visible.
- [x] Evaluation needs `eval()` + `no_grad()` and a switch back to `train()`; do not mix per-epoch and per-step schedulers.
