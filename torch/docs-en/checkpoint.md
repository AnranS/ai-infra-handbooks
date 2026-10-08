# 7. Saving, loading and reproducing

<p class="lead">"Training was interrupted, just carry on" sounds obvious, but saving the weights alone does not carry on. This chapter uses a runnable experiment to show what really belongs in a checkpoint, and settles <code>torch.load</code>'s <code>weights_only</code> and the management of random seeds.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. If you only save `state_dict()` and resume, will the result match an uninterrupted run? Why?
    2. What does a checkpoint need, at minimum, to resume exactly?
    3. What is `torch.load`'s `weights_only`, and why should you not casually turn it off?
    4. After `torch.manual_seed(0)`, if some other code calls `randn` once, does it affect your result?
    5. How do you make the result fully deterministic, and at what cost?

??? success "Answers (try first, then expand to compare)"
    1. No. Adam's first and second moments and its step count are gone, which amounts to warming up all over again.
    2. The parameters, the optimizer state, how many steps have been taken, the scheduler state, where the data reader is, and the random number generator state.
    3. It defaults to `True` and restores only tensors and a small allow-list, executing no code from the file. Turning it off allows arbitrary code execution during deserialization, which is a real attack surface when loading weights of unknown origin.
    4. Yes. The global seed is one shared state and every draw advances it. Use separate `Generator`s to isolate.
    5. `torch.use_deterministic_algorithms(True)` plus a fixed cuBLAS workspace. It is slower, and some operators have no deterministic implementation at all (they raise).

## Save the weights only and resuming does not line up {#只存权重续训就对不上}

```python title="save.py" ci="loose"
"""保存、加载与复现：checkpoint 里到底要存什么，以及种子怎么管"""
import torch
import torch.nn as nn


def make():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def steps(model, opt, k, gen):
    for _ in range(k):
        x = torch.randn(16, 4, generator=gen)
        loss = (model(x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return loss.item()


print("—— 只存权重，续训就对不上 ——")
model, opt = make()
gen = torch.Generator().manual_seed(1)
steps(model, opt, 20, gen)
only_weights = {k: v.clone() for k, v in model.state_dict().items()}
full = {"model": model.state_dict(), "optimizer": opt.state_dict(),
        "step": 20, "rng": torch.get_rng_state(), "data_rng": gen.get_state()}
torch.save(full, "ckpt.pt")
ref = steps(model, opt, 10, gen)                              # no interruption: just run 10 more steps

m1, o1 = make()                                               # (1) restore the weights only
m1.load_state_dict(only_weights)
g1 = torch.Generator().manual_seed(1)
loss1 = steps(m1, o1, 10, g1)

ck = torch.load("ckpt.pt", weights_only=False)                # (2) restore everything
m2, o2 = make()
m2.load_state_dict(ck["model"])
o2.load_state_dict(ck["optimizer"])
torch.set_rng_state(ck["rng"])
g2 = torch.Generator()
g2.set_state(ck["data_rng"])
loss2 = steps(m2, o2, 10, g2)

print(f"不中断走完 30 步的 loss：{ref:.6f}")
print(f"① 只恢复权重：          {loss1:.6f}  一致：{loss1 == ref}")
print(f"② 恢复权重+优化器+随机数：{loss2:.6f}  一致：{loss2 == ref}")
print("差别来自 Adam 的一阶、二阶矩和步数：优化器状态丢了，等于重新 warmup 一遍")

print("\n—— checkpoint 的最小集合 ——")
for k in full:
    print(f"  {k}")
print("再加上：学习率调度器的状态、数据读到了第几条、以及（混合精度时）GradScaler 的状态")

print("\n—— weights_only ——")
print("torch.load 默认 weights_only=True：只还原张量和一小撮白名单类型，不执行文件里的任意代码")
print("我们这份 checkpoint 全是张量和整数，默认模式也读得出来：", list(torch.load("ckpt.pt")))
print("但只要存了自定义的类（比如整个 model 对象、argparse 的 Namespace），默认模式就会拒绝，")
print("这时候的正确做法是**改成只存 state_dict**，而不是顺手把 weights_only 关掉——")
print("关掉它等于允许反序列化时执行任意代码，加载来路不明的权重时这是真实的攻击面")

print("\n—— 种子 ——")
torch.manual_seed(0)
a = torch.randn(3)
torch.manual_seed(0)
print("同一个种子，两次结果一样：", torch.equal(a, torch.randn(3)))
print("全局种子会被任何一处 randn 推着走；要隔离就各用各的 Generator（上一章 DataLoader 就是这么做的）")
print("想完全确定：torch.use_deterministic_algorithms(True) + 固定 cuBLAS 工作区，代价是变慢，而且有些算子没有确定性实现")
```

```text title="output"
—— 只存权重，续训就对不上 ——
不中断走完 30 步的 loss：0.001702
① 只恢复权重：          0.001545  一致：False
② 恢复权重+优化器+随机数：0.001702  一致：True
差别来自 Adam 的一阶、二阶矩和步数：优化器状态丢了，等于重新 warmup 一遍

—— checkpoint 的最小集合 ——
  model
  optimizer
  step
  rng
  data_rng
再加上：学习率调度器的状态、数据读到了第几条、以及（混合精度时）GradScaler 的状态

—— weights_only ——
torch.load 默认 weights_only=True：只还原张量和一小撮白名单类型，不执行文件里的任意代码
我们这份 checkpoint 全是张量和整数，默认模式也读得出来： ['model', 'optimizer', 'step', 'rng', 'data_rng']
但只要存了自定义的类（比如整个 model 对象、argparse 的 Namespace），默认模式就会拒绝，
这时候的正确做法是**改成只存 state_dict**，而不是顺手把 weights_only 关掉——
关掉它等于允许反序列化时执行任意代码，加载来路不明的权重时这是真实的攻击面

—— 种子 ——
同一个种子，两次结果一样： True
全局种子会被任何一处 randn 推着走；要隔离就各用各的 Generator（上一章 DataLoader 就是这么做的）
想完全确定：torch.use_deterministic_algorithms(True) + 固定 cuBLAS 工作区，代价是变慢，而且有些算子没有确定性实现
```

The experiment shows something easy to overlook: **the weights are only part of the state**.

- **Optimizer state**: Adam keeps a first and a second moment per parameter, plus a step count $t$ for bias correction. Lose it and the first dozens of steps after restoring amount to warming up again, which shows as a visible step in the loss curve;
- **Random number state**: which units dropout drops and how the data is shuffled both come from it. To resume bit-for-bit, this has to be saved too;
- **Where the data is**: real training streams the data, so you have to record which shard and which record;
- **Scheduler state**: a cosine schedule needs to know which step it is on;
- **GradScaler**: with fp16 mixed precision, the scale factor is state as well.

!!! tip "Save the state_dict, not the model"
    `torch.save(model, path)` pickles the whole object, requires an identical code structure to load (class names and module paths included), and forces `weights_only` off. The standard approach is always to save the `state_dict`, construct the model on load, and then `load_state_dict`.

!!! interview "How to explain it"
    On "how to make resuming match an uninterrupted run": a checkpoint cannot be only the weights — it also needs the **optimizer state** (Adam's two moments and its step count), the **scheduler state**, **where the data reader is** and the **random number state**, plus the **GradScaler** under mixed precision. Then a safety practice: always save a `state_dict` rather than the whole model object, and leave `torch.load`'s `weights_only` on — turning it off allows arbitrary code execution during deserialization. Finish with reproducibility: the global seed is shared state that every draw advances, so isolate with separate `Generator`s; full determinism additionally needs `use_deterministic_algorithms`, which costs speed and is unsupported by some operators.

## Exercises {#练习}

**1. Complete the checkpoint.** Add the scheduler state to this chapter's `full` dictionary and verify that a cosine schedule with warmup continues from where it was rather than starting over.

??? success "An approach"
    `sched.state_dict()` / `sched.load_state_dict()`. Without it the learning rate restarts from step 0 after restoring — a very visible jump in the curve, and enough to disturb a model that had just converged.

**2. Sharded saving.** For a large model, one `torch.save` of tens of gigabytes is slow. Look up `safetensors`' sharded format and give two advantages over pickle.

??? success "Answer"
    First **safety**: a pure data format that executes no code on load. Second, it can be **mmapped and read on demand**, so the whole file need not be read into memory, and several processes loading the same weights share the page cache. In distributed training there is a third: each rank writes only its own shard.

**3. Track down one irreproducibility.** Deliberately omit the `rng` state and compare the loss after resuming; then remove dropout and see whether the difference disappears.

??? success "An approach"
    Without dropout the only randomness left is the data order; if the loader also uses a fixed generator, the results agree again. This elimination approach is how you find out which source of randomness is affecting reproducibility.

## Summary {#小结}

- [x] Saving only the weights does not resume: Adam's moments and step count are gone, which is warming up again.
- [x] A complete checkpoint = parameters + optimizer + scheduler + data position + RNG state (+ GradScaler).
- [x] Always save a `state_dict`, and leave `torch.load`'s `weights_only` on.
- [x] The global seed is shared state; isolate with separate `Generator`s, and full determinism also needs the non-deterministic operators turned off.
