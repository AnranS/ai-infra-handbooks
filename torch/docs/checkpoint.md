# （七）保存、加载与复现

<p class="lead">"训练中断了，接着跑"听起来理所当然，但只存权重是接不上的。这一章用一个能跑的实验说明 checkpoint 里到底要存什么，顺带讲清 <code>torch.load</code> 的 <code>weights_only</code> 和随机数种子的管理。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 只保存 `state_dict()` 续训，和没中断过相比结果会一样吗？为什么？
    2. 一个能精确续训的 checkpoint 至少要存哪些东西？
    3. `torch.load` 的 `weights_only` 是什么？为什么不该随手关掉？
    4. `torch.manual_seed(0)` 之后，另一处代码调了一次 `randn`，会影响你的结果吗？
    5. 怎么让结果完全确定？代价是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 不一样。Adam 的一阶、二阶矩和步数都丢了，等于重新 warmup 一遍。
    2. 模型参数、优化器状态、已经走了多少步、学习率调度器的状态、数据读到了哪里，以及随机数生成器的状态。
    3. 默认 `True`，只还原张量和白名单类型，不执行文件里的任意代码。关掉它等于允许反序列化时执行任意代码，加载来路不明的权重是真实的攻击面。
    4. 会。全局种子是一个共享的状态，任何一次采样都会把它推着走。要隔离就各用各的 `Generator`。
    5. `torch.use_deterministic_algorithms(True)`，再固定 cuBLAS 的工作区。代价是变慢，而且有些算子根本没有确定性实现（会直接报错）。

## 只存权重，续训就对不上

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
ref = steps(model, opt, 10, gen)                              # 不中断，再走 10 步

m1, o1 = make()                                               # ① 只恢复权重
m1.load_state_dict(only_weights)
g1 = torch.Generator().manual_seed(1)
loss1 = steps(m1, o1, 10, g1)

ck = torch.load("ckpt.pt", weights_only=False)                # ② 全部恢复
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

```text title="输出"
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

这个实验说明了一件常被忽略的事：**权重只是状态的一部分**。

- **优化器状态**：Adam 为每个参数维护一阶矩和二阶矩，还有一个步数 $t$ 用来做偏差修正。丢了它，恢复后的前几十步相当于重新 warmup，损失曲线上会出现一个可见的台阶；
- **随机数状态**：dropout 丢哪些、数据怎么 shuffle，都取决于它。要做到"续训和不中断逐位一致"，这个也得存；
- **数据位置**：真实训练里数据是流式读的，得记下读到第几个分片、第几条；
- **调度器状态**：余弦调度要知道走到第几步了；
- **GradScaler**：用 fp16 混合精度时，scale 因子也是状态。

!!! tip "存 state_dict，别存整个模型"
    `torch.save(model, path)` 会把整个对象用 pickle 存下来，加载时要求代码结构完全一致（类名、模块路径都不能变），而且必须关掉 `weights_only`。标准做法永远是存 `state_dict`，加载时先构造模型再 `load_state_dict`。

!!! interview "怎么讲清楚"
    讲"怎么做到续训和不中断一致"：checkpoint 不能只存权重，至少还要**优化器状态**（Adam 的两个矩和步数）、**调度器状态**、**数据读到哪里**、**随机数状态**，混合精度时还有 **GradScaler**。再讲一条安全实践：永远存 `state_dict` 而不是整个模型对象，`torch.load` 的 `weights_only` 默认开着就别关——关掉等于允许反序列化执行任意代码。最后补复现：全局种子是共享状态，任何一处采样都会推动它，要隔离就各用各的 `Generator`；完全确定还要 `use_deterministic_algorithms`，代价是变慢且部分算子不支持。

## 练习

**1. 补齐 checkpoint。** 给本章的 `full` 字典再加上调度器状态，验证带 warmup 的余弦调度在续训后学习率是接着走的，而不是从头开始。

??? success "参考思路"
    `sched.state_dict()` / `sched.load_state_dict()`。不存它的话，恢复后学习率会从第 0 步重新开始——曲线上是一个很显眼的回跳，而且会把刚收敛的模型打乱。

**2. 分片保存。** 模型很大时，`torch.save` 一次写几十 GB 很慢。查一下 `safetensors` 的分片格式，说明它相比 pickle 的两个好处。

??? success "参考答案"
    一是**安全**：纯数据格式，加载时不执行任何代码；二是**可以 mmap 按需读**，不用把整个文件读进内存，多进程加载同一份权重时还能共享页缓存。分布式训练里还有第三个好处：每个 rank 只写自己那一片。

**3. 查一次不可复现。** 故意不存 `rng` 状态，续训后对比 loss；再把 dropout 去掉，看差异是否消失。

??? success "参考思路"
    去掉 dropout 之后，随机性只剩数据顺序；如果数据加载器也用固定 generator，结果就又一致了。这个排除法能帮你定位"到底是哪一处随机性"在影响复现。

## 小结

- [x] 只存权重续训接不上：Adam 的矩和步数丢了，等于重新 warmup。
- [x] 完整的 checkpoint = 参数 + 优化器 + 调度器 + 数据位置 + 随机数状态（+ GradScaler）。
- [x] 永远存 `state_dict`，`torch.load` 的 `weights_only` 默认开着就别关。
- [x] 全局种子是共享状态；要隔离用各自的 `Generator`，要完全确定还得关掉非确定性算子。
