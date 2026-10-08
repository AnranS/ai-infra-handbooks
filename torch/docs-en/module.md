# 5. nn.Module: building a model

<p class="lead"><code>nn.Module</code> does one thing: <strong>it manages your parameters</strong>. A registered parameter can be found by the optimizer, travels with the model to the GPU, and goes into the <code>state_dict</code>; an unregistered one is nothing at all. This chapter settles the registration rules, buffers, <code>state_dict</code> and <code>train/eval</code>, and shows two traps beginners fall into.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How does `nn.Parameter` differ from an ordinary tensor?
    2. What is a buffer, and how does it differ from a parameter?
    3. What happens if you put submodules in a plain Python list?
    4. Does `model.eval()` turn off gradients?
    5. What is in a `state_dict()`?

??? success "Answers (try first, then expand to compare)"
    1. An `nn.Parameter` assigned to a Module attribute is **registered automatically**: it appears in `parameters()`, is updated by the optimizer, travels with `.to(device)`, and goes into the `state_dict`. Nothing happens for a plain tensor.
    2. A tensor that is part of the model's state but is not trained — BatchNorm's running statistics, a RoPE cos/sin table. Register it with `register_buffer`: it goes into the `state_dict` and travels with `.to()`, but is not in `parameters()`.
    3. They are not registered. The optimizer cannot find them, `.to(device)` does not move them, and they are not saved.
    4. No. It only switches the layers with two behaviours (dropout, batchnorm). Turning off gradients is `no_grad`'s job, and the two must be kept apart.
    5. Every registered parameter and buffer, keyed by the path built from the attribute names (`blocks.0.fc.weight`).

## Registration, buffers and state_dict {#注册buffer-与-state_dict}

```python title="module.py"
"""nn.Module：参数怎么注册、buffer 是什么、state_dict 长什么样、train 和 eval 有什么区别"""
import torch
import torch.nn as nn

torch.manual_seed(0)


class Block(nn.Module):
    def __init__(self, d_in, d_out, p=0.5):
        super().__init__()                                    # leave this out and registering parameters below fails outright
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)
        self.scale = nn.Parameter(torch.ones(d_out))          # a trainable parameter of our own
        self.register_buffer("calls", torch.zeros(1))         # state that is not trained but travels with state_dict

    def forward(self, x):
        self.calls += 1
        return self.drop(self.norm(self.fc(x)) * self.scale)


class Net(nn.Module):
    def __init__(self, d=4, n_block=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, d) for _ in range(n_block))   # use ModuleList, not a plain list
        self.head = nn.Linear(d, 2)

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)                                      # residual
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
        self.layers = [nn.Linear(2, 2) for _ in range(2)]     # how not to do it

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

net.apply(init)                                               # apply recurses into every submodule
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

```text title="output"
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

Things to remember:

- **Assignment is registration**. `self.fc = nn.Linear(...)` both attaches the submodule and makes its parameters part of this module's. Which is why `super().__init__()` has to be the first line: without it the registration machinery does not exist and the assignment raises;
- **A plain list of submodules is wasted code**. `Broken` above has 0 parameters. This trap is dangerous because the code runs and the loss does go down (the other layers are learning); those layers simply never update. When some part "seems not to be learning at all", look here first;
- **A buffer is not a parameter**: never updated by the optimizer, but saved and moved with the device. Use it for your own position-encoding tables and causal masks, not a plain attribute (which does not follow `.to()` and raises a device mismatch on the GPU);
- **`state_dict` is a flat dictionary** keyed by attribute path. Loading with `strict=True` (the default) demands both sides match exactly, and the return value tells you what is missing and what is unexpected — that return value is how you diagnose loading an old checkpoint into a changed model;
- **`train()` / `eval()` govern behaviour, not gradients**. Inference needs both: `model.eval()` and `with torch.no_grad()`. With only the first, dropout is off but memory still grows; with only the second, dropout is still dropping and the same input gives two different answers.

!!! tip "apply for initialisation"
    `model.apply(fn)` recurses `fn` into every submodule, so with an `isinstance` check you can say "every Linear like this, every LayerNorm like that". That is the standard approach when writing a model from scratch, and easier to maintain than scattering initialisation through every `__init__`.

!!! interview "How to explain it"
    On `nn.Module`: its core job is **parameter management** — assigning an `nn.Parameter` to an attribute registers it, so the optimizer can find it, `.to(device)` moves it and `state_dict` saves it. Then three easy mistakes: submodules belong in an `nn.ModuleList`, not a plain list (otherwise the parameter count is 0 and those layers never update); state that is not trained but must be saved goes in `register_buffer` (BN statistics, RoPE tables); and `eval()` only switches dropout/BN behaviour and **does not** turn off gradients, so inference needs `eval()` and `no_grad()` together.

## Exercises {#练习}

**1. Verify registration.** Replace the list in `Broken` with an `nn.ModuleList` and count the parameters again; then try `.to(torch.float64)` and see which layers changed under each version.

??? success "An approach"
    With `ModuleList` the parameters appear. The `.to()` difference is even more visible: the layers in a plain list keep their dtype, so the forward pass raises "expected scalar type Double but found Float" — a great many "device/type mismatch" errors start right here.

**2. Tied weights.** Make the output layer reuse the embedding's weight matrix, then count `parameters()` and see whether it is counted once or twice.

??? success "Answer"
    After `self.head.weight = self.embed.weight`, `parameters()` **deduplicates** and counts it once (it deduplicates by object identity). But `state_dict()` contains both keys pointing at the same tensor — the file is slightly larger, and loading it back still gives you sharing.

**3. Freezing and partial loading.** Load a checkpoint whose structure has changed with `strict=False`, then print the returned missing / unexpected keys and explain each.

??? success "An approach"
    `missing_keys` are in the model but not the file (newly added layers); `unexpected_keys` are the reverse (removed layers). The usual fine-tuning pattern is `strict=False` for the backbone, separate initialisation for the new head, and the return value written to the log — otherwise one misspelled name silently skips a weight and all you see is a mysteriously bad result.

## Summary {#小结}

- [x] An `nn.Parameter` assigned to an attribute is registered automatically; `super().__init__()` must come first.
- [x] Submodules go in `nn.ModuleList` / `nn.ModuleDict`; a plain list registers nothing.
- [x] State that is not trained but must be saved and moved goes in `register_buffer`.
- [x] `state_dict` is a flat dictionary; read the returned missing / unexpected keys to diagnose loading.
- [x] `eval()` only switches dropout/BN behaviour, not gradients — pair it with `no_grad` for inference.
