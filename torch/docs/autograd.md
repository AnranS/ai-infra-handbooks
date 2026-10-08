# （四）autograd 怎么用

<p class="lead">autograd 的接口小得出奇：`requires_grad`、`backward()`、`.grad`，再加上 `no_grad` 和 `detach`。难的是几个"为什么"——梯度为什么是累加的、为什么我的 `.grad` 是 None、为什么加一句 `+= 1` 就报错。这一章把这些都做成能跑的实验。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么训练循环里每一步都要 `zero_grad()`？什么时候反而不该清零？
    2. `no_grad` 和 `detach` 有什么区别？
    3. 哪三种情况下 `.grad` 会是 None？
    4. 为什么有的原地操作会让 backward 报错，有的不会？
    5. 同一个 loss 连着 `backward()` 两次会发生什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 梯度是**累加**到 `.grad` 上的，不清零就会把上一步的加进来。梯度累积（用多个小 batch 凑一个大 batch）恰恰是故意不清零。
    2. `no_grad` 是一个上下文，块内所有运算都不建图；`detach` 作用在单个张量上，把它从图里摘出来（但仍共享存储）。想关掉一整段用前者，想切断一条路径用后者。
    3. 没开 `requires_grad`；不是叶子张量（中间结果，除非 `retain_grad()`）；在 `no_grad` 里算出来的（根本没建图）。
    4. 看这个算子的反向需不需要用到被改掉的那个值。乘法的反向只要输入，改输出没关系；`sigmoid` 的反向要用自己的输出，改掉就炸。
    5. 报错。计算图在第一次 backward 之后就释放了，除非 `backward(retain_graph=True)`。

## 五个实验

```python title="grad.py"
"""autograd 怎么用：backward、梯度累加、no_grad 与 detach，以及"梯度为什么是 None" """
import torch

print("—— 最小的例子 ——")
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()                                            # y = x1² + x2²
y.backward()
print(f"y = {y.item():.1f}，dy/dx = 2x = {x.grad.tolist()}")
print("只有 requires_grad=True 的**叶子**张量才有 .grad")

print("\n—— 梯度是累加的，所以每一步都要清零 ——")
x.grad.zero_()
for _ in range(3):
    (x ** 2).sum().backward()
print("连着 backward 三次，梯度变成三倍：", x.grad.tolist())
print("训练循环里 optimizer.zero_grad() 就是干这个的；要做梯度累积，恰恰是**故意**不清零")

print("\n—— 计算图用完就释放 ——")
z = (x ** 2).sum()
z.backward()
try:
    z.backward()
except RuntimeError as e:
    print("再来一次会报错：", str(e).split(".")[0])
print("要多次反传同一张图，用 backward(retain_graph=True)；但多数时候这是写错了的信号")

print("\n—— no_grad 与 detach ——")
w = torch.ones(3, requires_grad=True)
with torch.no_grad():                                         # 块内不建图，省显存也省时间
    out = w * 2
print("no_grad 里算出来的结果 requires_grad =", out.requires_grad)
tmp = w * 2
d = tmp.detach()                                              # 从图上摘下来，但还是同一块存储
print("detach 之后 requires_grad =", d.requires_grad, "，和原张量共享存储 =", d.data_ptr() == tmp.data_ptr())
print("推理用 no_grad（或者更快的 inference_mode）；只想切断一条路径的梯度用 detach")

print("\n—— 三种「梯度是 None」 ——")
a = torch.ones(2)                                             # ① 根本没开 requires_grad
b = (a * 2).sum()
print("① 没开 requires_grad：", a.grad)
c = torch.ones(2, requires_grad=True)
mid = c * 2                                                   # ② 中间结果不是叶子
mid.sum().backward()
print("② 非叶子张量：mid.is_leaf =", mid.is_leaf, "，.grad 不会被填（想看就先 mid.retain_grad()）")
e = torch.ones(2, requires_grad=True)
with torch.no_grad():                                         # ③ 在 no_grad 里算的，没建图
    f = (e * 2).sum()
print("③ 在 no_grad 里算的：f.requires_grad =", f.requires_grad, "，backward 会直接报错")

print("\n—— 原地操作会破坏反向需要的值 ——")
u = torch.tensor([1.0, 2.0], requires_grad=True)
v = (u * 3).sigmoid()                                         # sigmoid 的反向要用它自己的输出
try:
    v += 1                                                    # 原地把那个输出改掉了
    v.sum().backward()
except RuntimeError as e:
    print("报错：", str(e).split(",")[0])
print("改成 v = v + 1 就好了。不是所有原地操作都会出事——乘法的反向只要输入，改输出没关系；")
print("但 sigmoid、exp 这类反向要用自己输出的就会炸，所以带下划线的方法（add_、relu_、scatter_）都要留心")

print("\n—— 手动验证一次梯度：和数值差分对比 ——")
def fn(t):
    return (t.sin() * t).sum()

t = torch.tensor([0.3, 1.2], requires_grad=True)
fn(t).backward()
eps = 1e-4
num = [((fn(t.detach() + eps * torch.eye(2)[i]) - fn(t.detach() - eps * torch.eye(2)[i])) / (2 * eps)).item()
       for i in range(2)]
print("autograd：", [f"{v:.5f}" for v in t.grad.tolist()], " 数值差分：", [f"{v:.5f}" for v in num])
print("写了自定义算子就该这么验一遍；torch.autograd.gradcheck 是它的正式版本")
```

```text title="输出"
—— 最小的例子 ——
y = 13.0，dy/dx = 2x = [4.0, 6.0]
只有 requires_grad=True 的**叶子**张量才有 .grad

—— 梯度是累加的，所以每一步都要清零 ——
连着 backward 三次，梯度变成三倍： [12.0, 18.0]
训练循环里 optimizer.zero_grad() 就是干这个的；要做梯度累积，恰恰是**故意**不清零

—— 计算图用完就释放 ——
再来一次会报错： Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
要多次反传同一张图，用 backward(retain_graph=True)；但多数时候这是写错了的信号

—— no_grad 与 detach ——
no_grad 里算出来的结果 requires_grad = False
detach 之后 requires_grad = False ，和原张量共享存储 = True
推理用 no_grad（或者更快的 inference_mode）；只想切断一条路径的梯度用 detach

—— 三种「梯度是 None」 ——
① 没开 requires_grad： None
② 非叶子张量：mid.is_leaf = False ，.grad 不会被填（想看就先 mid.retain_grad()）
③ 在 no_grad 里算的：f.requires_grad = False ，backward 会直接报错

—— 原地操作会破坏反向需要的值 ——
报错： one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [2]]
改成 v = v + 1 就好了。不是所有原地操作都会出事——乘法的反向只要输入，改输出没关系；
但 sigmoid、exp 这类反向要用自己输出的就会炸，所以带下划线的方法（add_、relu_、scatter_）都要留心

—— 手动验证一次梯度：和数值差分对比 ——
autograd： ['0.58212', '1.36687']  数值差分： ['0.58174', '1.36733']
写了自定义算子就该这么验一遍；torch.autograd.gradcheck 是它的正式版本
```

逐条说：

- **梯度累加不是 bug，是设计**。显存放不下大 batch 时，拆成几个小 batch 分别反向、梯度自然加起来，最后更新一次——这就是梯度累积。所以 PyTorch 不会替你清零，清零是 `optimizer.zero_grad()` 的事；
- **计算图用完就释放**：为了省显存，反向一次之后中间结果就扔了。如果你真的需要反向两次（比如二阶梯度、GAN 的某些写法），传 `retain_graph=True`；但大多数时候看到这个报错，是代码写错了（比如把 loss 存起来后面又用了一次）；
- **`no_grad` 省的是显存和时间**：不建图就不用保存中间结果。推理时一定要加，否则显存会随着序列长度线性涨；
- **三种 None**：没开 `requires_grad`、不是叶子、在 `no_grad` 里算的。调试时按这三条依次排查，基本都能定位；
- **原地操作要看反向需要什么**。这一点很多教程说得太绝对（"不要用原地操作"）。真实的规则是：反向里要用到的那个值被改了才会出事。`x.add_(1)` 对一个刚算完乘法的张量没问题，对一个刚算完 `sigmoid` 的张量就会报错。拿不准就别用，热路径上想省显存再回来仔细算。

!!! tip "自己写了算子，就和数值差分比一次"
    把函数在某一点上下各挪一个 $\varepsilon$，用 $(f(x+\varepsilon) - f(x-\varepsilon)) / 2\varepsilon$ 近似导数，和 autograd 的结果比。上面那个例子两者在小数点后三位一致，剩下的差是差分本身的截断误差。正式一点用 `torch.autograd.gradcheck`（它会用 float64 做这件事）。

!!! interview "怎么讲清楚"
    讲 autograd 的使用：**梯度是累加的**，所以每步要 `zero_grad()`，而梯度累积就是故意不清零；**计算图反向一次就释放**，要重复反向得 `retain_graph=True`；**推理用 `no_grad`**（不建图，省显存和时间），**切断单条路径用 `detach`**。再讲"梯度是 None"的三种原因（没开 requires_grad / 不是叶子 / 在 no_grad 里算的）。最后补一个有深度的点：原地操作不是一律禁止，关键看这个算子的反向要不要用到被改掉的值——`sigmoid`、`exp` 这类保存自己输出的会炸，乘法不会。

## 练习

**1. 梯度累积。** 把一个 batch 拆成 4 份分别反向、最后更新一次，验证得到的梯度和一次性算整个 batch 相同。loss 要怎么缩放？

??? success "参考答案"
    每一份的 loss 要除以份数（因为默认的 loss 是对 batch 求平均），累加起来才等于整个 batch 的平均梯度。`loss = loss_fn(...) / accum_steps`，然后每 `accum_steps` 步调用一次 `optimizer.step()` 和 `zero_grad()`。

**2. 冻结一部分参数。** 两种做法——把参数的 `requires_grad` 设成 False，或者在前向里 `detach()`——分别有什么效果？优化器那边还要做什么？

??? success "参考思路"
    `requires_grad=False` 让这些参数不再收到梯度，但它们仍在优化器的参数组里（有 weight decay 的话还会被衰减，所以最好建优化器时就排除掉）。`detach()` 切断的是**数据流**，连带着上游所有参数都收不到梯度。微调时冻结底层一般用前者。

**3. 查一个 None。** 写一段会让 `.grad` 是 None 的代码，然后用本章的三条依次定位是哪一种。

??? success "参考思路"
    最常见的其实是第二种：想看某个中间激活的梯度，结果发现是 None。加一句 `h.retain_grad()` 即可。生产代码里更常见的是第一种——参数在 `no_grad` 块里被重建了，于是 `requires_grad` 丢了。

## 小结

- [x] 梯度累加到 `.grad` 上，所以每步要 `zero_grad()`；梯度累积是故意不清零。
- [x] 计算图反向一次就释放，重复反向要 `retain_graph=True`（通常说明写错了）。
- [x] `no_grad` 关掉整段的建图，`detach` 切断单个张量；`.grad` 是 None 的三种原因要会排查。
- [x] 原地操作是否安全，取决于这个算子的反向要不要用到被改掉的值。
