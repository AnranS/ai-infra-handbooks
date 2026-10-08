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
