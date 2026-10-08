import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
y = (x * w).sum()


def walk(fn, depth=0):
    if fn is None:
        return
    print("  " * depth + type(fn).__name__)
    for nxt, _ in fn.next_functions:
        walk(nxt, depth + 1)


walk(y.grad_fn)
mul = y.grad_fn.next_functions[0][0]
print("MulBackward0 保存了：", sorted(a for a in dir(mul) if a.startswith("_saved")))
y.backward()
print("x.grad =", x.grad.tolist(), "w.grad =", w.grad.tolist())
