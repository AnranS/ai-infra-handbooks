import torch


def mid_grad(x):
    x = x.detach().clone().requires_grad_(True)
    h = x.sigmoid()
    h.retain_grad()                       # h 不是叶子：不留一手就拿不到它的梯度
    (h * h).sum().backward()
    return h.grad, x.grad


def accumulated_grad(w, xs, ys):
    if w.grad is not None:
        w.grad = None
    total = sum(x.shape[0] for x in xs)
    for x, y in zip(xs, ys):
        loss = ((x @ w - y) ** 2).mean() * x.shape[0] / total   # 按样本数加权，加起来才等于大批
        loss.backward()
    return w.grad


def freeze_count(linear_layers, trainable_prefix):
    n = 0
    for i, layer in enumerate(linear_layers):
        keep = f"layer{i}".startswith(trainable_prefix)
        for p in layer.parameters():
            p.requires_grad_(keep)
            n += keep
    return n
