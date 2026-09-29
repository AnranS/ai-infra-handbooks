import numpy as np


def unbroadcast(grad, shape):
    """把广播后的梯度求和还原成 shape"""
    grad = np.asarray(grad)
    while grad.ndim > len(shape):                      # 广播时加在前面的维度：求和去掉
        grad = grad.sum(0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:              # 长度为 1 被广播的维度：求和并保留
            grad = grad.sum(i, keepdims=True)
    return grad


class Node:
    """反向节点：inputs 是前向的输入（Tensor），backward(grad) 返回对每个输入的梯度（与 inputs 一一对应）"""

    def __init__(self, *inputs):
        self.inputs = list(inputs)

    def backward(self, grad):
        raise NotImplementedError


class AddBackward(Node):
    def backward(self, grad):
        a, b = self.inputs
        return unbroadcast(grad, a.shape), unbroadcast(grad, b.shape)


class MulBackward(Node):
    def backward(self, grad):
        a, b = self.inputs
        return unbroadcast(grad * b.data, a.shape), unbroadcast(grad * a.data, b.shape)


class MatMulBackward(Node):                         # 只处理二维矩阵
    def backward(self, grad):
        a, b = self.inputs
        return grad @ b.data.T, a.data.T @ grad


class SumBackward(Node):
    def backward(self, grad):
        (a,) = self.inputs
        return (np.broadcast_to(grad, a.shape).copy(),)


class ReluBackward(Node):
    def backward(self, grad):
        (a,) = self.inputs
        return (grad * (a.data > 0),)


def _apply(node_cls, fn, *args):
    args = [a if isinstance(a, Tensor) else Tensor(a) for a in args]
    out = Tensor(fn(*[a.data for a in args]))
    if any(a.requires_grad for a in args):             # 有输入需要梯度时才记录反向节点
        out.grad_fn = node_cls(*args)
        out.requires_grad = True
    return out


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad_fn = None
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    def __add__(self, other):
        return _apply(AddBackward, np.add, self, other)

    __radd__ = __add__

    def __mul__(self, other):
        return _apply(MulBackward, np.multiply, self, other)

    __rmul__ = __mul__

    def __matmul__(self, other):
        return _apply(MatMulBackward, np.matmul, self, other)

    def sum(self):
        return _apply(SumBackward, np.sum, self)

    def relu(self):
        return _apply(ReluBackward, lambda x: np.maximum(x, 0), self)

    def backward(self, grad=None):
        grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        if self.grad_fn is None:
            if self.requires_grad:
                self.grad = grad.copy() if self.grad is None else self.grad + grad
            return
        # 1. 数出每个节点的入边：它的输出被几个节点当作输入用了
        deps, stack, seen = {}, [self.grad_fn], {id(self.grad_fn)}
        while stack:
            node = stack.pop()
            for t in node.inputs:
                if t.grad_fn is not None:
                    deps[id(t.grad_fn)] = deps.get(id(t.grad_fn), 0) + 1
                    if id(t.grad_fn) not in seen:
                        seen.add(id(t.grad_fn))
                        stack.append(t.grad_fn)
        # 2. 入边全部到齐的节点才执行；同一个节点收到的多份梯度先加起来（InputBuffer）
        buffers, ready = {id(self.grad_fn): grad}, [self.grad_fn]
        while ready:
            node = ready.pop()
            g = buffers.pop(id(node))
            for t, gi in zip(node.inputs, node.backward(g)):
                if gi is None or not t.requires_grad:
                    continue
                if t.grad_fn is None:                  # 叶子：AccumulateGrad
                    t.grad = np.array(gi, dtype=np.float64) if t.grad is None else t.grad + gi
                    continue
                k = id(t.grad_fn)
                buffers[k] = gi if k not in buffers else buffers[k] + gi
                deps[k] -= 1
                if deps[k] == 0:
                    ready.append(t.grad_fn)
