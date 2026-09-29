import numpy as np


def unbroadcast(grad, shape):
    """把广播后的梯度求和还原成 shape"""
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
        """递归写法：结果对，但共享的子图会被重复计算"""
        grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=np.float64)
        if self.grad_fn is None:
            if self.requires_grad:
                self.grad = grad.copy() if self.grad is None else self.grad + grad
            return
        for t, g in zip(self.grad_fn.inputs, self.grad_fn.backward(grad)):
            if t.requires_grad:
                t.backward(g)
