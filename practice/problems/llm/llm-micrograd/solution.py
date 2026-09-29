import math


class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "指数只支持普通数字"
        out = Value(self.data ** k, (self,))

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._wrap(other) - self

    def __rtruediv__(self, other):
        return self._wrap(other) / self

    def backward(self):
        order, visited = [], set()
        stack = [(self, False)]
        while stack:                              # 迭代式 DFS 求拓扑序
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._prev:
                if id(p) not in visited:
                    stack.append((p, False))
        self.grad = 1.0
        for node in reversed(order):
            node._backward()
