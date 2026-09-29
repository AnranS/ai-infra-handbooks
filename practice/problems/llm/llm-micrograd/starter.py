import math


class Value:
    def __init__(self, data, _prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_prev)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # TODO：__mul__、__pow__、__neg__、__sub__、__truediv__ 以及反向版本（__radd__ 等），exp、log、tanh、relu

    def backward(self):
        pass
