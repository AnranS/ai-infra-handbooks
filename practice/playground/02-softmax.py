"""numpy：广播与数值稳定的 softmax"""
import numpy as np

x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])


def naive_softmax(x):
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def stable_softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))          # 减去每行的最大值：结果不变，但不会溢出
    return e / e.sum(-1, keepdims=True)


with np.errstate(over="ignore", invalid="ignore"):
    print("朴素写法：\n", naive_softmax(x))
print("数值稳定的写法：\n", stable_softmax(x))
print("每行之和：", stable_softmax(x).sum(-1))
