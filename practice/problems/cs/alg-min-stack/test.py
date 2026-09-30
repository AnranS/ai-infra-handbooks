from checker import check
from solution import MinStack


def test_example():
    s = MinStack()
    s.push(-2)
    s.push(0)
    s.push(-3)
    check(s.get_min(), -3, "当前最小")
    check(s.pop(), -3, "弹出栈顶")
    check(s.get_min(), -2, "最小值要能恢复")
    check(s.top(), 0, "栈顶")


def test_empty():
    s = MinStack()
    check((s.pop(), s.top(), s.get_min()), (None, None, None), "空栈")


def test_duplicate_min():
    s = MinStack()
    for x in (2, 1, 1, 3):
        s.push(x)
    check(s.get_min(), 1, "最小值出现两次")
    s.pop()                                    # 弹 3
    check(s.get_min(), 1, "还有一个 1")
    s.pop()                                    # 弹一个 1
    check(s.get_min(), 1, "另一个 1 还在")
    s.pop()                                    # 弹另一个 1
    check(s.get_min(), 2, "只剩 2")


def test_increasing():
    s = MinStack()
    for x in range(5):
        s.push(x)
        check(s.get_min(), 0, "最小值一直是 0")
    for x in range(4, -1, -1):
        check(s.pop(), x, "依次弹出")


def test_all_pop():
    s = MinStack()
    s.push(1)
    s.pop()
    check(s.get_min(), None, "全部弹完后为空")
    s.push(5)
    check(s.get_min(), 5, "重新使用")


def test_many():
    import random
    rng = random.Random(0)
    s = MinStack()
    ref = []
    for _ in range(20000):
        if ref and rng.random() < 0.4:
            check(s.pop(), ref.pop(), "弹出值一致")
        else:
            x = rng.randrange(1000)
            s.push(x)
            ref.append(x)
        check(s.get_min(), min(ref) if ref else None, "最小值始终一致")
