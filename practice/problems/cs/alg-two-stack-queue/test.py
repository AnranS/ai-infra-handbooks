from checker import check
from solution import MyQueue


def test_example():
    q = MyQueue()
    q.push(1)
    q.push(2)
    check(q.peek(), 1, "队首")
    check(q.pop(), 1, "先进先出")
    check(q.empty(), False, "还有一个")
    check(q.pop(), 2, "第二个")
    check(q.empty(), True, "空了")


def test_empty():
    q = MyQueue()
    check((q.pop(), q.peek(), q.empty()), (None, None, True), "空队列")


def test_interleaved():
    q = MyQueue()
    q.push(1)
    check(q.pop(), 1, "压一个弹一个")
    q.push(2)
    q.push(3)
    check(q.pop(), 2, "先进的先出")
    q.push(4)
    check(q.pop(), 3, "中途又压入的排在后面")
    check(q.pop(), 4, "最后一个")
    check(q.pop(), None, "空了")


def test_peek_does_not_pop():
    q = MyQueue()
    q.push(7)
    check(q.peek(), 7, "peek")
    check(q.peek(), 7, "peek 不改变队列")
    check(q.pop(), 7, "还能弹出")


def test_fifo_order():
    q = MyQueue()
    for i in range(100):
        q.push(i)
    check([q.pop() for _ in range(100)], list(range(100)), "顺序完全一致")


def test_many():
    import random
    from collections import deque
    rng = random.Random(1)
    q, ref = MyQueue(), deque()
    for _ in range(30000):
        if ref and rng.random() < 0.45:
            check(q.pop(), ref.popleft(), "出队一致")
        else:
            x = rng.randrange(1000)
            q.push(x)
            ref.append(x)
        check(q.empty(), not ref, "空状态一致")
