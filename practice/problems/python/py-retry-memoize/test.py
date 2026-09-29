from checker import check, raises
from solution import memoize, retry


def test_example_retry_succeeds():
    """前两次失败、第三次成功"""
    calls = []

    @retry(3, exceptions=(ConnectionError,))
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    check(flaky(), "ok", "flaky()")
    check(len(calls), 3, "调用次数")


def test_retry_gives_up():
    calls = []

    @retry(2, exceptions=(ValueError,))
    def always():
        calls.append(1)
        raise ValueError("bad")

    with raises(ValueError, "重试用完之后"):
        always()
    check(len(calls), 2, "总调用次数")


def test_retry_other_exception_not_retried():
    calls = []

    @retry(5, exceptions=(ConnectionError,))
    def f():
        calls.append(1)
        raise KeyError("x")

    with raises(KeyError, "不在 exceptions 里的异常"):
        f()
    check(len(calls), 1, "不应该重试")


def test_retry_on_retry_and_wraps():
    seen = []

    @retry(3, on_retry=lambda n, e: seen.append((n, str(e))))
    def g(x):
        """文档"""
        if len(seen) < 2:
            raise RuntimeError(f"fail{len(seen)}")
        return x * 2

    check(g(21), 42, "g(21)")
    check(seen, [(1, "fail0"), (2, "fail1")], "on_retry 收到的参数")
    check(g.__name__, "g", "__name__")
    check(g.__doc__, "文档", "__doc__")


def test_example_memoize():
    calls = []

    @memoize()
    def square(x):
        calls.append(x)
        return x * x

    check([square(3), square(3), square(4)], [9, 9, 16])
    check(calls, [3, 4], "真正执行的调用")
    check(square.cache_info(), (1, 2, 2), "cache_info()")


def test_memoize_kwargs_and_clear():
    calls = []

    @memoize()
    def f(a, b=0):
        calls.append((a, b))
        return a + b

    f(1, b=2)
    f(1, b=2)
    f(1, 2)
    check(len(calls), 2, "f(1, b=2) 缓存命中，f(1, 2) 算作新调用")
    f.cache_clear()
    check(f.cache_info(), (0, 0, 0), "cache_clear() 之后")
    f(1, b=2)
    check(len(calls), 3, "清空后重新计算")


def test_memoize_lru():
    calls = []

    @memoize(maxsize=2)
    def f(x):
        calls.append(x)
        return x

    f(1), f(2), f(1)      # 1 最近用过
    f(3)                  # 淘汰 2
    f(1)                  # 命中
    f(2)                  # 重新计算
    check(calls, [1, 2, 3, 2], "真正执行的调用")
    check(f.cache_info()[2], 2, "缓存大小")


def test_memoize_recursive_fib():
    @memoize(maxsize=None)
    def fib(n):
        return n if n < 2 else fib(n - 1) + fib(n - 2)

    check(fib(200), 280571172992510140037611932413038677189525, "fib(200)")
    check(fib.__name__, "fib", "__name__")
