from checker import check
from solution import compose, make_counter, make_multipliers


def test_example_counter():
    c = make_counter(10, 5)
    check([c(), c(), c()], [10, 15, 20], "连续三次调用")


def test_counter_defaults_and_independent():
    a, b = make_counter(), make_counter()
    check([a(), a(), a()], [0, 1, 2], "默认计数器")
    check(b(), 0, "另一个计数器从头开始")


def test_example_compose():
    check(compose(str, abs, int)("-42"), "42", 'compose(str, abs, int)("-42")')


def test_compose_identity_and_order():
    check(compose()(5), 5, "compose() 是恒等函数")
    add1 = lambda x: x + 1  # noqa: E731
    dbl = lambda x: x * 2  # noqa: E731
    check(compose(add1, dbl)(3), 7, "compose(add1, dbl)(3) = add1(dbl(3))")
    check(compose(dbl, add1)(3), 8, "compose(dbl, add1)(3) = dbl(add1(3))")


def test_example_multipliers():
    check([f(3) for f in make_multipliers(4)], [0, 3, 6, 9], "make_multipliers(4)")


def test_multipliers_single():
    fs = make_multipliers(1)
    check(len(fs), 1, "函数个数")
    check(fs[0](100), 0, "第 0 个函数")
