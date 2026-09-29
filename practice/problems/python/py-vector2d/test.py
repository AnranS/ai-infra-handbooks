from checker import check, raises
from solution import Vector


def test_example():
    """repr、相等、abs、bool"""
    check(repr(Vector(1, 2)), "Vector(1, 2)", "repr(Vector(1, 2))")
    check(repr(Vector(1.5, -2)), "Vector(1.5, -2)", "repr(Vector(1.5, -2))")
    assert Vector(1, 2) == Vector(1, 2), "Vector(1, 2) == Vector(1, 2) 应该为 True"
    assert Vector(1, 2) != Vector(2, 1), "Vector(1, 2) != Vector(2, 1) 应该为 True"
    check(abs(Vector(3, 4)), 5.0, "abs(Vector(3, 4))")
    assert not Vector(0, 0), "零向量应该为假"
    assert Vector(0, 1), "非零向量应该为真"


def test_arithmetic():
    """加、减、数乘、取反"""
    v, w = Vector(1, 2), Vector(10, 20)
    check(v + w, Vector(11, 22), "v + w")
    check(w - v, Vector(9, 18), "w - v")
    check(v * 3, Vector(3, 6), "v * 3")
    check(3 * v, Vector(3, 6), "3 * v")
    check(-v, Vector(-1, -2), "-v")
    assert isinstance(v + w, Vector)


def test_hashable():
    """可以放进 set、当 dict 的键"""
    s = {Vector(1, 2), Vector(1, 2), Vector(2, 1)}
    check(len(s), 2, "集合里不同向量的个数")
    d = {Vector(0, 0): "origin"}
    check(d[Vector(0, 0)], "origin", "用相等的向量查字典")


def test_unpack_and_iter():
    x, y = Vector(7, 8)
    check((x, y), (7, 8), "解包")
    check(list(Vector(1, 2)), [1, 2], "list(Vector(1, 2))")


def test_immutable():
    """x、y 只读"""
    v = Vector(1, 2)
    check(v.x, 1, "v.x")
    check(v.y, 2, "v.y")
    with raises(AttributeError, "给 v.x 赋值"):
        v.x = 5


def test_eq_other_types():
    """和其他类型比较返回 NotImplemented"""
    check(Vector(1, 2).__eq__((1, 2)), NotImplemented, "Vector(1, 2).__eq__((1, 2))")
    assert Vector(1, 2) != (1, 2)
    with raises(TypeError, "Vector(1, 2) + 1"):
        Vector(1, 2) + 1
