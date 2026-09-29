from checker import check, raises
from solution import Backend


def setup():
    Backend.clear_registry()


def test_example():
    setup()

    class FlashInfer(Backend, name="flashinfer"):
        def forward(self, x):
            return x + 1

    class Torch(Backend, name="torch", priority=10):
        def forward(self, x):
            return x * 2

    check(Backend.create("torch").forward(3), 6, 'Backend.create("torch").forward(3)')
    check(Backend.create("flashinfer").forward(3), 4)
    check(Backend.available(), ["torch", "flashinfer"], "Backend.available()")
    check(Torch.name, "torch", "Torch.name")


def test_order_by_priority_then_name():
    setup()

    class B(Backend, name="b"):
        pass

    class A(Backend, name="a"):
        pass

    class C(Backend, name="c", priority=5):
        pass

    check(Backend.available(), ["c", "a", "b"], "available()")


def test_duplicate_name():
    setup()

    class X(Backend, name="x"):
        pass

    with raises(ValueError, "重复注册 x"):
        class Y(Backend, name="x"):
            pass


def test_abstract_intermediate():
    """没有 name 的中间类不注册，它的子类可以注册"""
    setup()

    class Base(Backend):
        def forward(self, x):
            return -x

    class Impl(Base, name="impl"):
        pass

    check(Backend.available(), ["impl"], "available()")
    check(Backend.create("impl").forward(5), -5)


def test_create_with_args_and_missing():
    setup()

    class Scaled(Backend, name="scaled"):
        def __init__(self, k, bias=0):
            self.k, self.bias = k, bias

        def forward(self, x):
            return self.k * x + self.bias

    check(Backend.create("scaled", 3, bias=1).forward(2), 7, "带参数创建")
    try:
        Backend.create("nope")
    except KeyError as e:
        assert "scaled" in str(e), f"错误信息里应该列出可用的后端，实际是：{e}"
    else:
        raise AssertionError("不存在的名字应该抛出 KeyError")


def test_registry_not_shadowed_on_subclass():
    """注册表只有一份：通过子类调用 available() 也看到全部"""
    setup()

    class P(Backend, name="p"):
        pass

    class Q(Backend, name="q"):
        pass

    check(P.available(), ["p", "q"], "P.available()")
