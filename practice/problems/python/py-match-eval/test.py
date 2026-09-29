from checker import check, raises
from solution import evaluate


def test_example():
    check(evaluate(("add", 1, ("mul", "x", 3)), {"x": 2}), 7)
    check(evaluate(("let", "y", 10, ("max", "y", 3, ("neg", 20)))), 10)
    check(evaluate(("if", 0, ("div", 1, 0), 5)), 5, "if 只求一个分支")


def test_arithmetic():
    check(evaluate(("sub", 10, 4)), 6)
    check(evaluate(("div", 7, 2)), 3.5)
    check(evaluate(("neg", ("add", 1.5, 1))), -2.5)
    check(evaluate(("max", 4)), 4, "max 只有一个参数")


def test_errors():
    with raises(NameError, "未定义的变量"):
        evaluate("z", {"x": 1})
    with raises(ZeroDivisionError, "除以 0"):
        evaluate(("div", 1, ("sub", 2, 2)))
    for bad in [("pow", 2, 3), ("add", 1), ("max",), [1, 2], True, None, ("let", 1, 2, 3), ("neg", 1, 2)]:
        with raises(ValueError, f"evaluate({bad!r})"):
            evaluate(bad)


def test_let_scoping():
    env = {"x": 1}
    check(evaluate(("let", "x", 5, ("add", "x", 1)), env), 6, "内层 x")
    check(env, {"x": 1}, "外层 env 不应该被修改")
    nested = ("let", "a", 2, ("let", "b", ("mul", "a", 3), ("add", "a", "b")))
    check(evaluate(nested), 8, "嵌套的 let")


def test_if_branches():
    check(evaluate(("if", ("sub", 3, 3), "missing", ("add", 1, 1))), 2, "条件为 0 走 else")
    check(evaluate(("if", 2, 42, "missing")), 42, "条件非零走 then")


def test_deep_expression():
    e = 0
    for i in range(200):
        e = ("add", e, i)
    check(evaluate(e), sum(range(200)), "嵌套 200 层")
