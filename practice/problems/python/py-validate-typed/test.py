from typing import Any, Literal, Optional, TypedDict, Union

from checker import check


class Req(TypedDict):
    prompt: str
    stop: list[str]


class Msg(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Chat(TypedDict):
    model: str
    messages: list[Msg]
    temperature: Optional[float]


def expect_error(value, tp, where):
    from solution import validate

    try:
        validate(value, tp)
    except TypeError as e:
        assert where in str(e), f"错误信息里应该包含位置 {where}，实际是：{e}"
        return
    raise AssertionError(f"validate({value!r}, {tp}) 应该抛出 TypeError")


def test_example():
    from solution import validate

    check(validate({"prompt": "hi", "stop": ["a"]}, Req), True)
    expect_error({"prompt": "hi", "stop": ["a", 1]}, Req, '$["stop"][1]')


def test_basic_types():
    from solution import validate

    check(validate(3, int), True)
    check(validate("x", str), True)
    expect_error(True, int, "$")
    expect_error("3", int, "$")


def test_containers():
    from solution import validate

    check(validate([1, 2], list[int]), True)
    check(validate({"a": [1.5]}, dict[str, list[float]]), True)
    check(validate((1, "a"), tuple[int, str]), True)
    check(validate((1, 2, 3), tuple[int, ...]), True)
    check(validate({1, 2}, set[int]), True)
    expect_error([1, "2"], list[int], "$[1]")
    expect_error({"a": [1.5, "x"]}, dict[str, list[float]], '$["a"][1]')
    expect_error((1, 2), tuple[int, str], "$[1]")
    expect_error((1, 2, 3), tuple[int, int], "$")


def test_unions_and_literals():
    from solution import validate

    check(validate(None, Optional[int]), True)
    check(validate(5, int | None), True)
    check(validate("s", Union[int, str]), True)
    check(validate("user", Literal["user", "assistant"]), True)
    check(validate(object(), Any), True)
    expect_error(2.5, int | None, "$")
    expect_error("system", Literal["user", "assistant"], "$")


def test_nested_typeddict():
    from solution import validate

    ok = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": None}
    check(validate(ok, Chat), True)
    bad = {"model": "m", "messages": [{"role": "user", "content": "hi"}, {"role": "bot", "content": "x"}],
           "temperature": 0.5}
    expect_error(bad, Chat, '$["messages"][1]["role"]')
    expect_error({"model": "m", "messages": []}, Chat, "temperature")
    expect_error({"model": "m", "messages": [], "temperature": 1.0, "extra": 1}, Chat, "extra")
