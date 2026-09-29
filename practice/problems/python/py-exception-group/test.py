from checker import check, raises
from solution import count_errors, validate_batch


def need_prompt(r):
    if "prompt" not in r:
        raise ValueError("缺少 prompt")


def need_int_tokens(r):
    if not isinstance(r.get("max_tokens", 1), int):
        raise TypeError("max_tokens 必须是整数")


def positive_tokens(r):
    if isinstance(r.get("max_tokens", 1), int) and r.get("max_tokens", 1) <= 0:
        raise ValueError("max_tokens 必须为正")


V = [need_prompt, need_int_tokens, positive_tokens]


def test_example():
    reqs = [{"prompt": "hi"}, {"max_tokens": "8"}]
    try:
        validate_batch(reqs, V)
    except ExceptionGroup as eg:
        check(eg.message, "1 个请求校验失败", "外层 ExceptionGroup 的消息")
        check(len(eg.exceptions), 1, "出错的请求数")
        inner = eg.exceptions[0]
        assert isinstance(inner, ExceptionGroup), "每个出错的请求应该是一个 ExceptionGroup"
        check(inner.message, "request 1", "内层消息")
        check([type(e).__name__ for e in inner.exceptions], ["ValueError", "TypeError"], "这个请求的错误（按校验顺序）")
    else:
        raise AssertionError("应该抛出 ExceptionGroup")
    check(count_errors(lambda: validate_batch(reqs, V)), {"value": 1, "type": 1}, "count_errors")


def test_all_valid():
    check(validate_batch([{"prompt": "a"}, {"prompt": "b", "max_tokens": 3}], V), None, "全部合法")
    check(count_errors(lambda: None), {"value": 0, "type": 0}, "没有异常")


def test_many_requests():
    reqs = [{"prompt": "x", "max_tokens": 0}, {}, {"prompt": "y"}, {"max_tokens": -1}, {"prompt": "z", "max_tokens": 2.5}]
    try:
        validate_batch(reqs, V)
    except ExceptionGroup as eg:
        check(eg.message, "4 个请求校验失败", "外层消息")
        check([g.message for g in eg.exceptions], ["request 0", "request 1", "request 3", "request 4"], "出错的请求")
    else:
        raise AssertionError("应该抛出 ExceptionGroup")
    check(count_errors(lambda: validate_batch(reqs, V)), {"value": 4, "type": 1}, "count_errors")


def test_count_plain_and_nested():
    def plain():
        raise ValueError("x")

    def nested():
        raise ExceptionGroup("a", [ValueError(1), ExceptionGroup("b", [TypeError(2), ValueError(3), TypeError(4)])])

    check(count_errors(plain), {"value": 1, "type": 0}, "单个 ValueError")
    check(count_errors(nested), {"value": 2, "type": 2}, "多层嵌套")


def test_other_exceptions_propagate():
    def mixed():
        raise ExceptionGroup("m", [ValueError(1), KeyError("k")])

    with raises(BaseExceptionGroup, "含有 KeyError 的组"):
        count_errors(mixed)
    with raises(KeyError, "单独的 KeyError"):
        count_errors(lambda: {}["missing"])
