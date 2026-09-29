import json
import random

from checker import check, raises
from solution import loads


def test_example():
    text = '{"model": "qwen3", "temperature": 0.7, "stop": ["\\n", "</s>"], "stream": true, "user": null}'
    check(loads(text), json.loads(text))


def test_numbers():
    for t in ["0", "-0", "12", "-7", "1.5", "-2e3", "1E-2", "6.02e+23", "123456789012345678901234567890"]:
        got, want = loads(t), json.loads(t)
        check((got, type(got)), (want, type(want)), f"loads({t!r})")
    for bad in ["012", "+1", ".5", "1.", "1e", "-", "--1", "1.2.3", "0x10"]:
        with raises(ValueError, f"loads({bad!r})"):
            loads(bad)


def test_strings():
    cases = ['"a\\"b\\\\c\\/d"', '"\\b\\f\\n\\r\\t"', '"\\u4e2d\\u6587"', '"\\ud83d\\ude00"', '"中文 😀"', '""']
    for t in cases:
        check(loads(t), json.loads(t), f"loads({t!r})")
    for bad in ['"abc', '"\\x"', '"\\u12"', '"a\nb"', "'single'"]:
        with raises(ValueError, f"loads({bad!r})"):
            loads(bad)


def test_structures_and_whitespace():
    for t in ["[]", "{}", " [ 1 , [ 2 , { } ] ] ", '{"a": {"b": [true, false, null]}}', '\n\t{"k"\r:\n1}\n']:
        check(loads(t), json.loads(t), f"loads({t!r})")
    for bad in ["[1,]", "[1 2]", '{"a" 1}', '{"a": 1,}', "{1: 2}", "[", "{", "tru", "nul", "[] []", ""]:
        with raises(ValueError, f"loads({bad!r})"):
            loads(bad)


def test_error_position():
    try:
        loads('{"a" 1}')
    except ValueError as e:
        assert "5" in str(e), f"错误信息应该包含出错位置 5，实际是：{e}"
    else:
        raise AssertionError("应该抛出 ValueError")


def rand_value(rng, depth=0):
    kind = rng.randint(0, 6 if depth < 4 else 3)
    if kind == 0:
        return rng.randint(-10 ** 6, 10 ** 6)
    if kind == 1:
        return rng.uniform(-1e5, 1e5)
    if kind == 2:
        return "".join(rng.choice('ab"\\\n\t中😀\u0001/') for _ in range(rng.randint(0, 8)))
    if kind == 3:
        return rng.choice([True, False, None])
    if kind in (4, 5):
        return [rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rand_value(rng, 9) if False else f"k{rng.randint(0, 99)}": rand_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def test_random_roundtrip():
    rng = random.Random(0)
    for i in range(200):
        v = rand_value(rng)
        text = json.dumps(v, ensure_ascii=rng.random() < 0.5, indent=rng.choice([None, 2]))
        check(loads(text), json.loads(text), f"第 {i} 个随机用例")
