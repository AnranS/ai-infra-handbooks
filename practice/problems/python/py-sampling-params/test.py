import dataclasses

from checker import check, raises
from solution import SamplingParams


def test_example():
    p = SamplingParams(temperature=0.7, stop=["</s>", "\n\n"])
    check(p.stop, ("</s>", "\n\n"), "p.stop")
    q = p.with_(max_tokens=128)
    check((q.max_tokens, q.temperature, q.stop), (128, 0.7, ("</s>", "\n\n")), "with_ 之后的字段")
    check(p.max_tokens, 16, "原对象不变")


def test_defaults_and_greedy():
    p = SamplingParams()
    check((p.temperature, p.top_p, p.top_k, p.max_tokens, p.stop), (1.0, 1.0, -1, 16, ()), "默认值")
    assert not p.greedy
    assert SamplingParams(temperature=0).greedy, "temperature=0 应该是 greedy"


def test_stop_normalization():
    check(SamplingParams(stop="END").stop, ("END",), "单个字符串")
    check(SamplingParams(stop=("a", "b")).stop, ("a", "b"), "元组")


def test_validation():
    for kwargs, name in [({"temperature": -0.1}, "temperature"), ({"top_p": 0}, "top_p"), ({"top_p": 1.5}, "top_p"),
                         ({"top_k": 0}, "top_k"), ({"top_k": -2}, "top_k"), ({"max_tokens": 0}, "max_tokens")]:
        try:
            SamplingParams(**kwargs)
        except ValueError as e:
            assert name in str(e), f"{kwargs} 的错误信息里应该包含字段名 {name}，实际是：{e}"
        else:
            raise AssertionError(f"SamplingParams(**{kwargs}) 应该抛出 ValueError")


def test_with_validates():
    with raises(ValueError, "with_(top_p=2)"):
        SamplingParams().with_(top_p=2)


def test_frozen_and_hashable():
    p = SamplingParams(stop=["x"])
    with raises(dataclasses.FrozenInstanceError, "给字段赋值"):
        p.temperature = 0.5
    check(len({p, SamplingParams(stop=("x",)), SamplingParams()}), 2, "放进 set 去重")
