import inspect
import re

import solution

_UNITS = {"b": 1, "kb": 1000, "mb": 1000 ** 2, "gb": 1000 ** 3, "kib": 1024, "mib": 1024 ** 2, "gib": 1024 ** 3}


def make(units=_UNITS, case_sensitive=False, allow_space=True, allow_negative=False, truncate=False):
    pattern = r"^\s*(-?\d+(?:\.\d+)?)" + (r"\s*" if allow_space else "") + r"([A-Za-z]*)\s*$"

    def parse_size(s):
        m = re.match(pattern, s)
        if not m:
            raise ValueError(f"无法解析：{s!r}")
        num, unit = m.groups()
        if not unit:
            unit = "b"
        key = unit if case_sensitive else unit.lower()
        table = {k if not case_sensitive else {"b": "B", "kb": "KB", "mb": "MB", "gb": "GB", "kib": "KiB",
                                               "mib": "MiB", "gib": "GiB"}[k]: v for k, v in units.items()}
        if key not in table:
            raise ValueError(f"未知单位：{unit}")
        value = float(num)
        if value < 0 and not allow_negative:
            raise ValueError("大小不能为负")
        if truncate:
            value = int(value)
        total = value * table[key]
        if total != int(total):
            raise ValueError("不是整数字节")
        return int(total)

    return parse_size


CORRECT = make()
MUTANTS = {
    "把 KiB/MiB/GiB 当成 1000 的幂": make(units={**_UNITS, "kib": 1000, "mib": 1000 ** 2, "gib": 1000 ** 3}),
    "单位大小写敏感": make(case_sensitive=True),
    "不允许数字和单位之间有空格": make(allow_space=False),
    "接受负数": make(allow_negative=True),
    "先把小数截断成整数再乘": make(truncate=True),
    "把 GB 当成 1024³": make(units={**_UNITS, "gb": 1024 ** 3}),
}


def _user_tests():
    return [(n, f) for n, f in vars(solution).items()
            if n.startswith("test_") and callable(f) and f.__module__ == "solution"]


def _run(fn, impl):
    try:
        fn(impl)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def test_example_enough_tests():
    """至少 3 个测试，每个接收 parse_size 参数"""
    tests = _user_tests()
    assert len(tests) >= 3, f"只写了 {len(tests)} 个测试，至少需要 3 个"
    for name, fn in tests:
        params = list(inspect.signature(fn).parameters)
        assert params == ["parse_size"], f"{name} 应该只接收一个参数 parse_size，实际参数是 {params}"


def test_example_pass_on_correct():
    """你的测试在正确实现上全部通过"""
    for name, fn in _user_tests():
        err = _run(fn, CORRECT)
        assert err is None, f"{name} 在正确实现上失败了：{err}"


def test_kills_all_mutants():
    """每个有 bug 的实现都被至少一个测试抓住"""
    tests = _user_tests()
    survived = [desc for desc, impl in MUTANTS.items() if all(_run(fn, impl) is None for _, fn in tests)]
    assert not survived, f"{len(MUTANTS) - len(survived)} / {len(MUTANTS)} 个变异体被抓住；没抓住的：" + "；".join(survived)
