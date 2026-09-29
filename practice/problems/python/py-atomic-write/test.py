import os
import tempfile
import time

from checker import check, raises
from solution import Timer, atomic_write


def _dir():
    return tempfile.mkdtemp(prefix="aw-")


def test_example_success():
    d = _dir()
    path = os.path.join(d, "config.json")
    with atomic_write(path) as f:
        f.write('{"a": 1}')
    with open(path) as f:
        check(f.read(), '{"a": 1}', "写入的内容")
    check(sorted(os.listdir(d)), ["config.json"], "目录里不应该留下临时文件")


def test_failure_keeps_original():
    d = _dir()
    path = os.path.join(d, "data.txt")
    with open(path, "w") as f:
        f.write("old")
    with raises(RuntimeError, "with 块里抛出的异常"):
        with atomic_write(path) as f:
            f.write("new but incomplete")
            raise RuntimeError("crash")
    with open(path) as f:
        check(f.read(), "old", "原文件的内容")
    check(sorted(os.listdir(d)), ["data.txt"], "临时文件应该被删除")


def test_not_written_until_exit():
    d = _dir()
    path = os.path.join(d, "x.txt")
    with atomic_write(path) as f:
        f.write("hello")
        assert not os.path.exists(path), "with 块结束之前目标文件不应该出现"
    assert os.path.exists(path)


def test_binary_and_bad_mode():
    d = _dir()
    path = os.path.join(d, "b.bin")
    with atomic_write(path, "wb") as f:
        f.write(b"\x00\x01")
    with open(path, "rb") as f:
        check(f.read(), b"\x00\x01", "二进制内容")
    with raises(ValueError, "mode='a'"):
        with atomic_write(path, "a"):
            pass


def test_example_timer():
    with Timer() as t:
        time.sleep(0.05)
    assert 0.04 <= t.elapsed < 1.0, f"sleep 0.05 秒，t.elapsed = {t.elapsed}"
    e1 = t.elapsed
    time.sleep(0.02)
    check(t.elapsed, e1, "结束之后 elapsed 不再变化")


def test_timer_inside_and_exceptions():
    with raises(ZeroDivisionError, "Timer 不应该吞掉异常"):
        with Timer() as t:
            time.sleep(0.01)
            mid = t.elapsed
            assert mid > 0, "块内读取 elapsed 应该大于 0"
            1 / 0
    assert t.elapsed >= mid
