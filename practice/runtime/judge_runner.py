"""判题内核：浏览器（Pyodide）和本地命令行（practice/judge.py）共用同一份代码。

一道题由三部分组成：用户代码（作为模块 `solution` 导入）、测试代码（若干个 test_* 函数，
可以是 async def）和题目元数据。run() 依次执行测试，返回每个用例的结果；断言失败时，
把失败的那一行和测试函数里的局部变量一起报告出来，效果类似 pytest 的断言改写。
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import linecache
import sys
import time
import traceback
import types

SOLUTION = "<solution>"
TESTS = "<tests>"
MAX_OUT = 4000


def _register(filename: str, code: str) -> None:
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)


def _short(value, limit: int = 300) -> str:
    try:
        import numpy as np  # noqa: F401  （没有 numpy 也能用）

        if type(value).__module__ == "numpy" and hasattr(value, "shape") and getattr(value, "size", 0) > 20:
            text = f"array(shape={value.shape}, dtype={value.dtype})\n{value!r}"
        else:
            text = repr(value)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + " …"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_OUT else text[:MAX_OUT] + "\n…（输出过长，已截断）"


def _user_traceback(exc: BaseException) -> str:
    """只保留用户代码和测试代码里的栈帧，隐藏判题器和模拟器自身。"""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename in (SOLUTION, TESTS)]
    lines = []
    for f in frames:
        where = "你的代码" if f.filename == SOLUTION else "测试"
        lines.append(f"  {where} 第 {f.lineno} 行，{f.name}()")
        if f.line:
            lines.append(f"    {f.line.strip()}")
    head = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return ("Traceback:\n" + "\n".join(lines) + "\n" + head) if lines else head


def _assert_message(exc: AssertionError) -> str:
    msg = str(exc)
    tb = exc.__traceback__
    frame, lineno = None, None
    while tb is not None:                       # 找到测试文件里触发断言的那一帧
        if tb.tb_frame.f_code.co_filename == TESTS:
            frame, lineno = tb.tb_frame, tb.tb_lineno
        tb = tb.tb_next
    if frame is None:
        return msg or "断言失败"
    line = linecache.getline(TESTS, lineno).strip()
    parts = [msg] if msg else []
    if not msg:
        parts.append(f"失败的断言：{line}")
    shown = []
    for name, value in frame.f_locals.items():
        if name.startswith("_") or isinstance(value, (types.ModuleType, types.FunctionType, type)):
            continue
        if name in line:
            shown.append(f"  {name} = {_short(value)}")
    if shown and not msg:
        parts.append("相关变量：\n" + "\n".join(shown))
    return "\n".join(parts)


def load_solution(code: str, name: str = "solution") -> types.ModuleType:
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        del sys.modules[key]
    _register(SOLUTION, code)
    module = types.ModuleType(name)
    module.__file__ = SOLUTION
    sys.modules[name] = module
    exec(compile(code, SOLUTION, "exec"), module.__dict__)
    return module


def collect(test_code: str) -> tuple[dict, list]:
    _register(TESTS, test_code)
    module = types.ModuleType("tests")         # 注册成真正的模块：TypedDict / dataclass 解析前向引用时要用
    module.__file__ = TESTS
    sys.modules["tests"] = module
    ns = module.__dict__
    exec(compile(test_code, TESTS, "exec"), ns)
    tests = [(k, v) for k, v in ns.items() if k.startswith("test_") and callable(v)]
    tests.sort(key=lambda kv: kv[1].__code__.co_firstlineno)
    return ns, tests


async def run(code: str, test_code: str, mode: str = "submit", solution_name: str = "solution") -> dict:
    """mode="run" 只跑名字以 test_example 开头的样例用例（没有就跑第一个），mode="submit" 跑全部。"""
    t0 = time.perf_counter()
    out = io.StringIO()
    result: dict = {"status": "accepted", "cases": [], "stdout": "", "error": None, "passed": 0, "total": 0}
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            load_solution(code, solution_name)
    except SyntaxError as e:
        result.update(status="compile_error", error=f"语法错误：第 {e.lineno} 行\n{(e.text or '').rstrip()}\n{e.msg}")
        result["stdout"] = _clip(out.getvalue())
        return result
    except BaseException as e:  # noqa: BLE001  用户代码导入时抛出的任何异常
        result.update(status="runtime_error", error="导入你的代码时出错：\n" + _user_traceback(e))
        result["stdout"] = _clip(out.getvalue())
        return result
    result["stdout"] = _clip(out.getvalue())

    try:
        _, tests = collect(test_code)
    except BaseException as e:  # noqa: BLE001
        msg = _user_traceback(e)
        if isinstance(e, ImportError):
            msg = "测试需要的函数或类没有找到（是不是改了名字？）\n" + msg
        result.update(status="runtime_error", error=msg)
        return result

    if mode == "run":
        examples = [t for t in tests if t[0].startswith("test_example")]
        tests = examples or tests[:1]
    result["total"] = len(tests)
    first_bad = None
    for name, fn in tests:
        case = {"name": name, "doc": (inspect.getdoc(fn) or "").split("\n")[0], "status": "pass", "message": "",
                "stdout": "", "ms": 0.0}
        buf = io.StringIO()
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                ret = fn()
                if inspect.isawaitable(ret):
                    await ret
        except AssertionError as e:
            case.update(status="fail", message=_assert_message(e))
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001  包括用户代码里的 exit()
            if type(e).__name__ == "Skip":
                case.update(status="skip", message=str(e))
            else:
                case.update(status="error", message=_user_traceback(e))
        case["ms"] = round((time.perf_counter() - start) * 1000, 2)
        case["stdout"] = _clip(buf.getvalue())
        if case["status"] == "pass":
            result["passed"] += 1
        elif case["status"] == "skip":
            result["skipped"] = result.get("skipped", 0) + 1
        elif first_bad is None:
            first_bad = case["status"]
        result["cases"].append(case)
    if first_bad:
        result["status"] = "wrong_answer" if first_bad == "fail" else "runtime_error"
    result["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return result


def run_sync(code: str, test_code: str, mode: str = "submit") -> dict:
    return asyncio.run(run(code, test_code, mode))
