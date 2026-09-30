"""C++ 练习题的判题：g++ / clang++ 以 C++20 编译，默认开 AddressSanitizer + UndefinedBehaviorSanitizer，
并发题（problem.md 里写 sanitize: thread）改用 ThreadSanitizer。sanitizer 报告任何问题都算不通过。

题目目录里的 test.cpp 以 `#include "user.cpp"` 引入你的代码，输出 CASE 行（见 runtime/cpp/judge.hpp）。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from cudajudge import parse

HERE = Path(__file__).resolve().parent
BUILD = HERE / "workspace" / ".build"
REPORTS = ("ERROR: AddressSanitizer", "ERROR: LeakSanitizer", "runtime error:", "WARNING: ThreadSanitizer")


def compiler():
    return os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")


def judge(p, code: str) -> dict:
    cxx = compiler()
    if not cxx:
        return {"status": "compile_error", "cases": [], "passed": 0, "total": 0,
                "error": "没有找到 C++ 编译器（g++ 或 clang++）：macOS 运行 xcode-select --install，Linux 安装 g++"}
    work = BUILD / p.slug
    work.mkdir(parents=True, exist_ok=True)
    (work / "user.cpp").write_text(code, encoding="utf-8")
    (work / "test.cpp").write_text(p.tests, encoding="utf-8")
    san = p.sanitize or "address,undefined"
    exe = work / "test_cpp"
    cmd = [cxx, "-std=c++20", "-g", "-O1", "-Wall", "-Wextra", "-pthread", f"-fsanitize={san}", "-fno-omit-frame-pointer",
           f"-I{HERE / 'runtime' / 'cpp'}", f"-I{work}", str(work / "test.cpp"), "-o", str(exe)]
    c = subprocess.run(cmd, capture_output=True, text=True)
    if c.returncode:
        return {"status": "compile_error", "cases": [], "passed": 0, "total": 0, "error": "编译失败：\n" + c.stderr[-4000:]}
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=%d" % (platform.system() != "Darwin"),
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
               TSAN_OPTIONS=f"halt_on_error=1:suppressions={HERE / 'runtime' / 'cpp' / 'tsan.supp'}")   # 只抑制已知的 libstdc++ 误报
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=180, cwd=work, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cases": [], "passed": 0, "total": 0, "error": "运行超过 180 秒（死循环或死锁？）"}
    res = parse(r.stdout)
    cases = res["cases"]
    passed = sum(x["status"] == "pass" for x in cases)
    report = next((m for m in REPORTS if m in r.stderr), None)
    err = None
    if report or r.returncode != 0:
        lines = [l for l in r.stderr.splitlines() if l.strip()]
        err = (f"sanitizer 报告了问题（{report.split(':')[0].strip()}）：\n" if report else f"程序异常退出（返回码 {r.returncode}）：\n") + \
            "\n".join(lines[:40])
    if cases and passed == len(cases) and not err:
        status = "accepted"
    elif err and not any(x["status"] == "fail" for x in cases):
        status = "runtime_error"
    else:
        status = "wrong_answer"
    kind = {"thread": "ThreadSanitizer"}.get(san, "AddressSanitizer + UBSan")
    return {"status": status, "cases": cases, "passed": passed, "total": len(cases), "error": err,
            "note": f"判题方式：{Path(cxx).name} -std=c++20，{kind}", "stdout": ""}
