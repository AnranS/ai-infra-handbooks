#!/usr/bin/env python3
"""练习题的本地判题工具（macOS / Linux / WSL2 通用）。

    python practice/judge.py doctor              # 检查本机环境：能跑哪些题（PyTorch、MPS、CUDA、nvcc、Triton）
    python practice/judge.py bench [--quick]     # 实测显存带宽、矩阵乘算力、多卡 all-reduce，作为性能门槛的分母
    python practice/judge.py list [--book cuda]  # 题目列表
    python practice/judge.py start 12            # 把第 12 题的模板复制到 practice/workspace/，并打印题目
    python practice/judge.py test 12             # 判题：跑 practice/workspace/ 里你的代码
    python practice/judge.py test 12 my.py --run # 只跑样例；也可以指定别的文件
    python practice/judge.py test 40 kernel.cu   # CUDA C++ 题：nvcc 编译后在 GPU 上运行
    python practice/judge.py solution 12         # 查看参考解答
    python practice/judge.py check               # （维护者）所有参考解答都要通过、所有模板都不能通过

题号可以写编号（12）或 id（py-lru-cache）。浏览器里的判题用的是同一个内核（runtime/judge_runner.py）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime"))

import problems as P  # noqa: E402

WORKSPACE = HERE / "workspace"
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
STATUS = {"accepted": ("通过", "32"), "wrong_answer": ("答案错误", "31"), "runtime_error": ("运行错误", "31"),
          "compile_error": ("编译错误", "31"), "timeout": ("超时", "33")}


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


# ---------------------------------------------------------------- 环境

def detect() -> dict:
    info = {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
            "wsl": "microsoft" in platform.release().lower(), "numpy": None, "torch": None, "device": "cpu",
            "gpu": None, "triton": None, "nvcc": shutil.which("nvcc"), "cxx": shutil.which("g++") or shutil.which("clang++")}
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu"] = torch.cuda.get_device_name(0)
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["device"] = "mps"
            info["gpu"] = "Apple GPU (MPS)"
    except ImportError:
        pass
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        pass
    if info["nvcc"] is None and Path("/usr/local/cuda/bin/nvcc").exists():
        info["nvcc"] = "/usr/local/cuda/bin/nvcc"
    return info


def doctor(_args):
    info = detect()
    ok, no = c("✓", "32"), c("✗", "31")
    plat = "WSL2" if info["wsl"] else {"Darwin": "macOS", "Linux": "Linux"}.get(info["os"], info["os"])
    print(f"平台：{plat} {info['machine']}，Python {info['python']}")
    rows = [
        ("numpy", info["numpy"], "所有浏览器题（Python / numpy / GPU 模拟器 / Triton 模拟器）"),
        ("PyTorch", info["torch"], "需要 PyTorch 的题"),
        ("GPU", info["gpu"] and f"{info['gpu']}（{info['device']}）", "PyTorch 题跑在 GPU 上"),
        ("Triton", info["triton"] if info["device"] == "cuda" else None, "Triton 题用真 Triton（否则自动用模拟器）"),
        ("nvcc", info["nvcc"] if info["device"] == "cuda" else None, "CUDA C++ 题在真卡上编译运行"),
        ("C++ 编译器", info["cxx"], "没有 GPU 时，CUDA C++ 题用 CPU 模拟器检查正确性"),
    ]
    for name, value, use in rows:
        print(f"  {ok if value else no} {name:<10} {value or '未检测到':<34} {use}")
    if sys.version_info < (3, 10):
        print(c("  Python 版本太旧：需要 3.10 以上", "31"))
    envs = ["browser", "local"] + (["torch"] if info["torch"] else []) + (["cuda"] if info["device"] == "cuda" and info["nvcc"] else [])
    problems, _ = P.load_all()
    n = sum(p.env in envs for p in problems)
    print(f"本机可以完整判题的题目：{n} / {len(problems)}")
    if info["torch"]:
        print("实测本机的带宽和算力（性能门槛的分母）：python practice/judge.py bench")
    if not info["numpy"]:
        print("先安装依赖：见 practice/README.md（macOS: practice/env/setup-macos.sh；WSL2: practice/env/setup-wsl2.sh）")


# ---------------------------------------------------------------- 判题

def _prepare(p: P.Problem):
    if "triton" in p.all_requires:
        import tritonkit

        return tritonkit.install()
    return None


def judge_python(p: P.Problem, code: str, mode: str, timeout: int = 120) -> dict:
    import judge_runner

    _prepare(p)
    use_alarm = hasattr(signal, "SIGALRM")

    def on_alarm(*_):
        raise TimeoutError

    if use_alarm:
        old = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(timeout)
    try:
        return asyncio.run(judge_runner.run(code, p.tests, mode))
    except TimeoutError:
        return {"status": "timeout", "cases": [], "error": f"运行超过 {timeout} 秒（死循环？）", "passed": 0, "total": 0}
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def print_result(res: dict) -> None:
    if res.get("note"):
        print(c(res["note"], "36"))
    for case in res.get("cases", []):
        mark = {"pass": c("✓", "32"), "fail": c("✗", "31"), "error": c("✗", "31"), "skip": c("-", "33")}[case["status"]]
        doc = f"  {case['doc']}" if case["doc"] else ""
        print(f" {mark} {case['name']}{c(doc, '90')}  {case['ms']:.0f} ms")
        if case["message"]:
            print("\n".join("     " + line for line in case["message"].splitlines()))
        if case.get("stdout"):
            print(c("     输出：", "90") + case["stdout"].rstrip().replace("\n", "\n           "))
    if res.get("error"):
        print(res["error"])
    if res.get("stdout"):
        print(c("程序输出：" if res.get("note") else "导入时的输出：", "90") + res["stdout"].rstrip())
    name, color = STATUS.get(res["status"], (res["status"], "31"))
    skipped = f"，跳过 {res['skipped']}" if res.get("skipped") else ""
    print(c(f"{name}：{res.get('passed', 0)} / {res.get('total', 0)} 个用例通过{skipped}", color))


def cmd_list(args):
    problems, chapters = P.load_all()
    book = None
    for p in problems:
        if args.book and p.book != args.book:
            continue
        if p.book != book:
            book = p.book
            print(c(f"\n{dict(P.BOOKS)[book]}", "1"))
        ch = chapters[p.book][p.chapter].title
        env = {"browser": "", "local": " [本地]", "torch": " [PyTorch]", "cuda": " [NVIDIA GPU]"}[p.env]
        print(f"  {p.number:>3}. {p.title:<24} {p.difficulty}  {c(ch, '90')}  id={p.slug}{env}")


def _workspace_file(p: P.Problem, ext=".py") -> Path:
    return WORKSPACE / f"{p.number:03d}-{p.slug}{ext}"


def cmd_start(args):
    p = P.find(args.id)
    WORKSPACE.mkdir(exist_ok=True)
    f = _workspace_file(p, ".cu" if args.cuda else ".py")
    if not f.exists():
        f.write_text((p.cuda or {}).get("starter", "") if args.cuda else p.starter, encoding="utf-8")
    print(c(f"#{p.number} {p.title}（{p.difficulty}）", "1"))
    print(p.description)
    print(c(f"\n模板已复制到 {f.relative_to(Path.cwd()) if f.is_relative_to(Path.cwd()) else f}", "36"))
    print(f"写完后运行：python practice/judge.py test {p.number}")


def cmd_test(args):
    p = P.find(args.id)
    path = Path(args.file) if args.file else None
    if path is None:
        for ext in (".py", ".cu"):
            if _workspace_file(p, ext).exists():
                path = _workspace_file(p, ext)
                break
    if path is None:
        raise SystemExit(f"没有找到你的代码：先运行 python practice/judge.py start {p.number}")
    code = path.read_text(encoding="utf-8")
    print(c(f"#{p.number} {p.title} ← {path}", "1"))
    if path.suffix == ".cu":
        import cudajudge

        res = cudajudge.judge(p, code, detect())
    else:
        res = judge_python(p, code, "run" if args.run else "submit")
    print_result(res)
    sys.exit(0 if res["status"] == "accepted" else 1)


def cmd_solution(args):
    p = P.find(args.id)
    print(c(f"#{p.number} {p.title} 参考解答", "1"))
    print(p.solution)
    if p.explanation:
        print(c("讲解：", "1"))
        print(p.explanation)


def _check_one(slug: str) -> tuple[str, list[str]]:
    p = P.find(slug)
    errors = []
    res = judge_python(p, p.solution, "submit")
    if res["status"] != "accepted":
        errors.append("参考解答没有通过：" + res["status"] + "\n" + (res.get("error") or "") + "\n" + "\n".join(
            f"{x['name']}: {x['message']}" for x in res["cases"] if x["status"] not in ("pass", "skip")))
    if not any(n.startswith("test_example") for n in _test_names(p.tests)):
        errors.append("缺少 test_example* 样例用例")
    res = judge_python(p, p.starter, "submit")
    if res["status"] == "accepted":
        errors.append("模板代码竟然通过了所有测试")
    if p.cuda and (shutil.which("g++") or shutil.which("clang++")):
        import cudajudge

        r = cudajudge.judge(p, p.cuda["solution"], {}, force_emulator=True)
        if r["status"] != "accepted":
            errors.append("CUDA C++ 参考解答在模拟器上没有通过：" + r["status"] + "\n" + (r.get("error") or "") + "\n" +
                          "\n".join(f"{x['name']}: {x['message']}" for x in r["cases"] if x["status"] != "pass"))
        r = cudajudge.judge(p, p.cuda["starter"], {}, force_emulator=True)
        if r["status"] == "accepted":
            errors.append("CUDA C++ 模板竟然通过了测试")
    for key in ("title", "chapter"):
        if not getattr(p, key):
            errors.append(f"缺少 {key}")
    return slug, errors


def _test_names(code: str):
    import re

    return re.findall(r"^(?:async\s+)?def (test_\w+)", code, re.M)


def cmd_check(args):
    problems, _ = P.load_all()
    slugs = [P.find(i).slug for i in args.ids] if args.ids else [p.slug for p in problems]
    from concurrent.futures import ProcessPoolExecutor

    bad = 0
    with ProcessPoolExecutor(max(1, min(os.cpu_count() or 2, 16))) as ex:
        for slug, errors in ex.map(_check_one, slugs):
            if errors:
                bad += 1
                print(c(f"✗ {slug}", "31"))
                for e in errors:
                    print("   " + e.strip().replace("\n", "\n   "))
            elif args.verbose:
                print(c(f"✓ {slug}", "32"))
    print(f"{len(slugs) - bad} / {len(slugs)} 道题检查通过")
    sys.exit(1 if bad else 0)


def cmd_bench(args):
    import bench

    bench.run(args.quick, args.cpu_procs)


def main():
    ap = argparse.ArgumentParser(description="练习题本地判题", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=doctor)
    s = sub.add_parser("list")
    s.add_argument("--book", choices=[b for b, _ in P.BOOKS])
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("start")
    s.add_argument("id")
    s.add_argument("--cuda", action="store_true", help="CUDA C++ 版本的模板")
    s.set_defaults(fn=cmd_start)
    s = sub.add_parser("test")
    s.add_argument("id")
    s.add_argument("file", nargs="?")
    s.add_argument("--run", action="store_true", help="只跑样例")
    s.set_defaults(fn=cmd_test)
    s = sub.add_parser("solution")
    s.add_argument("id")
    s.set_defaults(fn=cmd_solution)
    s = sub.add_parser("bench", help="实测本机的带宽、算力和 all-reduce 带宽")
    s.add_argument("--quick", action="store_true", help="缩小规模，十几秒跑完")
    s.add_argument("--cpu-procs", type=int, default=0, help="没有多张 GPU 时，用 N 个 CPU 进程走一遍 all-reduce 流程")
    s.set_defaults(fn=cmd_bench)
    s = sub.add_parser("check")
    s.add_argument("ids", nargs="*")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
