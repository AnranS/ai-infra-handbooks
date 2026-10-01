"""运行计算机基础手册里的示例（Python 与 C），确认页面上的输出与实际运行结果一致。例子针对 Linux，在 Linux 上检查。

约定：
  ```python title="x.py"        完整脚本：运行它，紧跟的 ```text title="输出"``` 必须与标准输出逐行一致
  ```c title="x.c"              完整的 C 程序：gcc -O2 -Wall -Wextra -Werror -pthread 编译后运行，输出同样逐行比对；
                                 flags="-lm" 这类属性追加到编译命令末尾
  ```text title="输出（本机示例）"  带括号说明的输出块是和机器相关的测量结果（耗时、带宽、缺页次数……），只要求程序跑通，不比对
  run="no"                      只做语法检查（py_compile / gcc -fsyntax-only），不运行
  ci="no"                       本机正常运行，但在 CI（环境变量 CI 非空）里只做语法检查：依赖特定硬件或权限的例子
  没有 title 的代码块是片段，不检查。同一页的程序写在同一个目录里，按页面顺序运行，可以读前面的程序留下的文件。

用法：python3 tools/check_code.py [docs/xxx/yyy.md ...]    （不带参数时检查所有页面）
解释器：环境变量 PYTHON（默认 python3）；IPC 一章用到 pyzmq。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "examples"
PYTHON = os.environ.get("PYTHON") or "python3"
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')



NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def same_output(want: list[str], got: list[str]) -> bool:
    """本机要求逐行完全一致；CI（环境变量 CI 非空）里允许数字有 5% 的相对误差（百分数 1 个百分点）：
    不同 CPU / BLAS 的浮点归约顺序不同，训练 loss、KL 这类数的第三位小数会变，文字部分仍要完全一致。"""
    if want == got:
        return True
    if not os.environ.get("CI") or len(want) != len(got):
        return False
    for w, g in zip(want, got):
        if NUM.sub("#", w) != NUM.sub("#", g):
            return False
        for ma, mb in zip(NUM.finditer(w), NUM.finditer(g)):
            x, y = float(ma.group()), float(mb.group())
            pct = w[ma.end():ma.end() + 1] == "%"              # 百分数（常常是两个数的差）按 1 个百分点算
            if abs(x - y) > (1.0 if pct else max(0.05 * max(abs(x), abs(y)), 0.011)):
                return False
    return True

def blocks(md: Path):
    lines = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence = m["indent"], m["fence"]
        j = i + 1
        while j < len(lines) and lines[j].strip() != fence:
            j += 1
        body = "\n".join(l[len(indent):] if l.startswith(indent) else l.lstrip() for l in lines[i + 1:j])
        yield {"lang": m["lang"], "attrs": dict(ATTR.findall(m["rest"])), "body": body + "\n", "line": i + 1}
        i = j + 1


def run(cmd, cwd, env, timeout=300):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None


def check_page(md: Path) -> tuple[int, list[str]]:
    items = list(blocks(md))
    work = BUILD / md.relative_to(ROOT / "docs").with_suffix("")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    progs = [(k, b) for k, b in enumerate(items)
             if (b["lang"] == "python" and b["attrs"].get("title", "").endswith(".py"))
             or (b["lang"] == "c" and b["attrs"].get("title", "").endswith(".c"))]
    for _, b in progs:
        (work / b["attrs"]["title"]).write_text(b["body"], encoding="utf-8")
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    errors = []
    for k, b in progs:
        title = b["attrs"]["title"]
        where = f"{md.relative_to(ROOT)}:{b['line']} {title}"
        dry = b["attrs"].get("run") == "no" or (os.environ.get("CI") and b["attrs"].get("ci") == "no")   # ci="no"：依赖特定硬件或权限（AVX-512、perf、numactl、mlock、O_DIRECT），CI 里只编译不运行
        if b["lang"] == "c":
            exe = title[:-2]
            cmd = ["gcc", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread"] + (["-fsyntax-only"] if dry else ["-o", exe])
            r = run(cmd + [title] + b["attrs"].get("flags", "").split(), work, env)
            if r is None or r.returncode:
                errors.append(f"{where}: 编译失败\n{(r.stderr if r else '超时')[-3000:]}")
                continue
            if dry:
                continue
            r = run([f"./{exe}"], work, env)
        else:
            r = run([PYTHON, "-m", "py_compile", title] if dry else [PYTHON, title], work, env)
            if dry:
                if r is None or r.returncode:
                    errors.append(f"{where}: 语法错误\n{(r.stderr if r else '超时')[-2000:]}")
                continue
        if r is None:
            errors.append(f"{where}: 运行超过 300 秒")
            continue
        if r.returncode:
            errors.append(f"{where}: 运行失败（返回码 {r.returncode}）\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if nxt and nxt["lang"] == "text" and nxt["attrs"].get("title") == "输出":
            want = [l.rstrip() for l in nxt["body"].rstrip("\n").splitlines()]
            got = [l.rstrip() for l in r.stdout.rstrip("\n").splitlines()]
            if not same_output(want, got):
                errors.append(f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got))
    return len(progs), errors


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    total, errs = 0, []
    for p in pages:
        n, e = check_page(p)
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(pages)} 个页面，{total} 个程序，{len(errs)} 个问题（{PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
