"""编译并运行 docs/ 里的 C++ 示例，确认页面上的输出和实际运行结果一致。

Markdown 里的约定：
  ```cpp title="name.cpp"          完整程序：g++ -std=c++20 编译，默认开 ASan + UBSan，运行一次
  ```cpp title="name.hpp"          头文件：写到同一页的构建目录里，供本页的程序 #include
  ```text title="输出"             紧跟在程序后面：程序的标准输出必须与它逐行一致
  没有 title 的代码块是片段，只用来讲解，不检查。

程序代码块可以带这些属性（写在 title 后面）：
  sanitize="thread"    改用 ThreadSanitizer（并发章节）；sanitize="none" 不开 sanitizer（测性能的例子）
  expect="fail"        故意演示错误：程序必须失败（sanitizer 报错或返回非 0），页面上的报告节选不做比对
  expect="compile-error"  故意演示编译错误：必须编译失败；可以再加 error="片段"，要求编译器的报错里包含它
  run="no"             只编译不运行
  project="名字"       属于一个多文件工程：按 title 里的相对路径写进同一个目录（CMakeLists.txt、头文件、源文件……），不单独编译；
                       工程里 ```bash title="…" project="名字" run="yes"``` 的脚本会在该目录里执行，必须成功，后面紧跟的"输出"块参与比对
  lib="yes"            这个 .cpp 不单独编译，而是被别的程序用 with="a.cpp b.cpp" 一起编译链接（演示多个翻译单元）
  flags="-O2 ..."      额外的编译参数；libs="-lfoo" 额外的链接参数

用法：python tools/check_code.py [docs/page.md ...]      （不带参数时检查所有页面）

macOS：默认用 Homebrew 的 LLVM（brew install llvm，自带较新的 libc++，支持 std::jthread），没有时用 Apple clang；
LeakSanitizer 不可用，依赖它的"故意泄漏"例子会跳过；TSan 的锁顺序检测不可靠，那个演示没报出来只记为差异；pybind11 / PyTorch 扩展用苹果自带的 clang 编译（Homebrew 的 LLVM 认不得新 SDK 的 .tbd）；libc++ 与 Linux 上的 libstdc++ 实现细节不同（sizeof(std::string)、
小字符串容量、哈希表的桶数等），输出与页面不一致只记为提示，不算失败。页面上的输出以 Linux + GCC 为准。
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "examples"
MACOS = platform.system() == "Darwin"
BREW_LLVM = next((Path(p) for p in ("/opt/homebrew/opt/llvm", "/usr/local/opt/llvm") if Path(p, "bin", "clang++").exists()), None)
if MACOS:
    CXX = os.environ.get("CXX") or (str(BREW_LLVM / "bin" / "clang++") if BREW_LLVM else shutil.which("clang++") or "clang++")
else:
    CXX = os.environ.get("CXX") or shutil.which("g++") or "g++"
# pybind11 / PyTorch 扩展一章用的 Python：带 torch（CPU 版即可）、pybind11、ninja，见本书 README
PYTHON = os.environ.get("PYTHON") or str(next((p for p in [ROOT / ".venv-py" / "bin" / "python"] if p.exists()), "python3"))
BASE_FLAGS = ["-std=c++20", "-g", "-O1", "-Wall", "-Wextra", "-pthread"]
# 新内核把 ASLR 的随机位数调到了 32（vm.mmap_rnd_bits），TSan 的影子内存映射会落空，
# 报 "FATAL: ThreadSanitizer: unexpected memory mapping"。用 setarch -R 关掉这一个进程的
# ASLR 就能绕过去（不需要 root；另一种办法是 sysctl -w vm.mmap_rnd_bits=28）。
def _tsan_launch() -> list:
    if MACOS or not shutil.which("setarch"):
        return []
    pre = ["setarch", platform.machine(), "-R"]
    # 容器的默认 seccomp 规则可能不让调 personality(ADDR_NO_RANDOMIZE)，先拿 true 试一下
    return pre if subprocess.run([*pre, "true"], capture_output=True).returncode == 0 else []


TSAN_LAUNCH = _tsan_launch()
if MACOS:
    BASE_FLAGS.append("-fexperimental-library")            # libc++ 里 std::jthread / stop_token 还需要这个开关
    if BREW_LLVM and not os.environ.get("CXX"):              # 链接 Homebrew LLVM 自带的 libc++，而不是系统里较旧的那份
        BASE_FLAGS += [f"-L{BREW_LLVM}/lib/c++", f"-Wl,-rpath,{BREW_LLVM}/lib/c++"]
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')


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
        while j < len(lines) and not (lines[j].strip() == fence):
            j += 1
        body = "\n".join(l[len(indent):] if l.startswith(indent) else l.lstrip() for l in lines[i + 1:j])
        yield {"lang": m["lang"], "attrs": dict(ATTR.findall(m["rest"])), "body": body + "\n", "line": i + 1}
        i = j + 1


NOTES: list[str] = []        # macOS 上的非致命差异


def check_page(md: Path) -> list[str]:
    errors = []
    items = list(blocks(md))
    work = BUILD / md.relative_to(ROOT / "docs").with_suffix("")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for b in items:                                    # 先写出所有头文件、被链接的源文件和工程文件
        title = b["attrs"].get("title", "")
        if "project" in b["attrs"] and title and b["attrs"].get("run") != "yes":
            dst = work / b["attrs"]["project"] / title
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(b["body"], encoding="utf-8")
        elif b["lang"] == "cpp" and (title.endswith((".hpp", ".h")) or b["attrs"].get("lib") == "yes"):
            (work / title).write_text(b["body"], encoding="utf-8")
    for k, b in enumerate(items):                      # 运行工程的构建 / 测试脚本
        a = b["attrs"]
        if b["lang"] != "bash" or a.get("run") != "yes" or "project" not in a:
            continue
        where = f"{md.relative_to(ROOT)}:{b['line']} {a.get('title', '')}"
        try:
            r = subprocess.run(["bash", "-euo", "pipefail", "-c", b["body"]], cwd=work / a["project"], capture_output=True,
                               text=True, timeout=900, env=dict(os.environ, PYTHON=PYTHON, PATH=f"{Path(PYTHON).parent}:{os.environ.get('PATH', '')}",
                                        **({"CXX": "/usr/bin/clang++", "CC": "/usr/bin/clang"} if MACOS else {})))
        except subprocess.TimeoutExpired:
            errors.append(f"{where}: 运行超过 600 秒")
            continue
        if r.returncode:
            errors.append(f"{where}: 脚本失败（返回码 {r.returncode}）\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if nxt and nxt["lang"] == "text" and nxt["attrs"].get("title") == "输出":
            want = [l.rstrip() for l in nxt["body"].rstrip("\n").splitlines()]
            got = [l.rstrip() for l in r.stdout.rstrip("\n").splitlines()]
            if want != got:
                msg = f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got)
                (NOTES if MACOS else errors).append(msg)
    for k, b in enumerate(items):
        title = b["attrs"].get("title", "")
        if b["lang"] != "cpp" or not title.endswith(".cpp") or b["attrs"].get("lib") == "yes" or "project" in b["attrs"]:
            continue
        a = b["attrs"]
        where = f"{md.relative_to(ROOT)}:{b['line']} {title}"
        src = work / title
        src.write_text(b["body"], encoding="utf-8")
        exe = work / (title[:-4] + ".out")
        san = a.get("sanitize", "address,undefined")
        flags = BASE_FLAGS + ([] if san == "none" else [f"-fsanitize={san}", "-fno-omit-frame-pointer"])
        if a.get("expect") not in ("fail", "compile-error"):
            flags.append("-Werror")
        flags += shlex.split(a.get("flags", ""))
        others = [str(work / f) for f in a.get("with", "").split()]
        cmd = [CXX, *flags, f"-I{work}", str(src), *others, "-o", str(exe), *shlex.split(a.get("libs", ""))]
        c = subprocess.run(cmd, capture_output=True, text=True)
        if a.get("expect") == "compile-error":
            if not c.returncode:
                errors.append(f"{where}: 这个例子应该编译失败，但编译通过了")
            elif a.get("error") and a["error"] not in c.stderr:
                errors.append(f"{where}: 编译报错里没有「{a['error']}」\n{c.stderr[-2000:]}")
            continue
        if c.returncode:
            errors.append(f"{where}: 编译失败\n{c.stderr[-3000:]}")
            continue
        if a.get("run") == "no":
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if MACOS and a.get("expect") == "fail" and nxt and "LeakSanitizer" in nxt["attrs"].get("title", "") + nxt["body"]:
            NOTES.append(f"{where}: macOS 上没有 LeakSanitizer，跳过这个泄漏演示（可以用 `leaks --atExit -- ./a.out` 检查）")
            continue
        # tsan.supp：抑制 libstdc++ 未插桩导致的 exception_ptr 引用计数误报（见文件内说明）
        env = dict(os.environ, ASAN_OPTIONS=f"detect_leaks={0 if MACOS else 1}:abort_on_error=0", UBSAN_OPTIONS="halt_on_error=1",
                   TSAN_OPTIONS=f"halt_on_error=1:detect_deadlocks=1:suppressions={Path(__file__).resolve().parent / 'tsan.supp'}")
        try:
            launch = TSAN_LAUNCH if "thread" in san else []
            r = subprocess.run([*launch, str(exe)], capture_output=True, text=True,
                               timeout=120, cwd=work, env=env)
        except subprocess.TimeoutExpired:
            errors.append(f"{where}: 运行超过 120 秒")
            continue
        failed = r.returncode != 0 or "runtime error" in r.stderr or "WARNING: ThreadSanitizer" in r.stderr
        if a.get("expect") == "fail":
            if not failed:
                if MACOS and nxt and "lock-order-inversion" in nxt["body"]:
                    NOTES.append(f"{where}: macOS 上的 ThreadSanitizer 没有报出锁顺序反转（它的死锁检测在 Apple 平台上不可靠），"
                                 "这个演示在 Linux 上会被抓到")
                else:
                    errors.append(f"{where}: 这个例子应该失败（sanitizer 报错），但正常结束了")
            continue
        if failed:
            hint = ("\n（这台机器的 ASLR 位数让 TSan 映射不了影子内存。装上 util-linux 的 setarch，"
                    "或者 sysctl -w vm.mmap_rnd_bits=28）" if "unexpected memory mapping" in r.stderr else "")
            errors.append(f"{where}: 运行失败（返回码 {r.returncode}）{hint}\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        if nxt and nxt["lang"] == "text" and nxt["attrs"].get("title") == "输出":
            want = [l.rstrip() for l in nxt["body"].rstrip("\n").splitlines()]
            got = [l.rstrip() for l in r.stdout.rstrip("\n").splitlines()]
            if want != got:
                msg = f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got)
                (NOTES if MACOS else errors).append(msg)
    return errors


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        results = list(ex.map(check_page, pages))
    n = sum(1 for p in pages for b in blocks(p)
            if b["lang"] == "cpp" and b["attrs"].get("title", "").endswith(".cpp") and b["attrs"].get("lib") != "yes"
            and "project" not in b["attrs"])
    errs = [e for r in results for e in r]
    for e in errs:
        print("✗", e, "\n")
    for note in NOTES:
        print("·", note, "\n")
    extra = f"，{len(NOTES)} 条 macOS 上的差异提示（标准库实现不同或平台不支持，不算失败）" if NOTES else ""
    print(f"{len(pages)} 个页面，{n} 个 C++ 程序，{len(errs)} 个问题{extra}（{CXX}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
