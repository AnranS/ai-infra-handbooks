"""Run the Python examples embedded in docs/ with CPU PyTorch.

Conventions in the Markdown:
  ```python title="name.py"   a module file: written to build/code/ (importable by later pages), byte-compiled
  ```python                   runnable code, executed in order in one namespace per page; if the next block is
                              ```text title="输出"```, its stdout must match that block line by line
  ```pycon                    REPL session, checked with doctest (ELLIPSIS + NORMALIZE_WHITESPACE)
  ```python ci="loose"        runnable, but in CI (env CI set) its output is not compared: experiments that vary too much across machines
  ```py / ```bash / ...       illustrative, not executed

Pages run from the project root, so they can load models/Qwen3-0.6B.
Usage: .venv-llm/bin/python tools/check_code.py [docs/page.md ...]
"""

import os
import py_compile
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "build" / "code"
PY = Path(os.environ["PYTHON"]) if os.environ.get("PYTHON") else (ROOT / ".venv-llm" / "bin" / "python" if (ROOT / ".venv-llm" / "bin" / "python").exists() else Path(sys.executable))   # 本机用 .venv-llm；CI 里没有它，用 PYTHON 环境变量或当前解释器
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
TITLE = re.compile(r'title="([^"]+)"')

RUNNER_HEAD = '''\
import contextlib as __cl, doctest as __doctest, io as __io, os, re, sys as __sys, torch as __torch
__torch.manual_seed(0)
__failed = 0


NUM = re.compile(r"-?\\d+(?:\\.\\d+)?(?:e[+-]?\\d+)?")


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

def __run(code, where, lineno, expected=None):
    global __failed
    if expected is None:
        exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())
        return
    buf = __io.StringIO()
    with __cl.redirect_stdout(buf):
        exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())
    print(buf.getvalue(), end="")
    want = [l.rstrip() for l in expected.rstrip("\\n").splitlines()]
    got = [l.rstrip() for l in buf.getvalue().rstrip("\\n").splitlines()]
    if not same_output(want, got):
        __failed += 1
        print(f"{where}:{lineno}: 输出和页面不一致\\n--- 页面\\n" + "\\n".join(want) + "\\n--- 实际\\n" + "\\n".join(got), file=__sys.stderr)

class __Checker(__doctest.OutputChecker):             # CI 里 doctest 的数字也走容差
    def check_output(self, want, got, optionflags):
        if super().check_output(want, got, optionflags):
            return True
        return same_output(want.rstrip("\\n").splitlines(), got.rstrip("\\n").splitlines())


def __dt(text, where, lineno):
    global __failed
    test = __doctest.DocTestParser().get_doctest(text, globals(), where, where, lineno - 1)
    runner = __doctest.DocTestRunner(checker=__Checker(), optionflags=__doctest.ELLIPSIS | __doctest.NORMALIZE_WHITESPACE)
    runner.run(test, clear_globs=False)
    globals().update(test.globs)
    __failed += runner.failures

'''


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
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith(fence) and set(s) == {fence[0]}:
                break
            j += 1
        body = [ln[len(indent):] if ln.startswith(indent) else ln.lstrip() for ln in lines[i + 1:j]]
        t = TITLE.search(m["rest"])
        yield m["lang"], (t.group(1) if t else None), i + 2, "\n".join(body) + "\n"
        i = j + 1


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    CODE.mkdir(parents=True, exist_ok=True)
    # module files from every page first, so any page can import them
    for md in sorted((ROOT / "docs").rglob("*.md")):
        for lang, title, lineno, code in blocks(md):
            if lang == "python" and title:
                (CODE / title).write_text(code, encoding="utf-8")
                py_compile.compile(str(CODE / title), doraise=True)

    bad = []
    env = {**os.environ, "PYTHONPATH": str(CODE), "OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
    if sys.platform == "darwin":                  # macOS：分布式的例子用 gloo，让它走回环网卡
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with tempfile.TemporaryDirectory() as tmp:
        for md in pages:
            rel = os.path.relpath(md, ROOT)        # 数学基础手册的页面在 ../math/docs/ 下，也用这套环境核对
            md_lines = md.read_text(encoding="utf-8").splitlines()
            parts = [RUNNER_HEAD]
            n = 0
            items = list(blocks(md))
            for k, (lang, title, lineno, code) in enumerate(items):
                if lang == "python" and not title:
                    nxt = items[k + 1] if k + 1 < len(items) else None
                    expected = nxt[3] if nxt and nxt[0] == "text" and nxt[1] == "输出" else None
                    if expected is not None and os.environ.get("CI") and 'ci="loose"' in md_lines[lineno - 2]:
                        expected = None                  # ci="loose"：随机器变化太大的实验，CI 里只要求跑通
                    parts.append(f"__run({code!r}, {rel!r}, {lineno}, {expected!r})\n")
                    n += 1
                elif lang == "pycon":
                    parts.append(f"__dt({code!r}, {rel!r}, {lineno})\n")
                    n += 1
            if n == 0:
                print(f"[--] {rel} (no runnable blocks)")
                continue
            parts.append("if __failed:\n    __sys.exit(1)\n")
            script = Path(tmp) / (md.stem.replace("-", "_") + "_check.py")
            script.write_text("".join(parts), encoding="utf-8")
            start = time.time()
            p = subprocess.run([str(PY), str(script)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=1800)
            status = "ok" if p.returncode == 0 else "FAIL"
            print(f"[{status}] {rel} ({n} blocks, {time.time() - start:.0f}s)")
            if p.returncode:
                bad.append(rel)
                print(p.stdout[-3000:])
                print(p.stderr[-3000:])

    if not argv and not bad:
        out = ROOT / "docs" / "assets" / "llm-code.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            for f in sorted(CODE.glob("*.py")):
                tar.add(f, arcname=f"llm-code/{f.name}")
        print(f"packed {out.relative_to(ROOT)}")
    print(f"\n{len(pages) - len(bad)}/{len(pages)} pages passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
