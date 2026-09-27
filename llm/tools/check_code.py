"""Run the Python examples embedded in docs/ with CPU PyTorch.

Conventions in the Markdown:
  ```python title="name.py"   a module file: written to build/code/ (importable by later pages), byte-compiled
  ```python                   runnable code, executed in order in one namespace per page
  ```pycon                    REPL session, checked with doctest (ELLIPSIS + NORMALIZE_WHITESPACE)
  ```py / ```bash / ...       illustrative, not executed

Pages run from the project root, so they can load models/Qwen2.5-0.5B-Instruct.
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
PY = ROOT / ".venv-llm" / "bin" / "python"
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
TITLE = re.compile(r'title="([^"]+)"')

RUNNER_HEAD = '''\
import doctest as __doctest, sys as __sys, torch as __torch
__torch.manual_seed(0)
__failed = 0

def __run(code, where, lineno):
    exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())

def __dt(text, where, lineno):
    global __failed
    test = __doctest.DocTestParser().get_doctest(text, globals(), where, where, lineno - 1)
    runner = __doctest.DocTestRunner(optionflags=__doctest.ELLIPSIS | __doctest.NORMALIZE_WHITESPACE)
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
    with tempfile.TemporaryDirectory() as tmp:
        for md in pages:
            rel = str(md.relative_to(ROOT))
            parts = [RUNNER_HEAD]
            n = 0
            for lang, title, lineno, code in blocks(md):
                if lang == "python" and not title:
                    parts.append(f"__run({code!r}, {rel!r}, {lineno})\n")
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
