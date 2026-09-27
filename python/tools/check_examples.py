"""Run the code examples in docs/ so the site never shows broken code.

Conventions used in the Markdown:
  ```python  runnable code, executed in order in one namespace per page
  ```pycon   REPL session, checked with doctest (ELLIPSIS enabled)
  ```py      illustrative only (partial code, file layouts, deliberate errors), skipped

Usage: .venv-check/bin/python tools/check_examples.py [docs/some/page.md ...]
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<lang>[\w+-]*)")

RUNNER_HEAD = '''\
import doctest as __doctest, sys as __sys

__failed = 0

def __run(code, where, lineno):
    exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())

def __dt(text, where, lineno):
    global __failed
    if __name__ != "__main__":
        return
    test = __doctest.DocTestParser().get_doctest(text, globals(), where, where, lineno - 1)
    runner = __doctest.DocTestRunner(optionflags=__doctest.ELLIPSIS | __doctest.NORMALIZE_WHITESPACE)
    runner.run(test, clear_globs=False)
    globals().update(test.globs)   # DocTest works on a copy; keep names for later blocks
    __failed += runner.failures

'''


def blocks(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence, lang = m["indent"], m["fence"], m["lang"]
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith(fence) and set(s) == {fence[0]}:
                break
            j += 1
        body = [ln[len(indent):] if ln.startswith(indent) else ln.lstrip() for ln in lines[i + 1:j]]
        yield lang, i + 2, "\n".join(body) + "\n"
        i = j + 1


def build_script(md: Path) -> str:
    rel = str(md.relative_to(ROOT))
    parts = [RUNNER_HEAD]
    for lang, lineno, code in blocks(md.read_text(encoding="utf-8")):
        if lang == "python":
            parts.append(f"__run({code!r}, {rel!r}, {lineno})\n")
        elif lang == "pycon":
            parts.append(f"__dt({code!r}, {rel!r}, {lineno})\n")
    parts.append("if __name__ == '__main__' and __failed:\n    __sys.exit(1)\n")
    return "".join(parts)


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for md in pages:
            script = Path(tmp) / (md.stem.replace("-", "_") + "_check.py")
            script.write_text(build_script(md), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(script)], cwd=tmp, capture_output=True, text=True, timeout=300
            )
            status = "ok" if proc.returncode == 0 else "FAIL"
            print(f"[{status}] {md.relative_to(ROOT)}")
            if proc.returncode != 0:
                bad.append(md)
                print(proc.stdout[-4000:])
                print(proc.stderr[-4000:])
    print(f"\n{len(pages) - len(bad)}/{len(pages)} pages passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
