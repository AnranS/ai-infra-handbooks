"""重新运行某一页，把 ```pycon 块里对不上的期望输出改成实际输出（换模型、升级库之后用）。

用法：python3 tools/refresh_pycon.py llm docs/basics/language-model.md [--write]
和各手册的 check_code.py 一样：```python 块和 ```pycon 块按顺序在同一个命名空间里执行；pycon 按 doctest 的规则
（ELLIPSIS + NORMALIZE_WHITESPACE）比较，只改写对不上的那几条的期望输出，其余原样保留（包括故意写的 ...）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refresh_outputs import ROOT, parse, write_modules  # noqa: E402

RUNNER = '''\
import contextlib, doctest, io, json, sys, torch
torch.manual_seed(0)
PYCON_FIXES = {}

class __Rec(doctest.DocTestRunner):
    def report_failure(self, out, test, example, got):
        PYCON_FIXES.setdefault(test.name, {})[example.lineno] = got
    def report_unexpected_exception(self, out, test, example, exc_info):
        import traceback
        PYCON_FIXES.setdefault(test.name, {})[example.lineno] = "EXCEPTION: " + "".join(traceback.format_exception_only(*exc_info[:2]))

def __run(code, where, lineno):
    exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())

def __dt(text, name, lineno):
    test = doctest.DocTestParser().get_doctest(text, globals(), name, name, lineno - 1)
    r = __Rec(optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    r.run(test, clear_globs=False, out=lambda s: None)
    globals().update(test.globs)
'''


def main(argv: list[str]) -> int:
    write = "--write" in argv
    book, page = [a for a in argv if not a.startswith("--")]
    book_dir = ROOT / book
    write_modules(book_dir)
    md = (book_dir / page).resolve()
    lines = md.read_text(encoding="utf-8").splitlines()
    blocks = parse(lines)
    parts = [RUNNER]
    for k, b in enumerate(blocks):
        if b["lang"] == "python" and not b["title"]:
            parts.append(f"__run({b['body']!r}, {str(md)!r}, {b['start'] + 2})\n")
        elif b["lang"] == "pycon":
            parts.append(f"__dt({b['body']!r}, 'block{k}', {b['start'] + 2})\n")
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "fixes.json"
        parts.append(f"json.dump(PYCON_FIXES, open({str(dump)!r}, 'w'), ensure_ascii=False)\n")
        (Path(tmp) / "run.py").write_text("".join(parts), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(book_dir / "build" / "code"), "OMP_NUM_THREADS": "16",
               "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
        py = book_dir / ".venv-llm" / "bin" / "python"
        p = subprocess.run([str(py if py.exists() else sys.executable), str(Path(tmp) / "run.py")], cwd=book_dir,
                           env=env, capture_output=True, text=True, timeout=3600)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-4000:])
            return 1
        fixes = json.loads(dump.read_text(encoding="utf-8"))
    import doctest
    edits = 0
    for name, got_by_line in fixes.items():
        k = int(name.removeprefix("block"))
        b = blocks[k]
        pieces = doctest.DocTestParser().parse(b["body"])
        out, line = [], 0
        for piece in pieces:
            if isinstance(piece, str):
                out.append(piece)
                line += piece.count("\n")
                continue
            src = piece.source.rstrip("\n").split("\n")
            text = "\n".join((">>> " if i == 0 else "... ") + s for i, s in enumerate(src)) + "\n"
            want = piece.want
            if str(piece.lineno) in got_by_line:
                new = got_by_line[str(piece.lineno)]
                print(f"== 第 {b['start'] + 2 + piece.lineno} 行：{src[0][:70]}\n   旧：{want.rstrip()[:300]}\n   新：{new.rstrip()[:300]}")
                # doctest 里期望输出中的空行要写成 <BLANKLINE>，否则会被当成输出结束
                want = "\n".join(ln if ln.strip() else "<BLANKLINE>" for ln in new.rstrip("\n").split("\n")) + "\n"
                edits += 1
            out.append(text + want)
        if write:
            new_body = "".join(out).rstrip("\n").split("\n")
            ind = b["indent"]
            lines[b["start"] + 1:b["end"]] = [ind + ln if ln else "" for ln in new_body]
            blocks = parse(lines)                     # 行号变了，重新解析
    if write and edits:
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写回 {edits} 处")
    print(f"{md.relative_to(ROOT)}：{edits} 处期望输出对不上")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
