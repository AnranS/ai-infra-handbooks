"""重新运行某一页的代码，把紧跟在代码块后面的输出块更新成实际输出（换模型、升级库之后用）。

用法：
    python3 tools/refresh_outputs.py llm docs/transformer/build-llm.md            # 只报告：哪些输出变了
    python3 tools/refresh_outputs.py llm docs/transformer/build-llm.md --write    # 写回页面

规则：```python 代码块（没有 title 的，即可运行的例子）后面如果紧跟着一个 ```text 块（没有标题或标题是"输出"），
就把它当作这段代码的输出：内容换成实际输出，并加上 title="输出"，以后各手册的 check_code.py 会逐行核对它。
代码没有输出、或者输出块和代码之间隔着正文的，不动。耗时之类每次都不一样的输出，写回之后要手动去掉 title。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
TITLE = re.compile(r'title="([^"]+)"')

RUNNER = '''\
import contextlib, io, json, sys, torch
torch.manual_seed(0)
__outs = []
def __run(code, where, lineno):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile("\\n" * (lineno - 1) + code, where, "exec"), globals())
    __outs.append(buf.getvalue())
'''


def parse(lines: list[str]) -> list[dict]:
    """每个代码块：起止行号（围栏所在行）、语言、标题、正文"""
    out, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, j = m["fence"], i + 1
        while j < len(lines) and not (lines[j].strip().startswith(fence) and set(lines[j].strip()) == {"`"}):
            j += 1
        t = TITLE.search(m["rest"])
        body = [ln[len(m["indent"]):] if ln.startswith(m["indent"]) else ln.lstrip() for ln in lines[i + 1:j]]
        out.append({"start": i, "end": j, "lang": m["lang"], "title": t.group(1) if t else None,
                    "indent": m["indent"], "rest": m["rest"], "body": "\n".join(body) + "\n"})
        i = j + 1
    return out


def write_modules(book_dir: Path) -> None:
    """和各手册的 check_code.py 一样：先把所有页面里 ```python title="xxx.py" 的模块文件写到 build/code/"""
    code_dir = book_dir / "build" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    books = [book_dir] + ([ROOT / "llm"] if book_dir.name != "llm" else [])   # 推理系统手册也用大模型手册的模块
    for d in reversed(books):
        for md in sorted((d / "docs").rglob("*.md")):
            for b in parse(md.read_text(encoding="utf-8").splitlines()):
                if b["lang"] == "python" and b["title"] and b["title"].endswith(".py"):
                    (code_dir / b["title"]).write_text(b["body"], encoding="utf-8")


def main(argv: list[str]) -> int:
    write = "--write" in argv
    book, page = [a for a in argv if not a.startswith("--")]
    book_dir = ROOT / book
    write_modules(book_dir)
    md = (book_dir / page).resolve()
    lines = md.read_text(encoding="utf-8").splitlines()
    blocks = parse(lines)
    runnable = [b for b in blocks if b["lang"] == "python" and not b["title"]]
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "outs.json"
        script = RUNNER + "".join(f"__run({b['body']!r}, {str(md)!r}, {b['start'] + 2})\n" for b in runnable)
        script += f"json.dump(__outs, open({str(dump)!r}, 'w'), ensure_ascii=False)\n"
        (Path(tmp) / "run.py").write_text(script, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(book_dir / "build" / "code"), "OMP_NUM_THREADS": "16",
               "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
        py = book_dir / ".venv-llm" / "bin" / "python"
        p = subprocess.run([str(py if py.exists() else sys.executable), str(Path(tmp) / "run.py")], cwd=book_dir,
                           env=env, capture_output=True, text=True, timeout=3600)
        if p.returncode:
            print(p.stdout[-2000:], p.stderr[-4000:])
            return 1
        outs = json.loads(dump.read_text(encoding="utf-8"))
    changed = 0
    edits = []                                        # (起始行, 结束行, 新的行)
    for b, out in zip(runnable, outs):
        k = blocks.index(b)
        nxt = blocks[k + 1] if k + 1 < len(blocks) else None
        adjacent = nxt is not None and all(not lines[x].strip() for x in range(b["end"] + 1, nxt["start"]))
        if not (adjacent and nxt["lang"] == "text" and nxt["title"] in (None, "输出") and out.strip()):
            if out.strip():
                limit = 100000 if "--show" in argv else 600
                print(f"-- 第 {b['start'] + 1} 行的代码有输出，但后面没有紧跟的输出块：\n   " + out.strip().replace("\n", "\n   ")[:limit])
            continue
        old, new = nxt["body"].rstrip("\n"), out.rstrip("\n")
        if old == new and nxt["title"] == "输出":
            continue
        changed += 1
        print(f"== 第 {nxt['start'] + 1} 行的输出块{'（原来没有标题）' if nxt['title'] is None else ''}：")
        print("   旧：" + old.replace("\n", "\n       ")[:800])
        print("   新：" + new.replace("\n", "\n       ")[:800])
        ind = nxt["indent"]
        edits.append((nxt["start"], nxt["end"], [f'{ind}```text title="输出"'] + [ind + ln if ln else "" for ln in new.split("\n")] + [f"{ind}```"]))
    if write and edits:
        for s, e, new_lines in sorted(edits, reverse=True):
            lines[s:e + 1] = new_lines
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写回 {len(edits)} 个输出块")
    print(f"{md.relative_to(ROOT)}：{len(runnable)} 个代码块，{changed} 个输出块有变化")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
