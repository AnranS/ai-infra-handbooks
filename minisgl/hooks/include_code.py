"""MkDocs hook：把正文里的占位符替换成真实的源码和运行结果，保证教程与代码永远一致。

  @@code python/minisgl/core.py@@                         整个文件
  @@code python/minisgl/core.py:Req@@                     某个类或函数（按 AST 定位）
  @@code python/minisgl/scheduler/scheduler.py:Scheduler.overlap_loop@@   类中的方法
  @@code python/minisgl/core.py:Req hl=3-5@@              额外高亮第 3～5 行（相对片段）
  @@output ch07_llm@@                                      examples/ch07_llm.py 的运行输出
                                                           （由 tools/check.py 生成到 docs/_outputs/）
  @@upstream scheduler/scheduler.py:Scheduler.overlap_loop@@
                                                           指向官方仓库对应代码的链接（带行号），
                                                           行号来自 docs/_outputs/upstream_index.json

路径都相对于本手册目录（minisgl/）。找不到时抛异常，让 mkdocs build --strict 直接失败。
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
UPSTREAM_REPO = "https://github.com/sgl-project/mini-sglang/blob"
_CODE = re.compile(r"^(?P<indent>[ \t]*)@@code (?P<path>[^:@\s]+)(?::(?P<sym>[\w.]+))?(?P<opts>[^@]*)@@\s*$", re.M)
_OUTPUT = re.compile(r"^(?P<indent>[ \t]*)@@output (?P<name>[\w-]+)@@\s*$", re.M)
_UPSTREAM = re.compile(r"@@upstream (?P<path>[^:@\s]+)(?::(?P<sym>[\w.]+))?@@")
_LANG = {".py": "python", ".cu": "cuda", ".cuh": "cuda", ".toml": "toml", ".sh": "bash"}


def find_symbol(source: str, symbol: str) -> tuple[int, int]:
    """返回符号在源码中的起止行号（从 1 开始，含装饰器）。"""
    tree = ast.parse(source)
    body = tree.body
    node = None
    for part in symbol.split("."):
        node = next((n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef,
                     ast.AsyncFunctionDef)) and n.name == part), None)
        if node is None:
            raise KeyError(symbol)
        body = node.body
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return start, node.end_lineno


def _block(indent: str, lang: str, title: str, code: str, opts: str) -> str:
    hl = re.search(r"hl=([\d\- ,]+)", opts)
    attrs = f'title="{title}"' + (f' hl_lines="{hl.group(1).strip()}"' if hl else "")
    fence = f"```{lang} {attrs}\n{code.rstrip()}\n```"
    return textwrap.indent(fence, indent)


def _replace_code(m: re.Match) -> str:
    path = BOOK / m["path"]
    source = path.read_text(encoding="utf-8")
    lang = _LANG.get(path.suffix, "text")
    title = m["path"].removeprefix("python/")
    if m["sym"]:
        start, end = find_symbol(source, m["sym"])
        code = textwrap.dedent("\n".join(source.splitlines()[start - 1:end]))
        title = f"{title} · {m['sym']}"
    else:
        code = source
    return _block(m["indent"], lang, title, code, m["opts"] or "")


def _replace_output(m: re.Match) -> str:
    out = BOOK / "docs" / "_outputs" / f"{m['name']}.txt"
    text = out.read_text(encoding="utf-8")
    return textwrap.indent(f"```text title=\"运行结果\"\n{text.rstrip()}\n```", m["indent"])


def _replace_upstream(m: re.Match) -> str:
    index = json.loads((BOOK / "docs" / "_outputs" / "upstream_index.json").read_text())
    key = m["path"] + (f":{m['sym']}" if m["sym"] else "")
    entry = index["symbols"][key]
    url = f"{UPSTREAM_REPO}/{index['commit']}/python/minisgl/{m['path']}"
    label = f"{m['path']}" + (f" · {m['sym']}" if m["sym"] else "")
    if "lines" in entry:
        a, b = entry["lines"]
        url += f"#L{a}-L{b}"
        label += f"（第 {a}～{b} 行）"
    return f"[`{label}`]({url})"


def on_page_markdown(markdown: str, page, config, files) -> str:
    markdown = _CODE.sub(_replace_code, markdown)
    markdown = _OUTPUT.sub(_replace_output, markdown)
    return _UPSTREAM.sub(_replace_upstream, markdown)
