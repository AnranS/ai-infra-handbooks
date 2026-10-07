"""MkDocs hook: export every chapter that has runnable Python into a Jupyter notebook, and link it from the page.

Each code block becomes a cell, following the conventions of the handbooks' code checkers:

    ```python title="x.py"               a file: `%%writefile x.py`; in the training and CUDA books it is a complete
                                          script and is also run (`!python x.py`, or `!torchrun ... x.py` for torchrun="N",
                                          not at all for run="no")
    ```python                            a code cell (in the training and CUDA books, an untitled block is a fragment
                                          and stays in the Markdown)
    ```pycon                             code cells with the `>>> ` / `... ` prompts removed, split after every
                                          statement that shows output; a statement shown raising is wrapped in try
    ```text title="输出"                  the expected output, as a Markdown cell after the code
    anything else (bash, cpp, text...)   left in the Markdown cells as it is

In the LLM and serving books a chapter often imports module files written in earlier chapters (the serving book also
uses the LLM book's mini_llm.py); the notebook starts with `%%writefile` cells for every such module it needs, directly
or through other modules, so each notebook runs on its own.

The prose between code blocks becomes Markdown cells: admonitions are turned into block quotes (collapsible ones,
such as answer keys, into <details>), figures and
the interactive widgets into a link to the page, and relative or cross-handbook links into absolute URLs.
Notebooks are written to `<site>/notebooks/<chapter>.ipynb`; `page.meta["notebook"]` tells theme/main.html to show a
download chip. Pages whose code is only illustrative (no python / pycon block) get no notebook.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path

SITE = "https://anrans.github.io/ai-infra-handbooks/"
REPO = Path(__file__).resolve().parent.parent
_EN_PORTAL = ("", "roadmap/", "plan/")       # 和 crosslinks.py 的 EN_PORTAL 一致：英文站已有的门户页
_lang = {"en": False}                        # 当前在构建哪种语言（英文站在 en/ 下，链接和说明用英文）
BOOKS = ("python", "cpp", "math", "llm", "cuda", "train", "serving", "minisgl", "media")
# 分布式训练和 CUDA 手册里，带 title 的 python 块是完整的脚本（要运行），不带 title 的是片段；
# 大模型原理和推理系统手册里，带 title 的是模块文件（只写成文件），不带 title 的按顺序运行
SCRIPT_BOOKS = {"train", "cuda"}
FENCE = re.compile(r"^(`{3,}|~{3,})([\w+-]*)(.*)$")
TITLE = re.compile(r'title="([^"]+)"')
TORCHRUN = re.compile(r'torchrun="(\d+)"')
RUN_NO = re.compile(r'run="no"')
LINK = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)(\{[^}]*\})?")
NOTES_EN = {
    "llm": "Run it from the repository's `llm/` directory (it needs `models/Qwen3-0.6B`; see the Setup page). Code cells in a chapter run in order.",
    "math": "Run it from the repository's `llm/` directory: this book uses the LLM book's environment (it needs `models/Qwen3-0.6B`; a few examples read the frozen sample text under `docs/assets/`). The module files it uses, such as `mini_llm.py`, are written by the first cells.",
    "serving": "Run it from the repository's `serving/` directory (it needs `models/Qwen3-0.6B`; see the Setup page). Code cells in a chapter run in order.",
    "train": "Training scripts are written to files with `%%writefile` and then run; multi-process examples use `torchrun` (the gloo backend works on CPU). Scripts in a chapter run in order.",
    "cuda": "The Python scripts in this chapter are written to files with `%%writefile` and then run on CPU (the CPU build of PyTorch is enough); scripts that need a GPU are only written, not run.",
    "python": "Interactive examples (`>>>`) have their prompts removed and became runnable cells (statements that raise on purpose are wrapped in try). The examples target Python 3.14.",
}
NOTES = {
    "math": "放在仓库的 `llm/` 目录下运行：这本书的例子用大模型原理手册的环境（需要 `models/Qwen3-0.6B`，环境见站点的「学习环境」页；部分例子会读取 `docs/assets/` 下冻结的样本文本当语料）；用到的模块文件（如 `mini_llm.py`）已经放在开头的代码格里。",
    "llm": "放在仓库的 `llm/` 目录下运行（需要 `models/Qwen3-0.6B`，环境见站点的「学习环境」页；个别章节会读取 `docs/` 下的书稿当语料）；同一章的代码按顺序执行。",
    "serving": "放在仓库的 `serving/` 目录下运行（需要 `models/Qwen3-0.6B`，环境见站点的「学习环境」页；个别章节会读取 `../llm/docs/` 下的书稿当语料）；同一章的代码按顺序执行。",
    "train": "训练脚本先用 `%%writefile` 写成文件再运行，多进程的例子用 `torchrun`（CPU 上用 gloo 后端即可）；同一章的脚本按顺序执行。",
    "cuda": "这一章的 Python 脚本先用 `%%writefile` 写成文件再运行，CPU 上即可（PyTorch 的例子装 CPU 版就行）；需要 GPU 的脚本只写成文件、不运行。",
    "python": "交互式的例子（`>>>`）已经去掉提示符，改成可以直接运行的代码格（故意报错的语句包在 try 里）。例子按 Python 3.14 写；多进程的例子在 notebook 里要把任务函数放进单独的 .py 文件再导入（macOS 默认用 spawn 启动子进程）。",
}

_pages: dict[str, dict] = {}
_modules: dict[str, dict[str, str]] = {}                  # 每本书可以导入的模块文件：文件名 → 源码（大模型原理、推理系统手册）
IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_]\w*)[\w.]*\s+import|import\s+([A-Za-z_]\w*))", re.M)


def _module_index(book: str, docs_dir: Path) -> dict[str, str]:
    """书里所有 title="x.py" 的模块文件（推理系统手册和数学基础手册还能用大模型原理手册的，比如 mini_llm.py）"""
    if book not in _modules:
        index = {}
        dirs = [docs_dir] + ([REPO / "llm" / "docs"] if book in ("serving", "math") else [])
        for d in dirs:
            for md in sorted(d.rglob("*.md")):
                lines = md.read_text(encoding="utf-8").split("\n")
                for i, ln in enumerate(lines):
                    m = FENCE.match(ln)
                    t = TITLE.search(ln) if m and m.group(2) == "python" else None
                    if t and t.group(1).endswith(".py"):
                        j = i + 1
                        while j < len(lines) and not re.match(r"^`{3,}\s*$", lines[j]):
                            j += 1
                        index.setdefault(t.group(1)[:-3], "\n".join(lines[i + 1:j]))
        _modules[book] = index
    return _modules[book]


def _needed_modules(code: str, own: set[str], index: dict[str, str]) -> list[str]:
    """这一章的代码（直接或间接）导入、但不在本章定义的模块，按依赖顺序排好（被依赖的在前）"""
    order, seen = [], set()

    def visit(name):
        if name in seen or name not in index:
            return
        seen.add(name)
        for a, b in IMPORT.findall(index[name]):
            visit(a or b)
        if name not in own:
            order.append(name)

    for a, b in IMPORT.findall(code):
        visit(a or b)
    return order


def _abs_link(url: str, book: str, page_path: str) -> str:
    if re.match(r"^(https?:|mailto:|#)", url):
        return url
    m = re.match(rf"^({'|'.join(BOOKS)})://(.*)$", url)
    base = SITE + ("en/" if _lang["en"] else "")
    if m:
        return base + m.group(1) + "/" + m.group(2)
    if url.startswith("root://"):
        target = url[len("root://"):]
        first = target.split("#")[0].split("/")[0]
        first = first + "/" if first else ""
        return (base if first in _EN_PORTAL else SITE) + target
    path, _, anchor = url.partition("#")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(page_path), path)) if path else page_path
    target = re.sub(r"(index)?\.md$", "", target)
    return f"{base}{book}/{target.rstrip('/') + '/' if target and target != '.' else ''}{'#' + anchor if anchor else ''}"


def _prose(lines: list[str], book: str, page_path: str, page_url: str) -> str:
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(!!!|\?\?\?\+?)\s+\w+(?:\s+"([^"]*)")?\s*$', line)
        if m:                                              # 提示框：变成引用块
            body, j = [], i + 1
            while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
                body.append(lines[j][4:])
                j += 1
            while body and not body[-1].strip():
                body.pop()
            body = [LINK.sub(lambda mm: f"{mm.group(1)}[{mm.group(2)}]({_abs_link(mm.group(3), book, page_path)})", b) for b in body]
            if m.group(1).startswith("???"):                 # 折叠的提示框（参考答案等）：保持折叠
                out += ["<details>", f"<summary>{m.group(2) or '展开'}</summary>", ""] + body + ["", "</details>", ""]
            else:
                if m.group(2):
                    out.append(f"> **{m.group(2)}**")
                    out.append(">")
                out += [("> " + b).rstrip() for b in body]
                out.append("")
            i = j
            continue
        if re.match(r"^!\[[^\]]*\]\([^)]+\.svg\)\{\s*\.aig-svg\s*\}\s*$", line):
            cap = re.match(r"^!\[([^\]]*)\]", line).group(1) or "图"
            out.append(f"*（{cap}：见[网页版]({page_url})）*")
        elif 'class="aig-widget"' in line:
            out.append(f"*（这里有一个交互小工具，见[网页版]({page_url})）*")
        else:
            line = re.sub(r"</?p[^>]*>", "", line)
            line = LINK.sub(lambda mm: f"{mm.group(1)}[{mm.group(2)}]({_abs_link(mm.group(3), book, page_path)})", line)
            out.append(line)
        i += 1
    return "\n".join(out).strip()


def _pycon_cells(body: str) -> list[str]:
    """把一段交互式会话拆成代码格：每个有输出的语句结束一格（这样格子最后一个表达式的值就是会话里显示的那个）；
    会话里故意抛出异常的语句包进 try，只打印异常"""
    stmts = []                                             # [代码行, 输出行]
    for ln in body.split("\n"):
        if ln.startswith(">>> ") or ln == ">>>":
            stmts.append([[ln[4:]], []])
        elif (ln.startswith("... ") or ln == "...") and stmts:
            stmts[-1][0].append(ln[4:])
        elif stmts:
            stmts[-1][1].append(ln)
    cells, cur = [], []
    for code, out in stmts:
        while code and not code[-1].strip():
            code.pop()
        if out and out[0].startswith("Traceback"):
            cur += ["try:"] + ["    " + c for c in code] + ["except Exception as e:                 # 会话里这一句本来就会报错", "    print(type(e).__name__, e)"]
        else:
            cur += code
        if any(o.strip() for o in out):
            cells.append("\n".join(cur))
            cur = []
    if cur:
        cells.append("\n".join(cur))
    return [c for c in cells if c.strip()]


def _cell(kind: str, text: str) -> dict:
    src = text.split("\n")
    src = [s + "\n" for s in src[:-1]] + [src[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build(book: str, page_path: str, markdown: str, page_url: str, modules: dict[str, str] | None = None) -> dict | None:
    lines = markdown.split("\n")
    h1 = next((k for k, ln in enumerate(lines) if ln.startswith("# ")), None)
    body_lines = lines[:h1] + lines[h1 + 1:] if h1 is not None else lines
    cells, prose, has_code, i = [], [], False, 0

    def flush():
        text = _prose(prose, book, page_path, page_url)
        if text:
            cells.append(_cell("markdown", text))
        prose.clear()

    src_lines, lines = lines, body_lines
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            prose.append(lines[i])
            i += 1
            continue
        fence, lang, rest = m.groups()
        j = i + 1
        while j < len(lines) and not (lines[j].startswith(fence[0] * len(fence)) and not lines[j].strip(fence[0]).strip()):
            j += 1
        body = "\n".join(lines[i + 1:j])
        title = TITLE.search(rest)
        if lang == "python" and (title or book not in SCRIPT_BOOKS):
            flush()
            has_code = True
            if title:
                cells.append(_cell("code", f"%%writefile {title.group(1)}\n{body}"))
                if book in SCRIPT_BOOKS and not RUN_NO.search(rest) and title.group(1).endswith(".py"):
                    n = TORCHRUN.search(rest)
                    run = f"!torchrun --standalone --nproc-per-node {n.group(1)} {title.group(1)}" if n else f"!python {title.group(1)}"
                    cells.append(_cell("code", run))
            else:
                # 校验脚本把模块文件写在 build/code/ 下；notebook 里模块就在当前目录
                cells.append(_cell("code", body.replace('"build/code/', '"')))
        elif lang == "pycon":
            flush()
            for code in _pycon_cells(body):
                has_code = True
                cells.append(_cell("code", code))
        elif lang == "text" and title and title.group(1) == "输出":
            flush()
            cells.append(_cell("markdown", "**输出**（本书环境里的实际输出）：\n\n```text\n" + body + "\n```"))
        else:
            prose.extend(lines[i:j + 1])
        i = j + 1
    flush()
    if not has_code:
        return None
    title = next((ln[2:].strip() for ln in src_lines if ln.startswith("# ")), page_path[:-3])
    if _lang["en"]:
        head = f"# {title}\n\nThe [web version]({page_url}) of this chapter. {NOTES_EN.get(book, '')}".strip()
    else:
        head = f"# {title}\n\n本章的[网页版]({page_url})。{NOTES.get(book, '')}".strip()
    if modules and book not in SCRIPT_BOOKS:
        code = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
        own = set(re.findall(r"^%%writefile (\w+)\.py", code, re.M))
        need = _needed_modules(code, own, modules)
        if need:
            prep = [_cell("markdown", ("**Setup: modules from other chapters.** Run the next cells to write "
                                       + ", ".join(f"`{n}.py`" for n in need) + " to the current directory so the code below can import them.")
                          if _lang["en"] else
                          ("**准备：本章用到的其他章节的模块。** 运行下面几格，会在当前目录写出 "
                           + "、".join(f"`{n}.py`" for n in need) + "，之后的代码才能导入它们。"))]
            prep += [_cell("code", f"%%writefile {n}.py\n{modules[n]}") for n in need]
            cells[0:0] = prep
    cells.insert(0, _cell("markdown", head))
    for k, c in enumerate(cells):
        c["id"] = f"cell-{k}"
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def on_page_markdown(markdown, page, config, files):
    book = (config.get("extra") or {}).get("book") or Path(config["docs_dir"]).parent.name
    _lang["en"] = (config.get("extra") or {}).get("lang") == "en"
    src = Path(page.file.abs_src_path).read_text(encoding="utf-8")
    if not re.search(r"^```(python|pycon)", src, re.M):
        return markdown
    rel = page.file.src_path
    modules = _module_index(book, Path(config["docs_dir"])) if book in ("llm", "serving", "math") else None
    nb = build(book, rel, src, SITE + ("en/" if _lang["en"] else "") + book + "/" + page.url, modules)
    if nb is not None:
        nb_path = "notebooks/" + rel[:-3] + ".ipynb"
        _pages[nb_path] = nb
        page.meta["notebook"] = nb_path
    return markdown


def on_post_build(config):
    site = Path(config["site_dir"])
    for nb_path, nb in _pages.items():
        out = site / nb_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    _pages.clear()
    _modules.clear()
