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
  @@diagram processes 标题@@                               内联 docs/assets/diagrams/processes.svg
  @@video lifecycle 标题@@                                 嵌入 docs/assets/videos/lifecycle.mp4

路径都相对于本手册目录（minisgl/）。找不到时抛异常，让 mkdocs build --strict 直接失败。

英文版（extra.lang == "en"）的差别：
  - 示意图从本次构建的 docs_dir 取，docs-en/ 里的英文图会覆盖同名的中文图；
  - 代码里的注释和 docstring 按 i18n-en-code.json 换成英文，代码、字符串字面量、行结构都不动
    （缺一条译文就抛异常，让 build --strict 失败；用 tools/code_i18n.py 列出缺的）；
  - examples/ 里打印用的字面量另按 i18n-en-strings.json 换成英文，"运行结果"取
    docs/_outputs_en/（由 tools/outputs_en.py 跑这份译过的脚本生成），所以英文页上的
    代码和输出始终对得上；
  - "运行结果"、"（第 a～b 行）"这些固定文字换成英文。
"""

from __future__ import annotations

import ast
import io
import json
import re
import textwrap
import tokenize
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
UPSTREAM_REPO = "https://github.com/sgl-project/mini-sglang/blob"
_CJK = re.compile(r"[㐀-鿿]")
_CODE_TR: dict = {}
_STR_TR: dict = {}
_CODE = re.compile(r"^(?P<indent>[ \t]*)@@code (?P<path>[^:@\s]+)(?::(?P<sym>[\w.]+))?(?P<opts>[^@]*)@@\s*$", re.M)
_OUTPUT = re.compile(r"^(?P<indent>[ \t]*)@@output (?P<name>[\w-]+)@@\s*$", re.M)
_UPSTREAM = re.compile(r"@@upstream (?P<path>[^:@\s]+)(?::(?P<sym>[\w.]+))?@@")
_DIAGRAM = re.compile(r"^@@diagram (?P<name>[\w-]+)(?: (?P<caption>[^@]+))?@@\s*$", re.M)
_VIDEO = re.compile(r"^@@video (?P<name>[\w-]+)(?: (?P<caption>[^@]+))?@@\s*$", re.M)
_TREE = re.compile(r"^@@tree@@\s*$", re.M)
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


def _docstring_pos(source: str) -> set:
    """每个模块 / 类 / 函数的 docstring 的 (行, 列)。"""
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                out.add((b[0].value.lineno, b[0].value.col_offset))
    return out


def translate_code(source: str, suffix: str, missing: set | None = None) -> str:
    """只把注释和 docstring 换成英文：代码、字符串字面量、缩进、行数都不变。"""
    if not _CJK.search(source):
        return source
    if suffix not in (".py", ".cu", ".cuh"):
        return source

    def tr(text: str) -> str:
        if not _CJK.search(text):
            return text
        if text in _CODE_TR:
            return _CODE_TR[text]
        if missing is not None:
            missing.add(text)
        elif _CODE_TR:                      # 英文版构建：少一条译文就让 build --strict 失败
            raise KeyError(f"i18n-en-code.json 缺这段注释的英文：{text[:60]!r}"
                           "（跑 tools/code_i18n.py 把缺的列出来）")
        return text

    if suffix != ".py":                                  # C++ / CUDA：只认 // 与 /* */
        return re.sub(r"//[^\n]*|/\*.*?\*/", lambda m: tr(m.group(0)), source, flags=re.S)

    docs = _docstring_pos(source)
    lines = source.splitlines(keepends=True)
    edits = []                                            # (起点, 终点, 新文本)，按字节偏移
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln))
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if not _CJK.search(t.string):
            continue
        if t.type == tokenize.COMMENT or (t.type == tokenize.STRING and t.start in docs):
            a = offsets[t.start[0] - 1] + t.start[1]
            b = offsets[t.end[0] - 1] + t.end[1]
            edits.append((a, b, tr(t.string)))
    out = source
    for a, b, new in reversed(edits):
        out = out[:a] + new + out[b:]
    return out


def _line_offsets(source: str) -> list:
    offs = [0]
    for ln in source.splitlines(keepends=True):
        offs.append(offs[-1] + len(ln))
    return offs


def _string_spans(source: str) -> list:
    """每个字符串字面量在源码里的 (起点, 终点)。Python 3.12 起 f-string 被拆成
    FSTRING_START / MIDDLE / END 三种 token，这里把它们合回一段，3.11 与 3.12+ 结果一致。"""
    offs = _line_offsets(source)

    def at(rc):
        return offs[rc[0] - 1] + rc[1]

    spans, depth, start = [], 0, 0
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        name = tokenize.tok_name.get(t.type, "")
        if name == "FSTRING_START":
            if depth == 0:
                start = at(t.start)
            depth += 1
        elif name == "FSTRING_END":
            depth -= 1
            if depth == 0:
                spans.append((start, at(t.end)))
        elif t.type == tokenize.STRING and depth == 0:
            spans.append((at(t.start), at(t.end)))
    return spans


def translate_strings(source: str, missing: set | None = None) -> str:
    """示例脚本里打印用的中文字面量换成英文（只用于 examples/：英文版的"运行结果"正是
    由这份译过的脚本跑出来的）。没有译文的（当作数据的中文，比如演示分词的例句）原样保留。"""
    if not _CJK.search(source):
        return source
    offs = _line_offsets(source)
    docs = {offs[r - 1] + c for r, c in _docstring_pos(source)}
    edits = []
    for a, b in _string_spans(source):
        text = source[a:b]
        if a in docs or not _CJK.search(text):
            continue
        if text in _STR_TR:
            edits.append((a, b, _STR_TR[text]))
        elif missing is not None:
            missing.add(text)
    out = source
    for a, b, new in reversed(edits):
        out = out[:a] + new + out[b:]
    return out


def _block(indent: str, lang: str, title: str, code: str, opts: str) -> str:
    hl = re.search(r"hl=([\d\- ,]+)", opts)
    attrs = f'title="{title}"' + (f' hl_lines="{hl.group(1).strip()}"' if hl else "")
    fence = f"```{lang} {attrs}\n{code.rstrip()}\n```"
    return textwrap.indent(fence, indent)


def _replace_code(m: re.Match, en: bool = False) -> str:
    path = BOOK / m["path"]
    source = path.read_text(encoding="utf-8")
    if en:
        source = translate_code(source, path.suffix)
        if m["path"].startswith("examples/"):
            source = translate_strings(source)
    lang = _LANG.get(path.suffix, "text")
    title = m["path"].removeprefix("python/")
    if m["sym"]:
        start, end = find_symbol(source, m["sym"])
        code = textwrap.dedent("\n".join(source.splitlines()[start - 1:end]))
        title = f"{title} · {m['sym']}"
    else:
        code = source
    return _block(m["indent"], lang, title, code, m["opts"] or "")


def _replace_output(m: re.Match, en: bool = False) -> str:
    out = BOOK / "docs" / "_outputs" / f"{m['name']}.txt"
    if en and (en_out := BOOK / "docs" / "_outputs_en" / f"{m['name']}.txt").exists():
        out = en_out                       # 英文版的输出由译过标签的同一份脚本跑出来
    text = out.read_text(encoding="utf-8")                      # 输出一个字都不改
    title = "What it prints" if en else "运行结果"
    return textwrap.indent(f"```text title=\"{title}\"\n{text.rstrip()}\n```", m["indent"])


def _replace_upstream(m: re.Match, en: bool = False) -> str:
    index = json.loads((BOOK / "docs" / "_outputs" / "upstream_index.json").read_text())
    key = m["path"] + (f":{m['sym']}" if m["sym"] else "")
    entry = index["symbols"][key]
    url = f"{UPSTREAM_REPO}/{index['commit']}/python/minisgl/{m['path']}"
    label = f"{m['path']}" + (f" · {m['sym']}" if m["sym"] else "")
    if "lines" in entry:
        a, b = entry["lines"]
        url += f"#L{a}-L{b}"
        label += f" (lines {a}-{b})" if en else f"（第 {a}～{b} 行）"
    return f"[`{label}`]({url})"


def _replace_diagram(m: re.Match, docs: Path) -> str:
    """内联 tools/diagrams.py 生成的 SVG（颜色用 CSS 类，跟随亮色 / 暗色主题）。
    docs 是本次构建的 docs_dir：英文版里 docs-en/ 的英文图已经覆盖了同名的中文图。"""
    svg = (docs / "assets" / "diagrams" / f"{m['name']}.svg").read_text(encoding="utf-8")
    caption = f"<figcaption>{m['caption'].strip()}</figcaption>" if m["caption"] else ""
    return f'<figure class="dg" markdown="0">{svg.strip()}{caption}</figure>'


def _replace_video(m: re.Match, page) -> str:
    """嵌入 tools/videos/ 生成的教学视频（docs/assets/videos/<名字>.mp4 与同名封面 .jpg）。"""
    name = m["name"]
    for ext in ("mp4", "jpg"):
        if not (BOOK / "docs" / "assets" / "videos" / f"{name}.{ext}").exists():
            raise FileNotFoundError(f"assets/videos/{name}.{ext}")
    prefix = "../" * page.url.count("/")  # "schedule/overlap/" -> "../../"
    caption = f"<figcaption>{m['caption'].strip()}</figcaption>" if m["caption"] else ""
    return (f'<figure class="vid" markdown="0"><video controls preload="none" playsinline '
            f'poster="{prefix}assets/videos/{name}.jpg" src="{prefix}assets/videos/{name}.mp4"></video>'
            f'{caption}</figure>')


def on_config(config):
    """英文版构建时读一次代码注释的译文表。"""
    if (config.get("extra") or {}).get("lang") == "en":
        for name, table in (("i18n-en-code.json", _CODE_TR), ("i18n-en-strings.json", _STR_TR)):
            f = BOOK / name
            table.clear()
            table.update(json.loads(f.read_text(encoding="utf-8")) if f.exists() else {})
    return config


def _render_tree(page_uri: str, en: bool) -> str:
    """这一章为止的 minisgl/ 文件树（tools/steps.py 生成的 steps.json）：本章新建 / 修改 / 提前引入的文件标出来，
    没动过的子包折叠成一行。"""
    steps = json.loads((BOOK / "steps.json").read_text(encoding="utf-8"))
    k = next((i for i, s in enumerate(steps) if s["page"] == page_uri), None)
    if k is None:
        raise KeyError(f"steps.json 里没有 {page_uri}，先跑 python tools/steps.py derive")
    s = steps[k]
    new, modified = set(s["new"]), set(s["modified"])
    preset = {f: ch for f, ch in s["preset"]}
    titles = {i: st["page"] for i, st in enumerate(steps)}
    L = (lambda zh, en_: en_ if en else zh)

    def tag(f: str) -> str:
        if f in new:
            return L("  ← 本章新建", "  ← new in this chapter")
        if f in modified:
            return L("  ← 本章修改", "  ← modified in this chapter")
        if f in preset:
            ch = preset[f]
            n = int(re.search(r"\d+", steps[ch]["main"]).group()) if ch is not None else 0
            return L(f"  ← 提前引入（第 {n} 章讲）", f"  ← brought in early (explained in chapter {n})")
        return ""

    touched = new | modified | set(preset)
    lines = ["minisgl/"]
    by_dir: dict[str, list[str]] = {}
    for f in s["files"]:
        d = f.split("/")[0] if "/" in f else ""
        by_dir.setdefault(d, []).append(f)
    for d in sorted(by_dir, key=lambda x: (x != "", x)):
        fs = by_dir[d]
        if d == "":
            for f in fs:
                lines.append(f"├── {f}{tag(f)}")
            continue
        if not any(f in touched for f in fs):
            lines.append(f"├── {d}/" + L(f"  （{len(fs)} 个文件，不变）", f"  ({len(fs)} files, unchanged)"))
            continue
        lines.append(f"├── {d}/")
        for f in fs:
            lines.append(f"│   ├── {f.split('/', 1)[1]}{tag(f)}")
    n_new, n_mod, n_pre = len(new), len(modified), len(preset)
    title = L(f"到这一章为止的文件：{len(s['files'])} 个（新建 {n_new}，修改 {n_mod}，提前引入 {n_pre}）",
              f"Files so far: {len(s['files'])} ({n_new} new, {n_mod} modified, {n_pre} brought in early)")
    return "```text title=\"" + title + "\"\n" + "\n".join(lines) + "\n```"


def on_page_markdown(markdown: str, page, config, files) -> str:
    en = (config.get("extra") or {}).get("lang") == "en"
    docs = Path(config["docs_dir"])
    markdown = _DIAGRAM.sub(lambda m: _replace_diagram(m, docs), markdown)
    markdown = _VIDEO.sub(lambda m: _replace_video(m, page), markdown)
    markdown = _TREE.sub(lambda m: _render_tree(page.file.src_uri, en), markdown)
    markdown = _CODE.sub(lambda m: _replace_code(m, en), markdown)
    markdown = _OUTPUT.sub(lambda m: _replace_output(m, en), markdown)
    return _UPSTREAM.sub(lambda m: _replace_upstream(m, en), markdown)
