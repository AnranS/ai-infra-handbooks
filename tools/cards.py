"""把各手册里"题目 + 答案"的内容抽成学习卡（各章的练习、面试题库），输出 cards.json 给 portal/cards/ 页面用。

题目：`??? success` 折叠块之前的那道题（从 `**N. ……**` 或 `N. ` 开头的那一行起）；答案：折叠块里缩进 4 格的内容。
各章开头的"自测"提示框如果紧跟着"自测参考答案"折叠块，就按编号把第 N 题和第 N 条答案配成一张卡（类型"自测"）。
Markdown 在构建时渲染成 HTML（公式留给页面上的 KaTeX），链接改写成站内的绝对路径。

用法：python tools/cards.py _site/cards/cards.json    （需要 markdown 和 pymdown-extensions，与 mkdocs 相同的环境）
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ["python", "cpp", "math", "torch", "llm", "cuda", "scratch", "train", "serving", "minisgl", "cs", "media", "sglang", "omni"]
DOCS = {"zh": "docs", "en": "docs-en"}                       # 英文卡片抽自已译的英文页；还没译的页面不出卡
MARKS = {"zh": ('!!! question "自测', '??? success "自测参考答案'),
         "en": ('!!! question "Self-test', '??? success')}   # 英文版各书的答案框标题不统一，按"紧跟在自测框后面"认
KINDS = {"zh": {"self": "自测", "interview": "面试题", "exercise": "练习"},
         "en": {"self": "Self-test", "interview": "Interview", "exercise": "Exercise"}}
QSTART = re.compile(r"^(\*\*\d+[.．]|\d+\.\s)")
LINK = re.compile(r'(href|src)="([^"]+)"')
SITES = "|".join(BOOKS)


FENCE_ATTRS = re.compile(r"^([ \t]*`{3,}[\w+-]*)[^\n`]*$", re.M)


def render(md_text: str) -> str:
    import markdown                                            # 只有生成 cards.json 时才需要（site_stats.py 只数卡片）

    md_text = FENCE_ATTRS.sub(r"\1", md_text)                  # 代码块只留语言，去掉 title="…" 和校验工具用的属性
    return markdown.markdown(md_text, extensions=["tables", "attr_list", "md_in_html", "pymdownx.highlight",
                                                    "pymdownx.superfences", "pymdownx.arithmatex"],
                             extension_configs={"pymdownx.arithmatex": {"generic": True},
                                                "pymdownx.highlight": {"use_pygments": False}})   # 代码不着色，文件小一半


def fix_links(html: str, book: str, page: str, lang: str = "zh") -> str:
    """改写成相对 cards/ 页面的链接：../<书>/<章节>/#锚点（英文卡片页在 en/cards/，门户页仍是中文版，要再退一级）"""
    root_up = "../" if lang == "zh" else "../../"

    def repl(m):
        attr, url = m.groups()
        if re.match(r"^(https?:|mailto:|#)", url):
            return m.group(0)
        cross = re.match(rf"^({SITES})://(.*)$", url)
        if cross:
            return f'{attr}="../{cross.group(1)}/{cross.group(2)}"'
        if url.startswith("root://"):
            return f'{attr}="{root_up}{url[len("root://"):]}"'
        path, _, anchor = url.partition("#")
        base = posixpath.dirname(page)
        target = posixpath.normpath(posixpath.join(base, path)) if path else page
        target = re.sub(r"(index)?\.md$", "", target)
        return f'{attr}="../{book}/{target.rstrip("/") + "/" if target else ""}{"#" + anchor if anchor else ""}"'
    return LINK.sub(repl, html)


def qa_blocks(lines: list[str]):
    """逐个给出 (小节标题, 题目的 Markdown, 答案的 Markdown)"""
    qstart, section = None, ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            section, qstart = line[3:].strip(), None
        elif QSTART.match(line):
            qstart = i
        elif line.startswith("??? success"):
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or lines[j].startswith("    ")):
                j += 1
            if qstart is not None:
                yield section, "\n".join(lines[qstart:i]).strip("\n"), "\n".join(l[4:] for l in lines[i + 1:j]).strip("\n")
            qstart = None
            i = j
            continue
        i += 1


def _numbered(lines: list[str], start: int) -> tuple[list[str], int]:
    """从 start 开始读一个提示框的正文（缩进 4 格），返回其中 `N. ` 开头的各条和正文结束的位置"""
    items, j = [], start
    while j < len(lines) and (lines[j].startswith("    ") or not lines[j].strip()):
        m = re.match(r"^    \d+\.\s+(.*)$", lines[j])
        if m:
            items.append(m.group(1))
        j += 1
    return items, j


def selftest_pairs(lines: list[str], lang: str = "zh"):
    """逐个给出 (自测题, 参考答案)"""
    q_mark, a_mark = MARKS[lang]
    for i, line in enumerate(lines):
        if line.startswith(q_mark):
            qs, j = _numbered(lines, i + 1)
            if j < len(lines) and lines[j].startswith(a_mark):
                answers, _ = _numbered(lines, j + 1)
                if len(answers) == len(qs):
                    yield from zip(qs, answers)


def count_cards(lang: str = "zh") -> int:
    total = 0
    for book in BOOKS:
        for md in (ROOT / book / DOCS[lang]).rglob("*.md"):
            lines = md.read_text(encoding="utf-8").splitlines()
            total += sum(1 for _ in qa_blocks(lines)) + sum(1 for _ in selftest_pairs(lines, lang))
    return total


# 2026-10 搬过家的页面：按旧路径再算一个 id（字段 o），学习卡页面据此把旧的复习记录迁到新 id 上
MOVED = {"math": lambda page: "llm/synthesis/quiz.md" if page == "quiz.md" else f"llm/math/{page}",
         "scratch": lambda page: "train/practice/one-gpu.md" if page == "one-gpu.md" else f"train/scratch/{page}"}


def extract(book: str, md: Path, lang: str = "zh") -> list[dict]:
    page = md.relative_to(ROOT / book / DOCS[lang]).as_posix()
    old = MOVED[book](page) if book in MOVED else None
    lines = md.read_text(encoding="utf-8").splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), page)
    kinds = KINDS[lang]
    cards = []
    for q, answer in selftest_pairs(lines, lang):
        cid = hashlib.sha1(f"{book}/{page}\n自测：{q}".encode()).hexdigest()[:12]
        cards.append({"id": cid, "b": book, "p": page[:-3], "t": title, "s": kinds["self"], "k": kinds["self"],
                      "q": fix_links(render(q), book, page, lang), "a": fix_links(render(answer), book, page, lang)})
        if old:
            cards[-1]["o"] = hashlib.sha1(f"{old}\n自测：{q}".encode()).hexdigest()[:12]
    for section, q, answer in qa_blocks(lines):
        cid = hashlib.sha1(f"{book}/{page}\n{q}".encode()).hexdigest()[:12]
        kind = kinds["interview"] if page.startswith("career/") else kinds["exercise"]
        cards.append({"id": cid, "b": book, "p": page[:-3], "t": title, "s": section, "k": kind,
                      "q": fix_links(render(q), book, page, lang), "a": fix_links(render(answer), book, page, lang)})
        if old:
            cards[-1]["o"] = hashlib.sha1(f"{old}\n{q}".encode()).hexdigest()[:12]
    return cards


def main(out: Path, lang: str = "zh") -> None:
    """英文卡片的 id 沿用中文版（同一道题，两种语言共用一份复习记录）；没译的页面跳过"""
    cards, skipped = [], []
    for book in BOOKS:
        for md in sorted((ROOT / book / "docs").rglob("*.md")):
            zh = extract(book, md, "zh")
            if lang == "zh":
                cards += zh
                continue
            en_md = ROOT / book / "docs-en" / md.relative_to(ROOT / book / "docs")
            if not en_md.exists():
                skipped.append(f"{book}/{md.relative_to(ROOT / book / 'docs')}")
                continue
            en = extract(book, en_md, "en")
            assert len(en) == len(zh), f"{en_md} 的卡片数（{len(en)}）和中文版（{len(zh)}）对不上"
            for card, src in zip(en, zh):
                card["id"] = src["id"]
                if "o" in src:
                    card["o"] = src["o"]
            cards += en
    ids = [c["id"] for c in cards]
    assert len(ids) == len(set(ids)), "卡片 id 重复"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    by_book = {b: sum(c["b"] == b for c in cards) for b in BOOKS}
    note = f"，跳过还没译的 {len(skipped)} 页" if skipped else ""
    print(f"cards{'(en)' if lang == 'en' else ''}: {len(cards)} 张 {by_book}{note} -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    language = "en" if "--lang" in args and args[args.index("--lang") + 1] == "en" else "zh"
    main(Path(args[0]), language)
