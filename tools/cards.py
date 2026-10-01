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
BOOKS = ["python", "cpp", "llm", "cuda", "train", "serving", "minisgl", "cs", "media"]
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


def fix_links(html: str, book: str, page: str) -> str:
    """改写成相对 cards/ 页面的链接：../<书>/<章节>/#锚点"""
    def repl(m):
        attr, url = m.groups()
        if re.match(r"^(https?:|mailto:|#)", url):
            return m.group(0)
        cross = re.match(rf"^({SITES})://(.*)$", url)
        if cross:
            return f'{attr}="../{cross.group(1)}/{cross.group(2)}"'
        if url.startswith("root://"):
            return f'{attr}="../{url[len("root://"):]}"'
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


def selftest_pairs(lines: list[str]):
    """逐个给出 (自测题, 参考答案)"""
    for i, line in enumerate(lines):
        if line.startswith('!!! question "自测'):
            qs, j = _numbered(lines, i + 1)
            if j < len(lines) and lines[j].startswith('??? success "自测参考答案'):
                answers, _ = _numbered(lines, j + 1)
                if len(answers) == len(qs):
                    yield from zip(qs, answers)


def count_cards() -> int:
    total = 0
    for book in BOOKS:
        for md in (ROOT / book / "docs").rglob("*.md"):
            lines = md.read_text(encoding="utf-8").splitlines()
            total += sum(1 for _ in qa_blocks(lines)) + sum(1 for _ in selftest_pairs(lines))
    return total


def extract(book: str, md: Path) -> list[dict]:
    page = md.relative_to(ROOT / book / "docs").as_posix()
    lines = md.read_text(encoding="utf-8").splitlines()
    title = next((l[2:].strip() for l in lines if l.startswith("# ")), page)
    cards = []
    for q, answer in selftest_pairs(lines):
        cid = hashlib.sha1(f"{book}/{page}\n自测：{q}".encode()).hexdigest()[:12]
        cards.append({"id": cid, "b": book, "p": page[:-3], "t": title, "s": "自测", "k": "自测",
                      "q": fix_links(render(q), book, page), "a": fix_links(render(answer), book, page)})
    for section, q, answer in qa_blocks(lines):
        cid = hashlib.sha1(f"{book}/{page}\n{q}".encode()).hexdigest()[:12]
        kind = "面试题" if page.startswith("career/") else "练习"
        cards.append({"id": cid, "b": book, "p": page[:-3], "t": title, "s": section, "k": kind,
                      "q": fix_links(render(q), book, page), "a": fix_links(render(answer), book, page)})
    return cards


def main(out: Path) -> None:
    cards = []
    for book in BOOKS:
        for md in sorted((ROOT / book / "docs").rglob("*.md")):
            cards += extract(book, md)
    ids = [c["id"] for c in cards]
    assert len(ids) == len(set(ids)), "卡片 id 重复"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    by_book = {b: sum(c["b"] == b for c in cards) for b in BOOKS}
    print(f"cards: {len(cards)} 张 {by_book} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
