"""把各本手册正文里带文件名的代码块导出成真实文件，放到 <书>/examples/ 下。

页面永远是唯一的源：这里只是把它投影成可以直接 clone、阅读和运行的文件。导出器带 --check，
build.sh 每次构建都跑一遍，文件和页面对不上就失败，不会漂。

布局是**每本书一个平铺目录**，和各书的校验脚本把脚本写到同一个工作目录里跑是一致的：
同一本书里同名的文件按页面顺序后面覆盖前面（《从零训练一个小模型》的 budget.py 就是这样），
examples/ 里保留的是最后那一份，README 会标出来。

用法：python3 tools/export_examples.py            # 导出（删掉已经不在页面里的旧文件）
      python3 tools/export_examples.py --check    # 只检查是否一致，不一致返回 1
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# minisgl 不在列：它的代码本来就是一个真实的 Python 包（minisgl/examples/ 是手写的章节脚本），不是从正文抽出来的
BOOKS = ["python", "cpp", "cs", "math", "torch", "llm", "cuda", "scratch", "train", "serving", "media", "sglang"]
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
TITLE = re.compile(r'title="([^"]+)"')
ATTR = re.compile(r'(\w+)="([^"]*)"')
EXTS = {"py", "cu", "cuh", "cpp", "cc", "c", "h", "hpp", "rs", "sh", "cuh", "triton"}
LANGS = {"python", "cuda", "cpp", "c", "rust", "bash", "sh"}
HEADER = "<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->"


def pages(book: str) -> list[Path]:
    """按 mkdocs.yml 的导航顺序列出页面（也就是读者的阅读顺序）；没排进导航的放最后"""
    docs = ROOT / book / "docs"
    nav = (ROOT / book / "mkdocs.yml").read_text(encoding="utf-8").split("\nnav:", 1)[1]
    nav = re.split(r"\n(?=\S)", nav, maxsplit=1)[0]
    order = [docs / m for m in re.findall(r"([\w./-]+\.md)\s*$", nav, re.M)]
    rest = sorted(set(docs.rglob("*.md")) - set(order))
    return [p for p in order if p.exists()] + rest


def code_files(book: str):
    """逐个给出 (页面相对路径, 文件名, 内容, 代码块属性)，按阅读顺序"""
    docs = ROOT / book / "docs"
    for md in pages(book):
        lines = md.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            m = FENCE.match(lines[i])
            if not m:
                i += 1
                continue
            j = i + 1
            while j < len(lines) and lines[j].strip() != m["fence"]:
                j += 1
            t = TITLE.search(m["rest"])
            name = t.group(1) if t else ""
            if m["lang"] in LANGS and "." in name and name.rsplit(".", 1)[1] in EXTS:
                indent = len(m["indent"])
                body = "\n".join(l[indent:] if l.startswith(m["indent"]) else l.lstrip() for l in lines[i + 1:j])
                yield md.relative_to(docs).as_posix(), name, body + "\n", dict(ATTR.findall(m["rest"]))
            i = j + 1


def readme(book: str, rows: list[tuple[str, str, str]]) -> str:
    title = (ROOT / book / "mkdocs.yml").read_text(encoding="utf-8").split("site_name:", 1)[1].split("\n", 1)[0].strip()
    lines = [HEADER, "", f"# 《{title}》正文里的代码", "",
             f"这里的 {len(rows)} 个文件逐字取自正文的代码块，页面是唯一的源；"
             "`tools/export_examples.py` 负责导出，构建时会检查两边一致。", "",
             "怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。", "",
             "| 文件 | 出自 | 备注 |", "| --- | --- | --- |"]
    for name, page, note in rows:
        url = f"https://anrans.github.io/ai-infra-handbooks/{book}/{page[:-3].removesuffix('index')}"
        lines.append(f"| `{name}` | [{page}]({url}) | {note} |")
    return "\n".join(lines) + "\n"


def export(book: str) -> dict[str, str]:
    """返回 {相对路径: 内容}，包含 README.md"""
    out: dict[str, str] = {}
    rows: dict[str, tuple[str, str]] = {}
    for page, name, body, attrs in code_files(book):
        note = []
        if attrs.get("run") == "no":
            note.append("需要 GPU，手册里只做语法检查")
        if name in out and out[name] != body:
            note.append(f"覆盖了 {rows[name][0]} 里的同名文件")
        out[name] = body
        rows[name] = (page, "；".join(note))
    if not out:
        return {}
    out["README.md"] = readme(book, [(n, p, note) for n, (p, note) in sorted(rows.items())])
    return out


def run(check: bool) -> int:
    bad, total = 0, 0
    for book in BOOKS:
        want = export(book)
        d = ROOT / book / "examples"
        have = {p.relative_to(d).as_posix(): p.read_text(encoding="utf-8") for p in d.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts} if d.exists() else {}
        total += len(want)
        if want == have:
            continue
        if check:
            missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
            changed = sorted(k for k in set(want) & set(have) if want[k] != have[k])
            print(f"✗ {book}/examples/ 和正文对不上：缺 {len(missing)}、多 {len(extra)}、内容不同 {len(changed)}")
            for k in (missing + extra + changed)[:5]:
                print(f"    {k}")
            bad += 1
            continue
        for rel in set(have) - set(want):
            (d / rel).unlink()
        for rel, body in want.items():
            f = d / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists() or f.read_text(encoding="utf-8") != body:
                f.write_text(body, encoding="utf-8")
        print(f"{book}/examples/：{len(want) - 1} 个文件")
    if check:
        print(f"examples：{total - len([b for b in BOOKS if export(b)])} 个文件，{'和正文一致' if not bad else '有不一致'}"
              if not bad else "运行 python3 tools/export_examples.py 重新导出")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run("--check" in sys.argv))
