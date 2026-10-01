"""全站各处写着的数字（每本书多少章、多少道练习题、面试题库多少题……）都从源文件统计出来，保证处处一致；
顺带检查学习路线图是否恰好覆盖每一章。

用法：python3 tools/site_stats.py          # 只检查：列出与统计值不一致的地方，有就返回 1（CI 里用）
      python3 tools/site_stats.py --fix    # 把不一致的地方改成统计值（build.sh 构建前调用）
      python3 tools/site_stats.py --json   # 打印统计值
      python3 tools/site_stats.py --chapters _site/roadmap/chapters.json   # 输出路线图的章节数据（各章页面的学习条用）
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ["python", "cpp", "llm", "cuda", "train", "serving", "minisgl", "cs"]
BOOK_TITLES = {"python": "Python 进阶手册", "cpp": "C++ 进阶手册", "llm": "大模型原理手册", "cuda": "CUDA 进阶手册",
               "train": "分布式训练手册", "serving": "推理系统手册", "minisgl": "手写 mini-sglang", "cs": "计算机基础手册"}


def nav_pages(book: str) -> list[str]:
    """mkdocs.yml 里 nav 列出的 .md（不含 index.md）。只解析 nav 这一段，不依赖 PyYAML。"""
    text = (ROOT / book / "mkdocs.yml").read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1]
    nav = re.split(r"\n(?=\S)", nav, maxsplit=1)[0]                  # 到下一个顶格的键为止
    pages = re.findall(r"([\w./-]+\.md)\s*$", nav, re.M)
    return [p for p in pages if p != "index.md"]


def minisgl_tests() -> int:
    """pytest 收集到的用例数：测试函数个数，带 parametrize 的按参数组数计"""
    n = 0
    for f in sorted((ROOT / "minisgl" / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                k = 1
                for d in node.decorator_list:
                    if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize" and len(d.args) >= 2:
                        if isinstance(d.args[1], (ast.List, ast.Tuple)):
                            k *= len(d.args[1].elts)
                n += k
    return n


def compute() -> dict[str, int]:
    s: dict[str, int] = {}
    for b in BOOKS:
        s[f"chapters.{b}"] = len(nav_pages(b))
        s[f"pages.{b}"] = s[f"chapters.{b}"] + 1
    s["chapters"] = sum(s[f"chapters.{b}"] for b in BOOKS)
    probs = list((ROOT / "practice" / "problems").glob("*/*/problem.md"))
    s["problems"] = len(probs)
    # 估算题：题库页搜“估算”能搜到的题（标题或标签里有“估算”）
    heads = [p.read_text(encoding="utf-8").split("\n---", 1)[0] for p in probs]
    s["problems.est"] = sum(bool(re.search(r"^(title|tags):.*估算", h, re.M)) for h in heads)
    career = ROOT / "serving" / "docs" / "career"
    s["interview"] = len(re.findall(r"^\*\*\d+\.", (career / "interview.md").read_text(encoding="utf-8"), re.M))
    s["designs"] = sum(len(re.findall(r"^## \d+\.", f.read_text(encoding="utf-8"), re.M)) for f in sorted(career.glob("design-answers-*.md")))
    s["mocks"] = len(re.findall(r"^## 第.套", (career / "mock-exams.md").read_text(encoding="utf-8"), re.M))
    s["tests.minisgl"] = minisgl_tests()
    sys.path.insert(0, str(ROOT / "tools"))
    from cards import count_cards                              # 学习卡：各章练习、面试题库里"题目 + 答案"的个数

    s["cards"] = count_cards()
    return s


# (文件, 正则, 统计项)：正则里第一个分组是要同步的数字；flags 默认 DOTALL
RULES: list[tuple[str, str, str]] = [
    ("portal/index.html", r"共 (\d+) 章，配有", "chapters"),
    ("portal/index.html", r"查看学习路线图：(\d+) 章", "chapters"),
    ("portal/index.html", r"八本手册的 (\d+) 章按", "chapters"),
    *[("portal/index.html", rf'<a class="card [^"]*" href="{b}/">.*?<div class="meta">(\d+) 章', f"chapters.{b}") for b in BOOKS],
    ("portal/index.html", r"(\d+) 个测试，CPU 上可验证", "tests.minisgl"),
    ("portal/index.html", r"<p>(\d+) 道高频题", "interview"),
    ("portal/index.html", r'<div class="meta">(\d+) 题 \+ ', "interview"),
    ("portal/index.html", r"(\d+) 道系统设计的完整参考答案", "designs"),
    ("portal/index.html", r"\+ (\d+) 道系统设计参考答案", "designs"),
    ("portal/index.html", r"(\d+) 套模拟面试卷", "mocks"),
    ("portal/index.html", r"\+ (\d+) 套模拟卷", "mocks"),
    ("portal/index.html", r"配套的 (\d+) 道编程题", "problems"),
    ("portal/index.html", r"抽出的 (\d+) 张学习卡", "cards"),
    ("portal/index.html", r'<div class="meta">(\d+) 张卡 · ', "cards"),
    ("portal/index.html", r'<div class="meta">(\d+) 题 · 简单', "problems"),
    ("portal/roadmap/index.html", r"八本手册的 (\d+) 章排成", "chapters"),
    ("portal/roadmap/index.html", r"<p>(\d+) 章按阶段排列", "chapters"),
    ("portal/setup/index.html", r"(\d+) 个测试在 CPU 上跑", "tests.minisgl"),
    ("portal/setup/index.html", r"(\d+) 个测试在 CPU 上全部通过", "tests.minisgl"),
    ("portal/plan/data.js", r"按主题整理的 (\d+) 题", "interview"),
    ("portal/plan/index.html", r"八本手册、(\d+) 道练习题", "problems"),
    ("tools/search_index.py", r"八本手册的 (\d+) 章按", "chapters"),
    ("tools/search_index.py", r"\"(\d+) 道估算题", "problems.est"),
    ("README.md", r"\*\*(\d+)\*\* 章 · ", "chapters"),
    ("README.md", r"(\d+) 章按 17 周排好", "chapters"),
    ("README.md", r"\*\*(\d+)\*\* 道练习题", "problems"),
    ("README.md", r"\*\*(\d+)\*\* 张学习卡", "cards"),
    ("README.md", r"\*\*(\d+)\*\* 道面试高频题", "interview"),
    ("README.md", r"(\d+) 个 pytest 测试", "tests.minisgl"),
    *[("README.md", rf"\*\*\[{re.escape(t)}\]\([^)]*\)\*\* · (\d+) 章", f"chapters.{b}") for b, t in BOOK_TITLES.items()],
]


def roadmap_chapters() -> list[dict]:
    """按路线图的顺序列出每一章：所在的阶段（w1……）、周次、级别（1 必学 / 2 推荐 / 3 选学）、重点方向"""
    text = (ROOT / "portal/roadmap/index.html").read_text(encoding="utf-8")
    block = text[text.index("  var STAGES = ["):text.index("\n  ];", text.index("  var STAGES = ["))]
    out = []
    for stage in re.split(r"\n    \{ id: ", block)[1:]:
        head = stage.split("\n", 1)[0]
        sid = re.match(r'"(\w+)"', head).group(1)
        m = re.search(r"weeks: \[([\d, ]+)\]", head)
        if m:
            ws = [int(w) for w in m.group(1).split(",")]
            weeks = f"第 {ws[0]} 周" if len(ws) == 1 else f"第 {ws[0]}～{ws[-1]} 周"
        else:                                                         # 与主线并行的阶段（Python 手册）写的是一句说明
            weeks = re.search(r'week: "([^"]*)"', head).group(1)
        m = re.search(r'book: "(\w+)"', head)
        book = m.group(1) if m else None
        for key, title, lv, dirs in re.findall(r'\["([\w:/\-]+)", "([^"]*)", (\d), "([^"]*)"\]', stage):
            b, path = key.split(":", 1) if ":" in key else (book, key)
            out.append({"id": f"{b}/{path}", "book": b, "path": path, "title": title, "lv": int(lv), "dirs": dirs,
                        "stage": sid, "weeks": weeks})
    return out


def roadmap_problems() -> list[str]:
    """学习路线图（portal/roadmap/index.html 的 STAGES）要恰好覆盖每一章各一次，新增章节时别忘了排进某一周"""
    seen = [(c["book"], c["path"]) for c in roadmap_chapters()]
    nav = {(b, p[:-3]) for b in BOOKS for p in nav_pages(b)}
    msgs = [f"路线图里重复出现：{b}/{p}" for b, p in sorted({x for x in seen if seen.count(x) > 1})]
    msgs += [f"路线图里没有排进任何一周：{b}/{p}" for b, p in sorted(nav - set(seen))]
    msgs += [f"路线图里的章节不存在：{b}/{p}" for b, p in sorted(set(seen) - nav)]
    return msgs


def run(fix: bool) -> int:
    stats = compute()
    bad = 0
    for msg in roadmap_problems():
        print("✗ " + msg)
        bad += 1
    texts: dict[str, str] = {}
    for rel, pat, key in RULES:
        text = texts.setdefault(rel, (ROOT / rel).read_text(encoding="utf-8"))
        rx = re.compile(pat, re.S | re.M)
        matches = list(rx.finditer(text))
        if not matches:
            print(f"✗ {rel}：找不到 {pat!r}（文案改过的话，同步修改 tools/site_stats.py 的 RULES）")
            bad += 1
            continue
        want = str(stats[key])
        out, last = [], 0
        for m in matches:
            if m.group(1) != want:
                line = text.count("\n", 0, m.start(1)) + 1
                print(f"{'已改' if fix else '✗'} {rel}:{line}  {key} 写的是 {m.group(1)}，实际 {want}")
                bad += 0 if fix else 1
            out.append(text[last:m.start(1)] + want)
            last = m.end(1)
        texts[rel] = "".join(out) + text[last:]
    if fix:
        for rel, text in texts.items():
            if (ROOT / rel).read_text(encoding="utf-8") != text:
                (ROOT / rel).write_text(text, encoding="utf-8")
    return 1 if bad else 0


def write_chapters(out: Path) -> None:
    chapters = roadmap_chapters()
    data = {"order": [c["id"] for c in chapters],
            "ch": {c["id"]: [c["stage"], c["weeks"], c["lv"], c["dirs"], c["title"]] for c in chapters}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    if "--chapters" in sys.argv:
        write_chapters(Path(sys.argv[sys.argv.index("--chapters") + 1]))
        sys.exit(0)
    if "--json" in sys.argv:
        print(json.dumps(compute(), ensure_ascii=False, indent=1))
        sys.exit(0)
    sys.exit(run("--fix" in sys.argv))
