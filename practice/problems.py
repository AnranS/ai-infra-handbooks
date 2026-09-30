"""读取题库：practice/problems/<手册>/<题目>/ 下的 problem.md、starter.py、solution.py、test.py。

problem.md 的开头是一段简单的元数据（不依赖 PyYAML）：

    ---
    title: LRU 缓存
    chapter: core/containers.md        # 对应手册里的章节
    difficulty: 中等                    # 简单 / 中等 / 困难
    tags: [OrderedDict, 哈希表]
    requires: [numpy]                  # 可选：numpy、triton、local、torch、cuda
    sanitize: thread                   # 可选，只用于 C++ 题：用 ThreadSanitizer（默认 ASan + UBSan）
    ---
    题目描述（Markdown，支持 $公式$）……

    <!-- 题解 -->
    题解的讲解（可选），和 solution.py 一起在"题解"页显示。

CUDA C++ 题另有 starter.cu、solution.cu、test.cu（在 WSL2 + NVIDIA GPU 上用 nvcc 编译运行）。
C++ 题用 starter.cpp、solution.cpp、test.cpp 代替三个 .py 文件（本地用 g++ / clang++ 加 sanitizer 判题）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
PROBLEMS = ROOT / "problems"
BOOKS = [("python", "Python 进阶"), ("cpp", "C++ 进阶"), ("llm", "大模型原理"), ("cuda", "CUDA 进阶"), ("train", "分布式训练"), ("serving", "推理系统"),
         ("minisgl", "手写 mini-sglang"), ("cs", "计算机基础")]
DIFFICULTY = {"简单": 1, "中等": 2, "困难": 3}
SOLUTION_MARK = "<!-- 题解 -->"


@dataclass
class Chapter:
    book: str
    path: str          # core/containers.md
    title: str
    part: str
    order: int

    @property
    def url(self) -> str:
        return f"{self.book}/{self.path[:-3]}/" if not self.path.endswith("index.md") else f"{self.book}/"


@dataclass
class Problem:
    slug: str
    book: str
    dir: Path
    title: str
    chapter: str
    difficulty: str
    tags: list = field(default_factory=list)
    requires: list = field(default_factory=list)
    description: str = ""
    explanation: str = ""
    starter: str = ""
    solution: str = ""
    tests: str = ""
    cuda: dict | None = None
    lang: str = "python"      # python 或 cpp
    sanitize: str = ""        # C++ 题：thread 表示用 ThreadSanitizer
    number: int = 0

    @property
    def all_requires(self) -> list[str]:
        """元数据里写的依赖，加上从代码里的 import 自动推断出来的（numpy、triton）。"""
        req = set(self.requires)
        code = self.tests + "\n" + self.solution + "\n" + self.starter
        if re.search(r"^\s*(import|from)\s+(numpy|gpusim|tritonkit|triton)\b", code, re.M):
            req.add("numpy")
        if re.search(r"^\s*(import|from)\s+(triton|tritonkit)\b", code, re.M):
            req.add("triton")
        return sorted(req)

    @property
    def env(self) -> str:
        """最低运行环境：browser（浏览器即可）、local（本地 Python，例如要用线程）、torch（需要 PyTorch）、cuda（需要 NVIDIA GPU）、
        cpp（C++ 题，本地编译）。"""
        if self.lang == "cpp":
            return "cpp"
        if "cuda" in self.requires:
            return "cuda"
        if "torch" in self.requires:
            return "torch"
        if "local" in self.requires:
            return "local"
        return "browser"


def parse_nav(book: str) -> list[Chapter]:
    text = (REPO / book / "mkdocs.yml").read_text(encoding="utf-8")
    nav = text[text.index("\nnav:") + 1:]
    chapters, part = [], ""
    for line in nav.splitlines()[1:]:
        if line and not line.startswith(" "):
            break
        m = re.match(r"^  - (.+):\s*$", line)
        if m:
            part = m.group(1).strip()
            continue
        m = re.match(r"^\s+- (.+):\s*(\S+\.md)\s*$", line)
        if m:
            chapters.append(Chapter(book, m.group(2), m.group(1).strip(), part, len(chapters)))
    return chapters


def _front_matter(text: str, path: Path) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: 缺少开头的 --- 元数据")
    end = text.index("\n---\n", 4)
    meta = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, value = line.partition(":")
        value = value.split(" #")[0].strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value
    return meta, text[end + 5:]


def load_problem(d: Path, book: str) -> Problem:
    meta, body = _front_matter((d / "problem.md").read_text(encoding="utf-8"), d / "problem.md")
    desc, _, expl = body.partition(SOLUTION_MARK)
    read = lambda name: (d / name).read_text(encoding="utf-8") if (d / name).exists() else ""  # noqa: E731
    lang = "cpp" if (d / "test.cpp").exists() else "python"
    ext = ".cpp" if lang == "cpp" else ".py"
    p = Problem(slug=d.name, book=book, dir=d, title=meta["title"], chapter=meta["chapter"],
                difficulty=meta.get("difficulty", "中等"), tags=meta.get("tags", []), requires=meta.get("requires", []),
                description=desc.strip(), explanation=expl.strip(), starter=read("starter" + ext),
                solution=read("solution" + ext), tests=read("test" + ext), lang=lang, sanitize=meta.get("sanitize", ""))
    if (d / "test.cu").exists():
        p.cuda = {"starter": read("starter.cu"), "solution": read("solution.cu"), "tests": read("test.cu")}
    if p.difficulty not in DIFFICULTY:
        raise ValueError(f"{d}: difficulty 只能是 简单 / 中等 / 困难")
    return p


def load_all() -> tuple[list[Problem], dict]:
    chapters = {b: {c.path: c for c in parse_nav(b)} for b, _ in BOOKS}
    problems = []
    for bi, (book, _) in enumerate(BOOKS):
        base = PROBLEMS / book
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if (p / "problem.md").exists()):
            p = load_problem(d, book)
            if p.chapter not in chapters[book]:
                raise ValueError(f"{d}: 章节 {p.chapter} 不在 {book}/mkdocs.yml 的导航里")
            problems.append(p)
    order = {b: i for i, (b, _) in enumerate(BOOKS)}
    problems.sort(key=lambda p: (order[p.book], chapters[p.book][p.chapter].order, DIFFICULTY[p.difficulty], p.slug))
    seen = set()
    for i, p in enumerate(problems, 1):
        if p.slug in seen:
            raise ValueError(f"题目 id 重复：{p.slug}")
        seen.add(p.slug)
        p.number = i
    return problems, chapters


def find(slug_or_number: str) -> Problem:
    problems, _ = load_all()
    for p in problems:
        if p.slug == slug_or_number or str(p.number) == str(slug_or_number).lstrip("#0") or \
                f"{p.number:03d}" == slug_or_number:
            return p
    raise SystemExit(f"没有找到题目：{slug_or_number}（用 python practice/judge.py list 查看题目列表）")
