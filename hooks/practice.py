"""MkDocs hook：在有配套练习题的章节末尾，加上"本章练习题"的链接（指向站点根目录下的 practice/）。

题目的归属写在 practice/problems/<手册>/<题目>/problem.md 的 chapter 字段里；这里只读元数据，不依赖第三方库。
"""

import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("practice_problems", Path(__file__).resolve().parent.parent / "practice" / "problems.py")
_P = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _P
_SPEC.loader.exec_module(_P)
_BY_CHAPTER: dict = {}


def on_config(config):
    _BY_CHAPTER.clear()
    problems, _ = _P.load_all()
    for p in problems:
        _BY_CHAPTER.setdefault((p.book, p.chapter), []).append(p)
    return config


def on_page_markdown(markdown, page, config, files):
    book = (config.get("extra") or {}).get("book")
    items = _BY_CHAPTER.get((book, page.file.src_uri))
    if not items:
        return markdown
    if (config.get("extra") or {}).get("lang") == "en":       # 英文站：题目还只有中文版
        env = {"browser": "", "local": ", local Python", "cpp": ", C++ judged locally", "torch": ", needs PyTorch", "cuda": ", needs an NVIDIA GPU"}
        diff = {"简单": "easy", "中等": "medium", "困难": "hard"}
        intro = ("    C++ exercises are judged locally: `python practice/judge.py test <number>` compiles them with g++ / clang++ and runs them under sanitizers."
                 if all(p.lang == "cpp" for p in items) else
                 "    Write the code and run the tests in your browser, or judge locally with `python practice/judge.py`. The exercises are in Chinese for now.")
        lines = [f'!!! example "Exercises for this chapter ({len(items)})"', intro, ""]
        for p in items:
            extra = env[p.env] + (", plus a CUDA C++ version" if p.cuda else "")
            lines.append(f"    - [{p.number}. {p.title}](root://practice/#/p/{p.slug}) ({diff.get(p.difficulty, p.difficulty)}{extra})")
        return markdown.rstrip() + "\n\n" + "\n".join(lines) + "\n"
    env = {"browser": "", "local": "，需要本地 Python", "cpp": "，C++ 本地判题", "torch": "，需要 PyTorch",
           "cuda": "，需要 NVIDIA GPU"}
    intro = ("    C++ 题在本地判题：`python practice/judge.py test <题号>` 用 g++ / clang++ 编译，并在 sanitizer 下运行。"
             if all(p.lang == "cpp" for p in items) else
             "    在浏览器里直接写代码、跑测试；也可以在本地用 `python practice/judge.py` 判题。")
    lines = [f'!!! example "本章练习题（{len(items)} 道）"', intro, ""]
    for p in items:
        extra = env[p.env] + ("，另有 CUDA C++ 版本" if p.cuda else "")
        lines.append(f"    - [{p.number}. {p.title}](root://practice/#/p/{p.slug})（{p.difficulty}{extra}）")
    return markdown.rstrip() + "\n\n" + "\n".join(lines) + "\n"
