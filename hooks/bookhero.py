"""MkDocs hook：每本手册首页的封面区。

把首页开头的标题和导语（index.md 里的 <h1> 与 <p class="lead">）包进一个封面块，前面放本书的图标，
后面加上章数、练习题数、验证方式，以及「开始阅读」「学习路线」「本书练习题」三个按钮。
章数取导航里的页数，练习题数数 practice/problems/<手册>/ 下的题目，都不用手写；样式在 theme 的 aig.css。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 各书示例代码的验证方式（详见 README 的「代码是怎么验证的」）
VERIFY = {
    "python": "Python 3.14 实跑，doctest 逐字核对",
    "cpp": "g++ 编译，在 ASan / UBSan / TSan 下运行",
    "math": "CPU 上实跑，结论在真实的 Qwen3-0.6B 上测量",
    "llm": "CPU 上实跑，与 HF 官方实现逐项核对",
    "cuda": "nvcc 12.9 / 13.4 编译，CPU 模拟器自检",
    "train": "CPU 多进程（gloo）实跑，与单进程逐项对齐",
    "serving": "源码导读基于 vLLM 0.30 / SGLang 0.5.20",
    "minisgl": "pytest 与 HF transformers 逐 token 对齐",
    "cs": "Python 与 C 程序在 Linux 上实跑",
    "media": "CPU 上用最小配置实跑，不依赖模型权重",
    "sglang": "命令与引用的历史代码在 SGLang 克隆上实跑核对",
}
_HEAD = re.compile(r'\A\s*(<h1\b[^>]*>.*?</h1>)\s*(<p class="lead">.*?</p>)', re.S)


def on_page_context(context, page, config, nav):
    book = (config.get("extra") or {}).get("book")
    m = _HEAD.match(page.content or "") if page.is_homepage else None
    if not book or not m:
        return context
    chapters = [p for p in nav.pages if not p.is_homepage]
    problems = len(list((ROOT / "practice" / "problems" / book).glob("*/problem.md")))
    facts = [f"<span><b>{len(chapters)}</b> 章</span>"]
    if problems:
        facts.append(f"<span><b>{problems}</b> 道练习题</span>")
    facts.append(f"<span>{VERIFY.get(book, '示例代码自动验证')}</span>")
    buttons = []
    if chapters:                                              # 首页就在手册根目录，章节的 url 可以直接当相对链接用
        buttons.append(f'<a class="aig-btn aig-btn--primary" href="{chapters[0].url}">开始阅读<span aria-hidden="true">→</span></a>')
    buttons.append('<a class="aig-btn" href="../roadmap/">学习路线</a>')
    if problems:
        buttons.append(f'<a class="aig-btn" href="../practice/#/?book={book}">本书练习题</a>')
    hero = ('<div class="aig-bookhero">'
            '<img class="aig-bookhero__icon" src="assets/favicon.svg" alt="" width="48" height="48">'
            f'{m.group(1)}{m.group(2)}'
            f'<div class="aig-bookhero__facts">{"".join(facts)}</div>'
            f'<div class="aig-bookhero__cta">{"".join(buttons)}</div>'
            '</div>')
    page.content = hero + page.content[m.end():]
    return context
