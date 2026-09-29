#!/usr/bin/env python3
"""把题库编译成静态站点：python practice/build.py _site/practice

输出 index.html / app.js / app.css / worker.js、runtime/*.py（浏览器里的判题内核），
data/index.json（题目列表、章节）和 data/p/<id>.json（每道题的描述、模板、测试、题解）。
只用标准库，CI 和本地都能跑。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import problems as P  # noqa: E402

RUNTIME = ["judge_runner.py", "checker.py", "gpusim.py", "minitl.py", "tritonkit.py", "assist.py", "playground.py"]


def main(out: Path) -> None:
    problems, chapters = P.load_all()
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "p").mkdir(parents=True)
    (out / "runtime").mkdir()
    for f in (HERE / "app").iterdir():
        shutil.copy(f, out / f.name)
    for name in RUNTIME:
        shutil.copy(HERE / "runtime" / name, out / "runtime" / name)

    digest = hashlib.sha1()
    for f in sorted(list((HERE / "runtime").glob("*.py")) + list((HERE / "app").iterdir())):
        digest.update(f.read_bytes())
    items = []
    used = {(p.book, p.chapter) for p in problems}
    for p in problems:
        req = p.all_requires
        data = {"slug": p.slug, "number": p.number, "title": p.title, "book": p.book, "chapter": p.chapter,
                "difficulty": p.difficulty, "tags": p.tags, "env": p.env, "requires": req,
                "description": p.description, "explanation": p.explanation, "starter": p.starter,
                "solution": p.solution, "tests": p.tests, "cuda": p.cuda, "lang": p.lang, "sanitize": p.sanitize}
        text = json.dumps(data, ensure_ascii=False)
        digest.update(text.encode())
        (out / "data" / "p" / f"{p.slug}.json").write_text(text, encoding="utf-8")
        items.append({k: data[k] for k in ("slug", "number", "title", "book", "chapter", "difficulty", "tags", "env",
                                           "requires")} | {"hasCuda": bool(p.cuda)})
    index = {
        "version": digest.hexdigest()[:10],
        "books": [{"id": b, "name": n} for b, n in P.BOOKS],
        "chapters": {b: [{"path": c.path, "title": c.title, "part": c.part, "url": c.url, "order": c.order}
                         for c in chs.values() if (b, c.path) in used] for b, chs in chapters.items()},
        "problems": items,
        "localGuide": (HERE / "LOCAL.md").read_text(encoding="utf-8"),
        # Playground 的模板：playground/*.py，第一行的文档字符串是标题
        "playground": [{"id": f.stem, "title": f.read_text(encoding="utf-8").split('"""')[1].strip(),
                        "code": f.read_text(encoding="utf-8").split('"""', 2)[2].lstrip("\n")}
                       for f in sorted((HERE / "playground").glob("*.py"))],
    }
    for tpl in index["playground"]:
        digest.update(tpl["code"].encode())
    index["version"] = digest.hexdigest()[:10]
    (out / "data" / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    # 静态资源加版本号，避免浏览器缓存旧的 app.js
    html = (out / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'(src|href)="(app\.js|app\.css)"', lambda m: f'{m.group(1)}="{m.group(2)}?v={index["version"]}"', html)
    (out / "index.html").write_text(html, encoding="utf-8")
    counts = {b: sum(p.book == b for p in problems) for b, _ in P.BOOKS}
    print(f"practice: {len(problems)} 道题 {counts} -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site/practice"))
