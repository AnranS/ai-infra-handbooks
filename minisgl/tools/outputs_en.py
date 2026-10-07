"""英文版的"运行结果"：把 examples/ 里打印用的中文字面量换成英文，跑一遍，写进 docs/_outputs_en/。

英文页上展示的代码和输出因此始终对得上：输出就是这份译过的脚本跑出来的。
作数据用的中文（比如演示分词的例句）不翻译，留在 i18n-en-strings.json 之外。

    python tools/outputs_en.py              # 跑全部示例
    python tools/outputs_en.py ch07 ch09    # 只跑指定的
    python tools/outputs_en.py --missing    # 只列出还没译的字面量
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "hooks"))
import include_code as ic  # noqa: E402

OUT = BOOK / "docs" / "_outputs_en"
WORK = BOOK / ".i18n-en" / "examples"
PY = os.environ.get("MINISGL_PY", str(BOOK / ".venv-llm" / "bin" / "python"))
ENV = dict(os.environ, PYTHONPATH=f"{BOOK / 'python'}:{BOOK / 'tests'}",
           TOKENIZERS_PARALLELISM="false", LOG_LEVEL="WARNING")


def load_tables() -> None:
    for name, table in (("i18n-en-code.json", ic._CODE_TR), ("i18n-en-strings.json", ic._STR_TR)):
        f = BOOK / name
        table.clear()
        table.update(json.loads(f.read_text(encoding="utf-8")) if f.exists() else {})


def main(argv: list[str]) -> None:
    load_tables()
    names = [a for a in argv if not a.startswith("--")]
    files = sorted((BOOK / "examples").glob("*.py"))
    if names:
        files = [f for f in files if any(n in f.stem for n in names)]
    if "--missing" in argv:
        missing: set = set()
        for f in files:
            ic.translate_strings(f.read_text(encoding="utf-8"), missing)
        print(json.dumps({k: "" for k in sorted(missing)}, ensure_ascii=False, indent=1))
        print(f"还有 {len(missing)} 个字面量没有英文译文", file=sys.stderr)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in files:
        src = f.read_text(encoding="utf-8")
        en = ic.translate_strings(ic.translate_code(src, ".py"))
        tgt = WORK / f.name
        tgt.write_text(en, encoding="utf-8")
        t = time.time()
        r = subprocess.run([PY, str(tgt)], cwd=BOOK, env=ENV, capture_output=True,
                           text=True, timeout=1800)
        if r.returncode != 0:
            ok = False
            print(f"[FAIL] {f.name}\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
            continue
        (OUT / f"{f.stem}.txt").write_text(r.stdout, encoding="utf-8")
        print(f"[ok] {f.name} ({time.time() - t:.0f}s)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
