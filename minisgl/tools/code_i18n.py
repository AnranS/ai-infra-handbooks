"""列出英文版还缺译文的代码注释 / docstring（只看正文真正引用到的片段）。

    python tools/code_i18n.py            # 打印缺失的片段（JSON，可直接填进 i18n-en-code.json）
    python tools/code_i18n.py --count    # 只报数量
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BOOK / "hooks"))
import include_code as ic  # noqa: E402

CODE = re.compile(r"^[ \t]*@@code (?P<path>[^:@\s]+)(?::(?P<sym>[\w.]+))?[^@]*@@\s*$", re.M)


def main(argv: list[str]) -> None:
    f = BOOK / "i18n-en-code.json"
    ic._CODE_TR.clear()
    ic._CODE_TR.update(json.loads(f.read_text(encoding="utf-8")) if f.exists() else {})
    missing: set = set()
    files: set = set()
    for md in sorted((BOOK / "docs").rglob("*.md")):
        files.update(m["path"] for m in CODE.finditer(md.read_text(encoding="utf-8")))
    for rel in sorted(files):                    # 整个文件翻一遍：正文引用的是文件里的片段
        path = BOOK / rel
        ic.translate_code(path.read_text(encoding="utf-8"), path.suffix, missing)
    if "--count" not in argv:
        print(json.dumps({k: "" for k in sorted(missing)}, ensure_ascii=False, indent=1))
    print(f"还有 {len(missing)} 段代码注释没有英文译文", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
