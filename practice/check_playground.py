"""用和浏览器里相同的运行器（runtime/playground.py）把 Playground 的模板都跑一遍，确认没有报错。

用法：python practice/check_playground.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime"))

import playground  # noqa: E402
import tritonkit  # noqa: E402


def main() -> int:
    bad = 0
    for f in sorted((HERE / "playground").glob("*.py")):
        code = f.read_text(encoding="utf-8")
        if re.search(r"^\s*(import|from)\s+matplotlib", code, re.M) and importlib.util.find_spec("matplotlib") is None:
            print(f"-  {f.name}：跳过（本机没有 matplotlib）")
            continue
        if re.search(r"^\s*(import|from)\s+triton", code, re.M):
            tritonkit.install(prefer_real=False)
        r = asyncio.run(playground.run(code))
        ok = r["error"] is None
        bad += not ok
        print(f"{'✓' if ok else '✗'}  {f.name}（{r['ms']:.0f} ms）" + ("" if ok else "\n" + r["error"]))
        for line in r["stdout"].strip().splitlines()[:3]:
            print("     " + line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
