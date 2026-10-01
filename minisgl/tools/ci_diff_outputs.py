"""CI 用：比较 docs/_outputs/*.txt 和 git 里提交的版本，数字允许 5% 的相对误差（百分数 1 个百分点）（不同机器的浮点归约顺序不同），
文字必须完全一致。带耗时的文件（启动时间、吞吐、延迟）和需要 nvcc 的文件不比。

用法：python tools/ci_diff_outputs.py        （在仓库任意位置运行）
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
SKIP = {"ch00_quickstart.txt", "ch03_models.txt", "ch12_message.txt", "ch15_server.txt", "ch21_benchmark.txt",   # 带耗时
        "ch19_kernels.txt"}                                                                                      # 要 nvcc
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def same(want: list[str], got: list[str]) -> bool:
    if want == got:
        return True
    if len(want) != len(got):
        return False
    for w, g in zip(want, got):
        if NUM.sub("#", w) != NUM.sub("#", g):
            return False
        for ma, mb in zip(NUM.finditer(w), NUM.finditer(g)):
            x, y = float(ma.group()), float(mb.group())
            pct = w[ma.end():ma.end() + 1] == "%"              # 百分数（常常是两个数的差）按 1 个百分点算
            if abs(x - y) > (1.0 if pct else max(0.05 * max(abs(x), abs(y)), 0.011)):
                return False
    return True


def main() -> int:
    bad = []
    for f in sorted((BOOK / "docs" / "_outputs").glob("*.txt")):
        if f.name in SKIP:
            continue
        rel = f.relative_to(ROOT).as_posix()
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if head.returncode:
            continue                                            # 新文件
        want = head.stdout.rstrip("\n").splitlines()
        got = f.read_text(encoding="utf-8").rstrip("\n").splitlines()
        if not same(want, got):
            bad.append(rel)
            print(f"✗ {rel} 和提交的版本不一致（数字差超过 5%，或文字有变）")
            subprocess.run(["git", "--no-pager", "diff", "--", rel], cwd=ROOT)
    print(f"{'有 ' + str(len(bad)) + ' 个输出漂移' if bad else '示例输出和正文一致（数字容差 5%）'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
