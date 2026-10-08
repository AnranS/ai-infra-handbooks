"""《PyTorch 速成》的代码核对。

每一页是独立的：同一页里的脚本写进同一个目录、按出现顺序运行，可以互相 import；
页与页之间互不影响。规则（哪些块要跑、怎么跟 ```text title="输出"``` 比对）和分布式训练手册一样，
直接复用 train/tools/check_code.py，只把根目录和工作目录换成本书的。

用法：python torch/tools/check_code.py [torch/docs/xxx.md ...]    （不带参数时检查全部）
解释器：环境变量 PYTHON，默认用 cpp/.venv-py/bin/python（装有 CPU 版 torch）。
"""

import importlib.util
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
TRAIN_CHECK = BOOK.parent / "train" / "tools" / "check_code.py"


def load():
    spec = importlib.util.spec_from_file_location("train_check_code", TRAIN_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = BOOK
    mod.BUILD = BOOK / "build" / "examples"
    return mod


def main(argv: list[str]) -> int:
    mod = load()
    pages = [Path(p).resolve() for p in argv] or sorted((BOOK / "docs").rglob("*.md"))
    total, errs = 0, []
    for p in pages:
        n, e = mod.check_page(p)
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(pages)} 个页面，{total} 个脚本，{len(errs)} 个问题（{mod.PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
