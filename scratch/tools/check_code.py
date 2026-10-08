"""《从零训练一个小模型》的代码核对。

四章的脚本是接力的：前一章生成的语料、分词器和 checkpoint 留给后一章用，所以不管检查哪一页，
整本书都按 ORDER 的顺序从头跑一遍，共用一个工作目录 build/examples/tutorial/。
规则（哪些块要跑、怎么跟 ```text title="输出"``` 比对）和分布式训练手册完全一样，
直接复用 train/tools/check_code.py，只把工作目录换成本书的。

用法：python scratch/tools/check_code.py [scratch/docs/xxx.md ...]    （不带参数时检查全部）
解释器：环境变量 PYTHON，默认用 cpp/.venv-py/bin/python（装有 CPU 版 torch）。
"""

import importlib.util
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
TRAIN_CHECK = BOOK.parent / "train" / "tools" / "check_code.py"
ORDER = ["data.md", "model.md", "scale.md", "one-gpu.md"]   # 接力顺序；index.md 里没有要跑的脚本


def load():
    spec = importlib.util.spec_from_file_location("train_check_code", TRAIN_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = BOOK
    mod.BUILD = BOOK / "build" / "examples"
    mod.work_dir = lambda md: mod.BUILD / "tutorial"         # 四章共用一个目录
    return mod


def main(argv: list[str]) -> int:
    mod = load()
    wanted = {Path(p).resolve() for p in argv}
    pages = [BOOK / "docs" / name for name in ORDER]
    if wanted:                                               # 只指定了某一页时，仍从头跑到那一页为止
        last = max((i for i, p in enumerate(pages) if p in wanted), default=len(pages) - 1)
        pages = pages[: last + 1]
    total, errs = 0, []
    for i, p in enumerate(pages):
        n, e = mod.check_page(p, fresh=(i == 0))
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(pages)} 个页面，{total} 个脚本，{len(errs)} 个问题（{mod.PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
