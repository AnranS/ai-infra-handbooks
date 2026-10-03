"""数学基础手册的代码核对。

这本书的例子用的是大模型原理手册的环境：自己实现的 mini_llm（由 llm/docs 里的模块文件生成）、models/Qwen3-0.6B、
冻结的样本文本 docs/assets/sample-passage.txt，路径都相对 llm/ 目录。所以这里不另写一套规则，直接调用
llm/tools/check_code.py：它在 llm/ 目录下运行页面里的代码块，输出与紧跟的 ```text title="输出"``` 逐行比对。

用法：
    llm/.venv-llm/bin/python math/tools/check_code.py                       # 全部章节
    llm/.venv-llm/bin/python math/tools/check_code.py math/docs/calculus.md # 指定页面
"""

import importlib.util
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
LLM_CHECK = BOOK.parent / "llm" / "tools" / "check_code.py"


def main(argv: list[str]) -> int:
    pages = [str(Path(p).resolve()) for p in argv] or [str(p) for p in sorted((BOOK / "docs").glob("*.md"))]
    spec = importlib.util.spec_from_file_location("llm_check_code", LLM_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(pages)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
