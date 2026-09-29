# C++ 进阶手册

面向 AI Infra 的现代 C++：值语义与 RAII、移动语义、所有权、模板与编译期派发、内存布局与内存池、线程与内存序、无锁队列与线程池、CMake 与 sanitizer、pybind11 与 PyTorch C++ 扩展，以及如何阅读推理基础库的 C++ 代码（以 vLLM 0.30.0 的 `csrc/` 为例）。

## 目录

- `docs/`：正文，按部分分为 `basics/`、`memory/`、`concurrency/`、`engineering/`
- `tools/check_code.py`：编译并运行正文里所有标了文件名的程序，确认页面上的输出与实际运行一致

## 校验

```bash
# 需要 g++ 12 以上（C++20）与 cmake；pybind11 一章还需要一个带 torch（CPU 版即可）、pybind11、ninja 的 Python 环境
uv venv .venv-py && uv pip install --python .venv-py/bin/python pybind11 ninja numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
python3 tools/check_code.py                         # 所有页面
python3 tools/check_code.py docs/concurrency/*.md   # 指定页面
```

约定（详见 `tools/check_code.py` 的说明）：

- ```` ```cpp title="x.cpp" ```` 是完整程序，默认在 ASan + UBSan 下编译运行（`-Wall -Wextra -Werror`），紧跟的 ```` ```text title="输出" ```` 必须与实际输出逐行一致；
- `sanitize="thread"` 改用 TSan，`sanitize="none" flags="-O2"` 用于测性能的例子（结果不做比对）；
- `expect="fail"` 是故意演示错误的程序（必须被 sanitizer 抓到），`expect="compile-error"` 必须编译失败；
- `project="名字"` 的代码块组成一个多文件工程，`run="yes"` 的 bash 脚本在工程目录里执行。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f cpp/mkdocs.yml`。
