# 项目与工具链

<p class="lead">会写 Python 和会做 Python 项目是两回事。这一章讲一个现代 Python 项目的标准配置：用 uv 管理环境和依赖，用 <code>pyproject.toml</code> 集中配置，用 ruff 统一风格，用 mypy 检查类型，用 pre-commit 和 CI 自动把关，最后打包发布。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么每个项目都要有独立的虚拟环境？
    2. `pyproject.toml` 里 `dependencies` 和开发依赖有什么区别？锁文件是做什么的？
    3. 什么是 src 布局？它解决了什么问题？
    4. ruff 替代了哪些工具？
    5. 一个项目从 clone 下来到能跑测试，理想情况下需要几条命令？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 不同项目依赖的包和版本可能冲突；装在系统的 Python 里会互相污染，也没法复现别人的环境。每个项目一个虚拟环境，互不影响。
    2. `dependencies` 是运行这个包必需的依赖，用户安装时会装上；开发依赖（测试、lint、类型检查工具）只在开发时需要。锁文件记录解析出的每个包的精确版本（包括间接依赖），保证每台机器、每次安装都一样。
    3. 源码放在 `src/包名/` 下，而不是仓库的根目录。这样测试时导入的一定是安装好的包，而不是恰好在当前目录下的源码，能暴露"忘了打包某个文件"之类的问题。
    4. flake8（及其各种插件）、isort、pyupgrade，还有 black 的格式化（`ruff format`），而且快得多。
    5. 理想情况下两条：`git clone` 之后 `uv sync` 装好环境，再 `uv run pytest`。

## 为什么是 uv

Python 的包管理工具曾经非常分裂：`pip`、`venv`、`virtualenv`、`pip-tools`、`pipenv`、`poetry`、`pyenv`、`pipx`……[uv](https://docs.astral.sh/uv/) 用一个工具覆盖了它们的功能，而且因为是 Rust 写的，速度快一到两个数量级。它已经成为新项目的主流选择。

| 以前的做法 | 用 uv |
| --- | --- |
| `pyenv install 3.14` | `uv python install 3.14` |
| `python -m venv .venv` | `uv venv`（通常不需要手动执行，`uv sync` 会自动创建） |
| `pip install requests` + 手动写进 requirements.txt | `uv add requests` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip-compile` 生成锁定版本 | `uv lock`（自动维护 `uv.lock`） |
| `source .venv/bin/activate && pytest` | `uv run pytest` |
| `pipx install ruff` | `uv tool install ruff`，或者临时运行 `uvx ruff` |
| `python -m build` + `twine upload` | `uv build` + `uv publish` |

!!! note "为什么需要虚拟环境"
    不同项目可能依赖同一个包的不同版本；往系统 Python 里装包还可能破坏操作系统自带的工具（所以现在很多发行版会直接拒绝 `pip install`，报 `externally-managed-environment`）。**每个项目一个虚拟环境**，依赖互不干扰，删掉 `.venv` 目录就能干净地重来。

## 从零创建一个项目

```bash
uv init --package logtool      # --package 生成可安装的包（src 布局），适合库和命令行工具
cd logtool
uv add httpx rich              # 运行时依赖
uv add --dev pytest ruff mypy  # 开发依赖：只在开发和 CI 里需要
uv run logtool                 # 运行 [project.scripts] 里定义的命令
```

生成的结构（加上后面要添加的测试目录）：

```text
logtool/
├── .python-version        # 项目使用的 Python 版本，uv 会自动安装
├── pyproject.toml         # 项目元数据、依赖、所有工具的配置
├── uv.lock                # 锁文件：所有依赖（包括间接依赖）的精确版本，要提交到 git
├── README.md
├── src/
│   └── logtool/
│       ├── __init__.py
│       ├── cli.py
│       └── parser.py
└── tests/
    ├── conftest.py
    └── test_parser.py
```

别人 clone 下来之后，只需要：

```bash
uv sync          # 按 uv.lock 创建一模一样的环境
uv run pytest    # 跑测试
```

### src 布局

把包放在 `src/` 目录下，而不是项目根目录。这样在项目根目录运行测试时，`import logtool` **导入的是安装到虚拟环境里的包**，而不是恰好在当前目录下的源码文件夹。它能提前暴露"忘记把某个文件打包进去"、"依赖当前工作目录"这类只会在用户那里出现的问题。uv 的 `--package` 模式默认就是 src 布局。

### 单文件脚本也能声明依赖

写一个一次性的小脚本，不值得建一个项目？用 PEP 723 的内联元数据，把依赖写在脚本开头：

```py
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "rich"]
# ///
import httpx
from rich import print

print(httpx.get("https://example.com").status_code)
```

`uv run fetch.py` 会自动创建一个临时环境、装好依赖再运行。`uv add --script fetch.py httpx` 可以帮你维护这个头部。

## `pyproject.toml`：一个文件管所有配置

```toml
[project]
name = "logtool"
version = "0.1.0"
description = "Analyze log files"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",
    "rich>=14",
]

[project.scripts]
logtool = "logtool.cli:main"          # 安装后得到 logtool 命令，调用 logtool/cli.py 的 main()

[dependency-groups]
dev = ["pytest>=9", "ruff>=0.16", "mypy>=2"]

[build-system]
requires = ["uv_build>=0.12,<0.13"]
build-backend = "uv_build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
```

- `dependencies` 写**宽松的下限**（`>=`），给使用者留出空间；精确版本由 `uv.lock` 锁定。
- 应用程序（部署的服务）和库的区别：应用靠锁文件保证每次部署一致；库不能锁定依赖的版本，否则会和使用者的其他依赖冲突。
- 所有工具的配置都集中在 `[tool.xxx]` 下，不再需要 `setup.cfg`、`.flake8`、`pytest.ini`、`mypy.ini` 一堆文件。

## 代码风格：ruff

[ruff](https://docs.astral.sh/ruff/) 是 Rust 写的 linter 和 formatter，一个工具替代了 flake8、isort、pyupgrade、black 以及几十个插件，速度快到可以在每次保存文件时运行。

```bash
uv run ruff check .          # 检查问题
uv run ruff check --fix .    # 自动修复能修的问题
uv run ruff format .         # 格式化（风格与 black 兼容）
```

上面配置里选择的规则集：

| 代码 | 来源 | 检查什么 |
| --- | --- | --- |
| `E`、`W` | pycodestyle | PEP 8 风格问题 |
| `F` | pyflakes | 未使用的导入和变量、未定义的名字 |
| `I` | isort | import 排序 |
| `B` | flake8-bugbear | 常见 bug：可变默认参数、`except:` 等 |
| `UP` | pyupgrade | 可以用更新语法改写的旧写法 |
| `SIM` | flake8-simplify | 可以简化的代码 |
| `RUF` | ruff 自有规则 | 各种杂项 |

!!! tip "关于 PEP 8"
    风格问题交给工具，不要在代码审查里争论空格和换行。但有几条 PEP 8 规则是工具管不了、需要自己遵守的：**命名**（函数和变量 `snake_case`，类 `PascalCase`，常量 `UPPER_CASE`）、**写有意义的名字**、以及"可读性很重要"。

## 类型检查：mypy

```bash
uv run mypy src/
```

新项目直接开 `strict = true`。老项目可以先用默认配置，再逐个模块收紧。第三方库没有类型信息时，找找有没有 `types-xxx` 存根包，或者在配置里对那个模块设置 `ignore_missing_imports`。详见[类型标注](../types/typing.md#类型检查器)。

## pre-commit：提交前自动检查

[pre-commit](https://pre-commit.com/) 在 `git commit` 时自动运行检查，不合格就拒绝提交，问题不会进入代码仓库。在项目根目录创建 `.pre-commit-config.yaml`：

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.8
    hooks:
      - id: ruff-check
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

然后：

```bash
uv tool install pre-commit
pre-commit install              # 装上 git 钩子
pre-commit run --all-files      # 手动对所有文件跑一次
pre-commit autoupdate           # 把各个 rev 升级到最新
```

## 持续集成（CI）

![图：从提交到发布的流水线](../assets/figures/ci-pipeline.svg){.aig-svg}

每次推送和合并请求都自动跑一遍检查和测试。GitHub Actions 的示例：

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v7
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --locked
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src/
      - run: uv run pytest
```

`--locked` 要求 `uv.lock` 和 `pyproject.toml` 一致，防止有人改了依赖却忘了更新锁文件。

## 配置与密钥

- 配置（数据库地址、API 密钥、功能开关）**通过环境变量传入**，不要写死在代码里，更不要提交到 git。
- 本地开发可以用 `.env` 文件，把它加进 `.gitignore`，同时提交一个不含真实值的 `.env.example`。
- 读取和校验配置：简单场景用 `os.environ`；复杂场景可以用 `pydantic-settings`，它能从环境变量构造出带类型校验的配置对象。
- 程序启动时**一次性**读取并校验所有配置，缺了什么立刻报错退出，不要等到运行半天后才发现。

## 打包与发布

```bash
uv version --bump minor        # 0.1.0 -> 0.2.0
uv build                       # 在 dist/ 下生成 .tar.gz（源码包）和 .whl（wheel）
uv publish                     # 上传到 PyPI（需要令牌，推荐在 CI 里用可信发布）
```

版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)：`主版本.次版本.修订号`。不兼容的改动升主版本，新增功能升次版本，修 bug 升修订号。公司内部的包通常发布到私有的包仓库，`uv publish --publish-url` 指定地址即可。

## 一个项目的完成度清单

- [ ] `pyproject.toml` 声明了依赖和 Python 版本，`uv.lock` 已提交
- [ ] `uv sync && uv run pytest` 在一台新机器上能直接跑通
- [ ] ruff 和 mypy 没有报错，并配置了 pre-commit
- [ ] CI 在每次提交时运行检查和测试
- [ ] 核心逻辑有测试（见下一章）
- [ ] 用 `logging` 记录日志，不在库代码里 `print`
- [ ] 配置和密钥来自环境变量，没有提交到仓库
- [ ] README 写清楚了项目是做什么的、怎么安装、怎么运行、怎么开发

!!! interview "面试怎么答"
    工程化题：每个项目一个独立的虚拟环境，依赖写在 `pyproject.toml`，锁文件（`uv.lock`）固定整个依赖树并提交，保证 CI 和线上可复现；开发依赖（测试、lint）和运行依赖分开；库用 src 布局，避免测试时意外导入工作目录里未安装的代码。工具链：uv 管 Python 版本、环境和依赖，ruff 负责格式化和 lint（替代 black、isort、flake8 等），mypy / pyright 做类型检查，pre-commit 和 CI 自动执行；配置和密钥走环境变量、启动时一次性校验。理想情况是 clone 之后一两条命令就能跑测试。

## 练习

**1. 搭建一个完整的项目骨架。** 用 uv 创建一个名为 `wordstat` 的命令行工具项目：`wordstat FILE --top N` 输出文件中出现最多的 N 个单词。要求：

- src 布局，`[project.scripts]` 里注册 `wordstat` 命令；
- 逻辑写在 `wordstat/core.py`（纯函数，方便测试），命令行解析写在 `wordstat/cli.py`；
- 配好 ruff、mypy（strict）、pytest，写至少 3 个测试；
- 配好 pre-commit 和 GitHub Actions；
- 最后执行 `uv build`，然后在另一个目录里 `uv tool install dist/wordstat-*.whl`，确认 `wordstat` 命令可用。

??? success "参考思路"
    ```bash
    uv init --package wordstat && cd wordstat
    uv add --dev pytest ruff mypy
    mkdir tests
    ```

    `src/wordstat/core.py`：

    ```python
    import re
    from collections import Counter

    WORD = re.compile(r"[a-z']+")

    def top_words(text: str, n: int) -> list[tuple[str, int]]:
        return Counter(WORD.findall(text.lower())).most_common(n)

    assert top_words("a b a", 1) == [("a", 2)]
    ```

    `src/wordstat/cli.py`：

    ```py
    import argparse
    from pathlib import Path

    from wordstat.core import top_words

    def main(argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(prog="wordstat")
        parser.add_argument("file", type=Path)
        parser.add_argument("--top", type=int, default=10)
        args = parser.parse_args(argv)
        text = args.file.read_text(encoding="utf-8")
        for word, count in top_words(text, args.top):
            print(f"{count:>6}  {word}")
    ```

    `pyproject.toml` 里把入口改成 `wordstat = "wordstat.cli:main"`，其余配置参考正文。测试写法见下一章 [pytest](testing.md)。

## 小结

- [x] 用 uv 管理 Python 版本、虚拟环境、依赖和锁文件；`uv sync` + `uv run` 是日常操作。
- [x] 所有元数据和工具配置集中在 `pyproject.toml`；锁文件要提交。
- [x] 库和命令行工具使用 src 布局。
- [x] ruff 负责风格和 lint，mypy 负责类型，pre-commit 和 CI 自动执行。
- [x] 配置和密钥走环境变量，程序启动时一次性校验。
