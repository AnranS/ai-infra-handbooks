# Projects and the toolchain

<p class="lead">Writing Python and running a Python project are two different things. This chapter covers a modern Python project's standard setup: uv for the environment and the dependencies, <code>pyproject.toml</code> for all the configuration, ruff for a consistent style, mypy for types, pre-commit and CI to enforce them automatically, and finally packaging and publishing.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does every project need its own virtual environment?
    2. What is the difference between `dependencies` and the development dependencies in `pyproject.toml`? What is a lock file for?
    3. What is the src layout? What problem does it solve?
    4. Which tools does ruff replace?
    5. Ideally, how many commands does it take from cloning a project to running its tests?

??? success "Answers (try it yourself first, then expand)"
    1. Different projects' packages and versions can conflict; installing into the system Python pollutes them all and makes somebody else's environment impossible to reproduce. One virtual environment per project keeps them apart.
    2. `dependencies` are what running the package requires and are installed for the user; the development dependencies (testing, linting, type checking) are only needed while developing. A lock file records the exact resolved version of every package (including the indirect ones), so every machine and every installation is the same.
    3. The source goes under `src/<package>/` rather than at the repository's root. Tests then import the installed package rather than source that happens to be in the current directory, which exposes problems like "a file was left out of the package".
    4. flake8 (and its many plugins), isort, pyupgrade, and black's formatting (`ruff format`), and it is far faster.
    5. Ideally two: `uv sync` after the `git clone` sets the environment up, then `uv run pytest`.

## Why uv {#为什么是-uv}

Python's packaging tools used to be badly fragmented: `pip`, `venv`, `virtualenv`, `pip-tools`, `pipenv`, `poetry`, `pyenv`, `pipx` and more. [uv](https://docs.astral.sh/uv/) covers all of them in one tool, and being written in Rust it is one to two orders of magnitude faster. It has become the mainstream choice for new projects.

| The old way | With uv |
| --- | --- |
| `pyenv install 3.14` | `uv python install 3.14` |
| `python -m venv .venv` | `uv venv` (rarely needed by hand, since `uv sync` creates it) |
| `pip install requests` + editing requirements.txt by hand | `uv add requests` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip-compile` to pin versions | `uv lock` (maintaining `uv.lock` automatically) |
| `source .venv/bin/activate && pytest` | `uv run pytest` |
| `pipx install ruff` | `uv tool install ruff`, or `uvx ruff` for a one-off |
| `python -m build` + `twine upload` | `uv build` + `uv publish` |

!!! note "Why a virtual environment is needed"
    Different projects may need different versions of the same package; and installing into the system Python can break the tools the operating system ships (which is why many distributions now refuse `pip install` outright with `externally-managed-environment`). **One virtual environment per project** keeps the dependencies apart, and deleting the `.venv` directory starts cleanly over.

## Creating a project from scratch {#从零创建一个项目}

```bash
uv init --package logtool      # --package produces an installable package (the src layout), which suits libraries and command-line tools
cd logtool
uv add httpx rich              # the runtime dependencies
uv add --dev pytest ruff mypy  # the development dependencies: needed only while developing and in CI
uv run logtool                 # run the command defined under [project.scripts]
```

The structure it generates (plus the test directory added later):

<!-- i18n:diagram 114289d73f -->
```text
logtool/
├── .python-version        # the Python version this project uses, installed by uv
├── pyproject.toml         # the project's metadata, dependencies and every tool's configuration
├── uv.lock                # the lock file: the exact version of every dependency, committed to git
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

After somebody else clones it, all they need is:

```bash
uv sync          # build an identical environment from uv.lock
uv run pytest    # run the tests
```

### The src layout {#src-布局}

The package goes under `src/` rather than at the project's root. Running the tests from the project root then makes `import logtool` **import the package installed into the virtual environment** rather than the source folder that happens to be in the current directory. That surfaces problems like "a file was left out of the package" and "it depends on the working directory" early, instead of only at the user's end. uv's `--package` mode uses the src layout by default.

### A single-file script can declare dependencies too {#单文件脚本也能声明依赖}

Writing a one-off script that is not worth a project? PEP 723's inline metadata puts the dependencies at the top of the script:

```py
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "rich"]
# ///
import httpx
from rich import print

print(httpx.get("https://example.com").status_code)
```

`uv run fetch.py` creates a temporary environment, installs the dependencies and runs it. `uv add --script fetch.py httpx` maintains that header for you.

## `pyproject.toml`: one file for every setting {#pyprojecttoml一个文件管所有配置}

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
logtool = "logtool.cli:main"          # installing gives a logtool command calling main() in logtool/cli.py

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

- `dependencies` carry **a loose lower bound** (`>=`), leaving the user room; the exact versions are pinned by `uv.lock`.
- The difference between an application (a deployed service) and a library: an application relies on the lock file to make every deployment identical, while a library must not pin its dependencies' versions or they collide with the user's.
- Every tool's configuration lives under `[tool.xxx]`, with no more `setup.cfg`, `.flake8`, `pytest.ini` and `mypy.ini` scattered about.

## Code style: ruff {#代码风格ruff}

[ruff](https://docs.astral.sh/ruff/) is a linter and formatter written in Rust, one tool replacing flake8, isort, pyupgrade, black and dozens of plugins, and fast enough to run on every save.

```bash
uv run ruff check .          # check for problems
uv run ruff check --fix .    # fix automatically what can be fixed
uv run ruff format .         # format (compatible with black's style)
```

The rule sets chosen in that configuration:

| Code | From | What it checks |
| --- | --- | --- |
| `E`, `W` | pycodestyle | PEP 8 style issues |
| `F` | pyflakes | unused imports and variables, undefined names |
| `I` | isort | the order of the imports |
| `B` | flake8-bugbear | common bugs: a mutable default argument, a bare `except:` and so on |
| `UP` | pyupgrade | old spellings that newer syntax can replace |
| `SIM` | flake8-simplify | code that can be simplified |
| `RUF` | ruff's own rules | assorted others |

!!! tip "About PEP 8"
    Leave style to the tool and do not argue about spaces and line breaks in code review. But a few PEP 8 rules are beyond a tool and are yours to follow: **naming** (`snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants), **choosing meaningful names**, and "readability counts".

## Type checking: mypy {#类型检查mypy}

```bash
uv run mypy src/
```

A new project turns `strict = true` straight on. An old one can start with the defaults and tighten module by module. When a third-party library has no type information, look for a `types-xxx` stubs package, or set `ignore_missing_imports` for that module in the configuration. See [type annotations](../types/typing.md#类型检查器).

## pre-commit: checks before every commit {#pre-commit提交前自动检查}

[pre-commit](https://pre-commit.com/) runs the checks at `git commit` time and refuses the commit when they fail, so problems never reach the repository. Create `.pre-commit-config.yaml` at the project's root:

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

Then:

```bash
uv tool install pre-commit
pre-commit install              # install the git hooks
pre-commit run --all-files      # run it over every file once by hand
pre-commit autoupdate           # upgrade every rev to the latest
```

## Continuous integration (CI) {#持续集成ci}

![Figure: the pipeline from a commit to a release](../assets/figures/ci-pipeline.svg){.aig-svg}

Every push and pull request runs the checks and the tests automatically. A GitHub Actions example:

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

`--locked` requires `uv.lock` to agree with `pyproject.toml`, so nobody can change a dependency and forget to update the lock file.

## Configuration and secrets {#配置与密钥}

- Configuration (the database address, API keys, feature switches) comes in **through environment variables**, never hard-coded and certainly never committed to git.
- Local development can use a `.env` file, added to `.gitignore`, with a `.env.example` holding no real values committed alongside.
- Reading and validating it: `os.environ` for simple cases; `pydantic-settings` for complicated ones, which builds a type-validated configuration object from the environment.
- Read and validate **all** the configuration once at startup and exit immediately on anything missing, rather than finding out hours into a run.

## Packaging and publishing {#打包与发布}

```bash
uv version --bump minor        # 0.1.0 -> 0.2.0
uv build                       # produces a .tar.gz (the source distribution) and a .whl (the wheel) under dist/
uv publish                     # upload to PyPI (a token is needed; trusted publishing from CI is recommended)
```

Version numbers follow [semantic versioning](https://semver.org/): `major.minor.patch`. An incompatible change raises the major, a new feature the minor, and a bug fix the patch. A company's internal packages usually go to a private index, which `uv publish --publish-url` points at.

## A project's completeness checklist {#一个项目的完成度清单}

- [ ] `pyproject.toml` declares the dependencies and the Python version, and `uv.lock` is committed
- [ ] `uv sync && uv run pytest` works on a fresh machine
- [ ] ruff and mypy report nothing, and pre-commit is configured
- [ ] CI runs the checks and the tests on every commit
- [ ] the core logic has tests (the next chapter)
- [ ] logging goes through `logging`, with no `print` in library code
- [ ] configuration and secrets come from the environment and are not committed
- [ ] the README says what the project is, how to install it, how to run it and how to develop it

!!! interview "Answering in an interview"
    On engineering: one virtual environment per project, dependencies declared in `pyproject.toml`, and a lock file (`uv.lock`) pinning the whole dependency tree and committed, so CI and production are reproducible; development dependencies (testing, linting) kept apart from the runtime ones; a library uses the src layout, so tests cannot import uninstalled code from the working directory by accident. The toolchain: uv for the Python version, the environment and the dependencies, ruff for formatting and linting (replacing black, isort, flake8 and more), mypy / pyright for types, and pre-commit and CI to run them; configuration and secrets come from the environment and are validated once at startup. Ideally one or two commands after cloning run the tests.

## Exercises {#练习}

**1. Build a complete project skeleton.** Create a command-line tool project called `wordstat` with uv: `wordstat FILE --top N` prints the N most frequent words in the file. Requirements:

- the src layout, with the `wordstat` command registered under `[project.scripts]`;
- the logic in `wordstat/core.py` (pure functions, easy to test) and the command-line parsing in `wordstat/cli.py`;
- ruff, mypy (strict) and pytest configured, with at least 3 tests;
- pre-commit and GitHub Actions configured;
- finally `uv build`, then `uv tool install dist/wordstat-*.whl` in another directory, confirming the `wordstat` command works.

??? success "The approach"
    ```bash
    uv init --package wordstat && cd wordstat
    uv add --dev pytest ruff mypy
    mkdir tests
    ```

    `src/wordstat/core.py`:

    ```python
    import re
    from collections import Counter

    WORD = re.compile(r"[a-z']+")

    def top_words(text: str, n: int) -> list[tuple[str, int]]:
        return Counter(WORD.findall(text.lower())).most_common(n)

    assert top_words("a b a", 1) == [("a", 2)]
    ```

    `src/wordstat/cli.py`:

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

    Change the entry point in `pyproject.toml` to `wordstat = "wordstat.cli:main"` and take the rest of the configuration from the body of this chapter. How to write the tests is in the next chapter, [pytest](testing.md).

## Summary {#小结}

- [x] Use uv for the Python version, the virtual environment, the dependencies and the lock file; `uv sync` + `uv run` are the daily commands.
- [x] All the metadata and tool configuration live in `pyproject.toml`; the lock file is committed.
- [x] A library or a command-line tool uses the src layout.
- [x] ruff handles style and linting, mypy handles types, and pre-commit and CI run them automatically.
- [x] Configuration and secrets come from the environment and are validated once at startup.
