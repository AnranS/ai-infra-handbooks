# Python 进阶手册

A Chinese-language MkDocs Material site that takes people who already know basic Python to confident, idiomatic use.

## Layout

- `docs/`: the pages (Markdown), grouped by part: `core/`, `types/`, `engineering/`, `concurrency/`, `practice/`
- `tools/check_examples.py`: runs every ` ```python ` block and doctests every ` ```pycon ` block (` ```py ` blocks are illustrative and skipped)
- `tools/check_pytest_blocks.py`: writes blocks that carry `title="path"` into a temp project and runs pytest on them (used by the testing chapter)

## Commands

```bash
# build dependencies (MkDocs 1.x + Material)
uv venv .venv && uv pip install --python .venv/bin/python -r requirements-docs.txt
# checker environment (Python 3.14 + pytest + mypy)
uv venv --python 3.14 .venv-check && uv pip install --python .venv-check/bin/python pytest mypy

.venv-check/bin/python tools/check_examples.py              # all pages
.venv-check/bin/python tools/check_pytest_blocks.py docs/engineering/testing.md
.venv/bin/mkdocs build --strict                              # output in site/
.venv/bin/mkdocs serve -a 0.0.0.0:8000                       # live preview while editing
```

To build all three handbooks into one site, run `./build.sh` at the repository root.
