"""Materialize code blocks that carry a title="path" into a temp project and run pytest on it.

Usage: .venv-check/bin/python tools/check_pytest_blocks.py docs/engineering/testing.md
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check_examples import blocks  # noqa: E402

TITLE = re.compile(r'title="([^"]+)"')


def titled_blocks(md: Path):
    lines = md.read_text(encoding="utf-8").splitlines()
    for lang, lineno, code in blocks(md.read_text(encoding="utf-8")):
        m = TITLE.search(lines[lineno - 2])
        if m:
            yield m.group(1), code


def main(argv):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for md in map(Path, argv):
            for rel, code in titled_blocks(md):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(code, encoding="utf-8")
                print("wrote", rel)
        for pkg in {p.parent for p in root.glob("src/*/*.py")}:
            (pkg / "__init__.py").touch()
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rxs"], cwd=root, env=env)
        return proc.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
