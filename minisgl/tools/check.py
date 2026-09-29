"""手写 mini-sglang 的全部验证，一条命令跑完：

1. 从正文里收集 @@upstream ...@@ 引用，在本地的官方仓库副本里定位行号，生成 docs/_outputs/upstream_index.json；
2. 运行 examples/*.py，把输出写进 docs/_outputs/<名字>.txt（正文里的"运行结果"就来自这里）；
3. 用两个版本的 nvcc 编译 CUDA kernel（自检程序和 PyTorch 扩展），并在 CPU 模拟器上运行自检；
4. 运行 pytest 测试套件（每章一个测试文件）。

用法：
    python tools/check.py                  全部
    python tools/check.py examples ch07_llm  只跑指定的示例
    python tools/check.py --skip-tests     跳过 pytest
需要：models/ 下的 Qwen3-0.6B、官方仓库副本（UPSTREAM 环境变量或 ~/src-reading/mini-sglang）、
CUDA 手册的 nvcc 工具链（~/cuda-handbook/.toolkit*）。
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "docs" / "_outputs"
UPSTREAM = Path(os.environ.get("UPSTREAM", Path.home() / "src-reading" / "mini-sglang"))
UPSTREAM_COMMIT = "9a91cfafe754aa85daee49998176275667eb58f2"
TOOLKITS = [Path.home() / "cuda-handbook" / d / "bin" / "nvcc" for d in (".toolkit12", ".toolkit")]
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{BOOK / 'python'}:{BOOK / 'tests'}", TOKENIZERS_PARALLELISM="false",
           LOG_LEVEL="WARNING")


def build_upstream_index() -> None:
    refs = set()
    for md in (BOOK / "docs").rglob("*.md"):
        refs.update(re.findall(r"@@upstream ([^@\s]+)@@", md.read_text(encoding="utf-8")))
    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert head == UPSTREAM_COMMIT, f"upstream checkout is at {head}, expected {UPSTREAM_COMMIT}"
    symbols = {}
    for ref in sorted(refs):
        path, _, sym = ref.partition(":")
        source = (UPSTREAM / "python" / "minisgl" / path).read_text(encoding="utf-8")
        if not sym:
            symbols[ref] = {}
            continue
        body, node = ast.parse(source).body, None
        for part in sym.split("."):
            node = next(n for n in body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                        and n.name == part)
            body = node.body
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        symbols[ref] = {"lines": [start, node.end_lineno]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "upstream_index.json").write_text(
        json.dumps({"commit": UPSTREAM_COMMIT, "symbols": symbols}, indent=1, ensure_ascii=False))
    print(f"[ok] upstream index: {len(symbols)} references")


def run_examples(names: list[str]) -> bool:
    files = sorted((BOOK / "examples").glob("*.py"))
    if names:
        files = [f for f in files if f.stem in names]
    ok = True
    for f in files:
        t = time.time()
        r = subprocess.run([PY, str(f)], cwd=BOOK, env=ENV, capture_output=True, text=True,
                           timeout=1800)
        text = r.stdout
        if r.returncode != 0:
            ok = False
            print(f"[FAIL] {f.name}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")
            continue
        (OUT / f"{f.stem}.txt").write_text(text, encoding="utf-8")
        print(f"[ok] {f.name} ({time.time() - t:.0f}s)")
    return ok


def check_cuda() -> bool:
    import torch.utils.cpp_extension as ext
    import sysconfig

    ok = True
    csrc = BOOK / "python" / "minisgl" / "kernel" / "csrc"
    incs = [f"-I{p}" for p in ext.include_paths()] + [f"-I{sysconfig.get_paths()['include']}", f"-I{csrc}"]
    for nvcc in TOOLKITS:
        version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout
        tag = re.search(r"release ([\d.]+)", version).group(1)
        for src, flags in [(BOOK / "tests" / "cuda" / "test_kv_kernels.cu", ["-std=c++17", f"-I{csrc}"]),
                           (csrc / "ext.cu", ["-std=c++20", "-DTORCH_EXTENSION_NAME=minisgl_kernels", *incs])]:
            r = subprocess.run([str(nvcc), "-O2", "-Wno-deprecated-gpu-targets", *flags, "-c", str(src),
                                "-o", "/dev/null"], capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else "FAIL"
            ok &= r.returncode == 0
            print(f"[{status}] nvcc {tag} {src.name}" + ("" if r.returncode == 0 else "\n" + r.stderr[-2000:]))
    r = subprocess.run([PY, str(BOOK / "tools" / "emu_kernels.py")], capture_output=True, text=True)
    print(("[ok] " if r.returncode == 0 else "[FAIL] ") + "CPU emulator:\n    " + r.stdout.strip().replace("\n", "\n    "))
    return ok and r.returncode == 0


def run_tests() -> bool:
    r = subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=BOOK, env=ENV)
    return r.returncode == 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "examples":
        return 0 if run_examples(argv[1:]) else 1
    build_upstream_index()
    ok = run_examples([])
    ok &= check_cuda()
    if "--skip-tests" not in argv:
        ok &= run_tests()
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
