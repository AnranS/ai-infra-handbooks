"""运行分布式训练手册里的 PyTorch 示例，确认页面上的输出与实际运行结果一致（CPU 版 PyTorch 即可，多进程用 gloo 后端）。

约定：
  ```python title="x.py"         完整脚本：运行它，紧跟的 ```text title="输出"``` 必须与标准输出逐行一致
  ```python title="x.py" torchrun="4"   多进程脚本：用 torchrun --standalone --nproc-per-node 4 启动（只比对标准输出，通常只有 rank 0 打印）
  ```python title="x.py" run="no"   需要 GPU 的脚本：只做语法检查（py_compile），页面上的输出不做比对
  没有 title 的代码块是片段，不检查。同一页的脚本写在同一个目录里，可以互相 import。

用法：python tools/check_code.py [docs/xxx/yyy.md ...]    （不带参数时检查所有页面）
解释器：环境变量 PYTHON，默认用 ../cpp/.venv-py/bin/python（装有 CPU 版 torch）。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "examples"
PYTHON = os.environ.get("PYTHON") or str(ROOT.parent / "cpp" / ".venv-py" / "bin" / "python")
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')


def blocks(md: Path):
    lines = md.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, fence = m["indent"], m["fence"]
        j = i + 1
        while j < len(lines) and lines[j].strip() != fence:
            j += 1
        body = "\n".join(l[len(indent):] if l.startswith(indent) else l.lstrip() for l in lines[i + 1:j])
        yield {"lang": m["lang"], "attrs": dict(ATTR.findall(m["rest"])), "body": body + "\n", "line": i + 1}
        i = j + 1


def check_page(md: Path) -> tuple[int, list[str]]:
    items = list(blocks(md))
    work = BUILD / md.relative_to(ROOT / "docs").with_suffix("")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scripts = [(k, b) for k, b in enumerate(items) if b["lang"] == "python" and b["attrs"].get("title", "").endswith(".py")]
    for _, b in scripts:
        (work / b["attrs"]["title"]).write_text(b["body"], encoding="utf-8")
    errors = []
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2", TORCHINDUCTOR_CACHE_DIR=str(work / ".inductor-cache"))
    for k, b in scripts:
        title = b["attrs"]["title"]
        where = f"{md.relative_to(ROOT)}:{b['line']} {title}"
        if b["attrs"].get("run") == "no":
            r = subprocess.run([PYTHON, "-m", "py_compile", title], cwd=work, capture_output=True, text=True)
            if r.returncode:
                errors.append(f"{where}: 语法错误\n{r.stderr[-2000:]}")
            continue
        cmd = [PYTHON, title]
        if b["attrs"].get("torchrun"):
            cmd = [PYTHON, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={b['attrs']['torchrun']}", title]
        try:
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            errors.append(f"{where}: 运行超过 600 秒")
            continue
        if r.returncode:
            errors.append(f"{where}: 运行失败\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if nxt and nxt["lang"] == "text" and nxt["attrs"].get("title") == "输出":
            want = [l.rstrip() for l in nxt["body"].rstrip("\n").splitlines()]
            got = [l.rstrip() for l in r.stdout.rstrip("\n").splitlines()]
            if want != got:
                errors.append(f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got))
    return len(scripts), errors


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    total, errs = 0, []
    for p in pages:
        n, e = check_page(p)
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(pages)} 个页面，{total} 个脚本，{len(errs)} 个问题（{PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
