"""运行分布式训练手册里的 PyTorch 示例，确认页面上的输出与实际运行结果一致（CPU 版 PyTorch 即可，多进程用 gloo 后端）。

约定：
  ```python title="x.py"         完整脚本：运行它，紧跟的 ```text title="输出"``` 必须与标准输出逐行一致
  ```python title="x.py" torchrun="4"   多进程脚本：用 torchrun --standalone --nproc-per-node 4 启动（只比对标准输出，通常只有 rank 0 打印）
  ```python title="x.py" run="no"   需要 GPU 的脚本：只做语法检查（py_compile），页面上的输出不做比对
  ```python title="x.py" ci="no"    本机正常运行，但在 CI 里只做语法检查（输出随机器微变的脚本，比如采样文本）
  ```python title="x.py" ci="loose" CI 里照常运行（后面的页面可能依赖它产出的文件），但不比对输出
  没有 title 的代码块是片段，不检查。同一页的脚本写在同一个目录里，可以互相 import。
  SHARED 里的目录例外：目录下的几页共用一个工作目录，按文件名顺序接力运行，前一页生成的文件留给后一页用；
  检查其中任何一页时，整个目录都会从头跑一遍（《从零训练一个小模型》拆成单独一本书后，这里暂时为空）。

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
SHARED: set[str] = set()                                    # 共用工作目录的教程目录（"从零训练"已拆成单独一本书，见 scratch/tools/check_code.py）
PYTHON = os.environ.get("PYTHON") or str(ROOT.parent / "cpp" / ".venv-py" / "bin" / "python")
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')



NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def same_output(want: list[str], got: list[str]) -> bool:
    """本机要求逐行完全一致；CI（环境变量 CI 非空）里允许数字有 5% 的相对误差（百分数 1 个百分点）：
    不同 CPU / BLAS 的浮点归约顺序不同，训练 loss、KL 这类数的第三位小数会变，文字部分仍要完全一致。"""
    if want == got:
        return True
    if not os.environ.get("CI") or len(want) != len(got):
        return False
    for w, g in zip(want, got):
        if NUM.sub("#", w) != NUM.sub("#", g):
            return False
        for ma, mb in zip(NUM.finditer(w), NUM.finditer(g)):
            x, y = float(ma.group()), float(mb.group())
            pct = w[ma.end():ma.end() + 1] == "%"              # 百分数（常常是两个数的差）按 1 个百分点算
            if abs(x - y) > (1.0 if pct else max(0.05 * max(abs(x), abs(y)), 0.011)):
                return False
    return True

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


def work_dir(md: Path) -> Path:
    rel = md.relative_to(ROOT / "docs")
    return BUILD / rel.parts[0] if rel.parts[0] in SHARED else BUILD / rel.with_suffix("")


def check_page(md: Path, fresh: bool = True) -> tuple[int, list[str]]:
    items = list(blocks(md))
    work = work_dir(md)
    if fresh and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    scripts = [(k, b) for k, b in enumerate(items) if b["lang"] == "python" and b["attrs"].get("title", "").endswith(".py")]
    for _, b in scripts:
        (work / b["attrs"]["title"]).write_text(b["body"], encoding="utf-8")
    errors = []
    env = dict(os.environ, PYTHONHASHSEED="0", OMP_NUM_THREADS="2", TORCHINDUCTOR_CACHE_DIR=str(work / ".inductor-cache"))
    if sys.platform == "darwin":                  # macOS：让 gloo 走回环网卡，避免主机名解析到外部地址导致连接失败
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    for k, b in scripts:
        title = b["attrs"]["title"]
        where = f"{md.relative_to(ROOT)}:{b['line']} {title}"
        if b["attrs"].get("run") == "no" or (os.environ.get("CI") and b["attrs"].get("ci") == "no"):   # ci="no"：采样文本这类随机器微变的输出，CI 里只做语法检查
            r = subprocess.run([PYTHON, "-m", "py_compile", title], cwd=work, capture_output=True, text=True)
            if r.returncode:
                errors.append(f"{where}: 语法错误\n{r.stderr[-2000:]}")
            continue
        cmd = [PYTHON, title]
        if b["attrs"].get("torchrun"):
            cmd = [PYTHON, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={b['attrs']['torchrun']}", title]
        try:
            r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=1800 if os.environ.get("CI") else 600, env=env)   # GitHub 运行器只有 4 核，从零训练那几页慢得多
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
            if os.environ.get("CI") and b["attrs"].get("ci") == "loose":   # ci="loose"：CI 里只要求跑通，不比对输出（采样文本、梯度范数这类随机器变的）
                pass
            elif not same_output(want, got):
                errors.append(f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got))
    return len(scripts), errors


def main(argv):
    pages = [Path(p).resolve() for p in argv] or sorted((ROOT / "docs").rglob("*.md"))
    expanded = []                                            # 共用工作目录的页面：整个目录按文件名顺序跑
    for p in pages:
        group = [p]
        if p.relative_to(ROOT / "docs").parts[0] in SHARED:
            group = sorted(p.parent.glob("*.md"))
        expanded += [q for q in group if q not in expanded]
    total, errs, started = 0, [], set()
    for p in expanded:
        w = work_dir(p)
        n, e = check_page(p, fresh=w not in started)
        started.add(w)
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(expanded)} 个页面，{total} 个脚本，{len(errs)} 个问题（{PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
