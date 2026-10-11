"""《手写 Transformer》的代码核对。

整本书的代码是接力的：每一章往 qwen/ 包里加文件，章末的 main 要用到前面所有章的文件。
所以不管检查哪一页，都按导航顺序从第 0 步开始把前面的页面重跑一遍，共用一个工作目录 build/examples/tutorial/。
规则和分布式训练手册一样：
  ```python title="x.py"              章末的 main：运行它，紧跟的 ```text title="输出"``` 必须与标准输出逐行一致
  ```python title="qwen/x.py"         包文件：只写进工作目录，不单独运行
  ```python title="x.py" ci="loose"   CI 里照常运行但不比对输出（带机器相关数字的脚本）
  含耗时的行（ms、秒、tok/s）只比对数字以外的部分，因为每次运行都会变。
用法：python transformer/tools/check_code.py [transformer/docs/xxx.md ...] [--write] [--only]
      --write 把实际输出写回页面的「输出」块（写完再看一遍 diff）
      --only  只运行指定页面的脚本，前面的页面只把文件写进工作目录（改一章时省时间；最终要不带它全跑一遍）
解释器：环境变量 PYTHON，默认 minisgl/.venv-llm/bin/python（有 torch、safetensors、tokenizers；transformers 只有前九章要）。
模型：models/Qwen3.5-0.8B（脚本会从工作目录往上找，也可以用 QWEN_MODEL 指定）。
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
WORK = BOOK / "build" / "examples" / "tutorial"
TRAIN_CHECK = BOOK.parent / "train" / "tools" / "check_code.py"
_VENV = BOOK.parent / "minisgl" / ".venv-llm" / "bin" / "python"
PYTHON = os.environ.get("PYTHON") or (str(_VENV) if _VENV.exists() else sys.executable)
TIMED = re.compile(r"\bms\b|秒|tok/s|分钟")
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?")


def load():
    spec = importlib.util.spec_from_file_location("train_check_code", TRAIN_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nav_order():
    """按 mkdocs.yml 的导航顺序列出页面"""
    nav = (BOOK / "mkdocs.yml").read_text(encoding="utf-8").split("\nnav:", 1)[1]
    return [BOOK / "docs" / m for m in re.findall(r"([\w./-]+\.md)\s*$", nav, re.M)]


def same(want, got):
    if want == got:
        return True
    if len(want) != len(got):
        return False
    for w, g in zip(want, got):
        if w == g:
            continue
        if TIMED.search(w) and NUM.sub("#", w) == NUM.sub("#", g):       # 耗时行：数字可以不同
            continue
        return False
    return True


def run_page(mod, md, write, run=True):
    items = list(mod.blocks(md))
    scripts = [(k, b) for k, b in enumerate(items) if b["lang"] == "python" and b["attrs"].get("title", "").endswith(".py")]
    for _, b in scripts:
        f = WORK / b["attrs"]["title"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(b["body"], encoding="utf-8")
    (WORK / "qwen").mkdir(exist_ok=True)
    (WORK / "qwen" / "__init__.py").touch()
    errors, replaced = [], {}
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONUNBUFFERED="1")
    for k, b in scripts:
        title = b["attrs"]["title"]
        if title.startswith("qwen/") or not run:
            continue
        where = f"{md.relative_to(BOOK)}:{b['line']} {title}"
        try:
            r = subprocess.run([PYTHON, title], cwd=WORK, capture_output=True, text=True, timeout=1800, env=env)
        except subprocess.TimeoutExpired:
            errors.append(f"{where}: 运行超过 1800 秒")
            continue
        if r.returncode:
            errors.append(f"{where}: 运行失败\n{(r.stdout + r.stderr)[-3000:]}")
            continue
        nxt = items[k + 1] if k + 1 < len(items) else None
        if nxt and nxt["lang"] == "text" and nxt["attrs"].get("title") == "输出":
            want = [l.rstrip() for l in nxt["body"].rstrip("\n").splitlines()]
            got = [l.rstrip() for l in r.stdout.rstrip("\n").splitlines()]
            if os.environ.get("CI") and b["attrs"].get("ci") == "loose":
                continue
            if not same(want, got):
                if write:
                    replaced[nxt["line"]] = got
                else:
                    errors.append(f"{where}: 输出和页面不一致\n--- 页面\n" + "\n".join(want) + "\n--- 实际\n" + "\n".join(got))
    if replaced:
        lines = md.read_text(encoding="utf-8").split("\n")
        for start in sorted(replaced, reverse=True):          # start 是输出块围栏行的行号（从 1 数）
            fence = re.match(r"\s*(`{3,}|~{3,})", lines[start - 1]).group(1)
            end = start                                        # 找到结束围栏
            while lines[end].strip() != fence:
                end += 1
            lines[start:end] = replaced[start]
        md.write_text("\n".join(lines), encoding="utf-8")
        print(f"已写回 {md.relative_to(BOOK)} 的 {len(replaced)} 个输出块")
    return len(scripts), errors


def main(argv):
    write, only = "--write" in argv, "--only" in argv
    args = [Path(a).resolve() for a in argv if not a.startswith("--")]
    order = nav_order()
    last = max(order.index(p) for p in args) if args else len(order) - 1
    pages = order[: last + 1]                                  # 接力：从头跑到要检查的最后一页
    mod = load()
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    total, errs = 0, []
    for p in pages:
        n, e = run_page(mod, p, write, run=not only or p in args)
        total += n
        errs += e
    for e in errs:
        print("✗", e, "\n")
    print(f"{len(pages)} 个页面，{total} 个代码文件，{len(errs)} 个问题（{PYTHON}）")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
