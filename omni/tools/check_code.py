"""核对《SGLang-Omni 源码导读》里的命令、脚本和引用的源码：全部在本地克隆上实跑、实截。

页面里的约定：
  ```bash title="xxx.sh"           用 bash 运行，工作目录是 sglang-omni 克隆的根目录；紧跟的 ```text title="输出" 块必须与标准输出逐行一致
  ```python title="xxx.py"         用本书的 Python 环境运行：脚本写到 build/code/ 并在那里运行，
                                   PYTHONPATH = build/code + 基准提交的源码树，所以同一页里后面的脚本可以 import 前面写出的文件
  ```python title="路径 @ 提交 L31-60"
                                   引用的源码：正文必须与 `git show 提交:路径` 的这些行逐字一致；多段用逗号分开（L10-20,45-60），
                                   段与段之间在正文里写一行 `...`；不写 L 就是整个文件；语言可以是 python 以外的（rust、yaml、toml……）。
                                   路径前面写 `sglang:` 表示引用的是 SGLang 仓库（SGLANG_SRC）而不是 sglang-omni。
  ```text title="输出（本机示例）"   和机器相关的输出（耗时等），只要求命令跑通，不比对
  run="no"                         只做语法检查，不运行（文件照样写到 build/code/，供同页后面的脚本调用）
  ci="loose"                       CI 里照常运行但不比对输出（环境变量 CI 非空时生效）
  没有 title 的代码块是片段，不检查。

脚本里可用的环境变量：
  REF         本书固定的 sglang-omni 提交，默认 921ea2c8（2026-10-10 的 main）
  OMNI_SRC    sglang-omni 完整克隆（默认 ~/sglang-omni-src），git 命令在这里跑
  OMNI_TREE   基准提交导出的源码树（build/tree-<REF>），运行代码时用它，不受克隆当前检出的影响
  PYTHON      本书的 Python 环境（默认 omni/.venv-omni/bin/python，见 README）

用法：python3 tools/check_code.py [--write] [docs/xxx.md ...]      不带页面时检查所有页面；--write 把实际输出写回页面
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
CODE = BUILD / "code"
SRC = Path(os.environ.get("OMNI_SRC") or Path.home() / "sglang-omni-src").expanduser()
SGLANG_SRC = Path(os.environ.get("SGLANG_SRC") or Path.home() / "sglang-src").expanduser()
REF = os.environ.get("REF") or "921ea2c8"
PYTHON = os.environ.get("PYTHON") or str(ROOT / ".venv-omni" / "bin" / "python")
LOOSE = bool(os.environ.get("CI"))
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})(?P<lang>[\w+-]*)(?P<rest>.*)$")
ATTR = re.compile(r'(\w+)="([^"]*)"')
QUOTE = re.compile(r"^(?:(?P<repo>sglang):)?(?P<path>\S+) @ (?P<commit>[0-9A-Za-z._-]+)(?: L(?P<lines>[\d,-]+))?$")


def parse(lines: list[str]) -> list[dict]:
    out, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, j = m["fence"], i + 1
        while j < len(lines) and not (lines[j].strip().startswith(fence) and set(lines[j].strip()) == {"`"}):
            j += 1
        attrs = dict(ATTR.findall(m["rest"]))
        q = QUOTE.match(attrs.get("title", ""))
        if q:                                                     # 引用块：title="[sglang:]路径 @ 提交 L行号"
            attrs.update({"title": q["path"], "commit": q["commit"], "repo": q["repo"] or "omni"})
            if q["lines"]:
                attrs["lines"] = q["lines"]
        body = [ln[len(m["indent"]):] if ln.startswith(m["indent"]) else ln.lstrip() for ln in lines[i + 1:j]]
        out.append({"start": i, "end": j, "lang": m["lang"], "attrs": attrs, "title": attrs.get("title"),
                    "indent": m["indent"], "body": body})
        i = j + 1
    return out


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def expected_quote(b: dict) -> list[str]:
    """按提交号和行号重新截取引用的代码"""
    repo = SGLANG_SRC if b["attrs"]["repo"] == "sglang" else SRC
    text = git(repo, "show", f"{b['attrs']['commit']}:{b['title']}").split("\n")
    segs = []
    for rng in b["attrs"].get("lines", f"1-{len(text)}").split(","):
        a, _, z = rng.partition("-")
        segs.append(text[int(a) - 1:int(z or a)])
    out: list[str] = []
    for k, seg in enumerate(segs):
        if k:
            out.append("...")
        out.extend(seg)
    return out


def ensure_tree() -> Path:
    """把基准提交导出成一棵独立的源码树（只导出一次）"""
    tree = BUILD / f"tree-{REF}"
    if not (tree / "sglang_omni").is_dir():
        tree.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", str(SRC), "archive", REF], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def run_block(b: dict, env: dict[str, str]) -> tuple[int, str]:
    CODE.mkdir(parents=True, exist_ok=True)
    script = CODE / b["title"]
    script.write_text("\n".join(b["body"]) + "\n", encoding="utf-8")
    if b["lang"] == "bash":
        # 不开 pipefail：`git log | head` 这类管道里 git 收到 SIGPIPE 是正常的
        cmd, cwd = ["bash", "-eu", str(script)], SRC
    else:
        cmd, cwd = [PYTHON, str(script)], CODE
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    return r.returncode, (r.stdout + (("\n[stderr]\n" + r.stderr[-3000:]) if r.returncode else "")).rstrip("\n")


def check_page(md: Path, write: bool, env: dict[str, str]) -> list[str]:
    lines = md.read_text(encoding="utf-8").splitlines()
    blocks = parse(lines)
    problems, edits = [], []          # edits: (start, end, new_body_lines)
    for k, b in enumerate(blocks):
        where = f"{md.relative_to(ROOT)}:{b['start'] + 1}"
        if "commit" in b["attrs"] and b["title"]:
            try:
                want = expected_quote(b)
            except subprocess.CalledProcessError as e:
                problems.append(f"{where}: git show 失败：{e.stderr.strip()[:200]}")
                continue
            fence = FENCE.match(lines[b["start"]])["fence"]
            if any(ln.lstrip().startswith(fence) for ln in want):
                problems.append(f"{where}: 引用的内容里有 {fence} 开头的行，外层围栏要用更多的反引号")
                continue
            if b["body"] != want:
                problems.append(f"{where}: 引用的 {b['title']} @ {b['attrs']['commit']} 与源码不一致")
                if write:
                    edits.append((b["start"], b["end"], want))
            continue
        if b["lang"] not in ("bash", "python") or not b["title"] or not b["title"].endswith((".sh", ".py")):
            continue
        if b["attrs"].get("run") == "no":
            CODE.mkdir(parents=True, exist_ok=True)              # 不运行，但照样写出文件：同页后面的脚本可能会调用它
            (CODE / b["title"]).write_text("\n".join(b["body"]) + "\n", encoding="utf-8")
            if b["lang"] == "python":
                r = subprocess.run([sys.executable, "-m", "py_compile", str(CODE / b["title"])], capture_output=True, text=True)
                if r.returncode:
                    problems.append(f"{where}: 语法错误 {r.stderr.strip()[:200]}")
            continue
        code, got = run_block(b, env)
        if code:
            problems.append(f"{where}: {b['title']} 退出码 {code}\n{got[-1500:]}")
            continue
        nxt = blocks[k + 1] if k + 1 < len(blocks) else None
        if not (nxt and nxt["lang"] == "text" and nxt["start"] == b["end"] + 2 and (nxt["title"] or "").startswith("输出")):
            continue
        if "本机示例" in (nxt["title"] or "") or (LOOSE and b["attrs"].get("ci") == "loose"):
            continue
        want = nxt["body"]
        got_lines = got.split("\n") if got else []
        if want != got_lines:
            problems.append(f"{where}: {b['title']} 的输出与页面不一致\n--- 页面\n" + "\n".join(want[:25]) + "\n--- 实际\n" + "\n".join(got_lines[:25]))
            if write:
                edits.append((nxt["start"], nxt["end"], got_lines))
    if write and edits:
        for start, end, body in sorted(edits, reverse=True):
            indent = FENCE.match(lines[start])["indent"]
            lines[start + 1:end] = [indent + ln if ln else "" for ln in body]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已写回 {len(edits)} 个块：{md.relative_to(ROOT)}")
    return problems


def main(argv: list[str]) -> int:
    write = "--write" in argv
    pages = [ROOT / a for a in argv if not a.startswith("--")] or sorted((ROOT / "docs").rglob("*.md"))
    if not (SRC / ".git").exists():
        print(f"找不到 sglang-omni 克隆：{SRC}（设置 OMNI_SRC）")
        return 2
    try:
        git(SRC, "cat-file", "-e", f"{REF}^{{commit}}")
    except subprocess.CalledProcessError:
        print(f"克隆里没有提交 {REF}：先 git fetch，或用 REF 指定别的提交")
        return 2
    tree = ensure_tree()
    env = dict(os.environ, REF=REF, OMNI_SRC=str(SRC), OMNI_TREE=str(tree), PYTHON=PYTHON,
               PYTHONPATH=f"{CODE}{os.pathsep}{tree}", CARGO_TARGET_DIR=str(BUILD / "cargo-target"),
               OMP_NUM_THREADS="2", LC_ALL="C.UTF-8", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_VERBOSITY="error", GIT_PAGER="cat", PAGER="cat")
    total = 0
    for md in pages:
        probs = check_page(md, write, env)
        total += len(probs)
        for p in probs:
            print("✗", p)
        print(f"{'✗' if probs else '✓'} {md.relative_to(ROOT)}")
    print(f"{'全部一致' if not total else f'{total} 处不一致'}（REF={REF}，OMNI_SRC={SRC}）")
    return 1 if total and not write else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
