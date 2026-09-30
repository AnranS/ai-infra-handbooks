"""核对书里引用的源码：文件路径、函数与类名、命令行参数、环境变量，在对应版本的 vLLM / SGLang 源码里是否还存在。

推理框架更新很快，书里写的 `vllm/v1/core/sched/scheduler.py`、`Scheduler.schedule`、`--enable-deterministic-inference`、
`VLLM_BATCH_INVARIANT` 这类名字，升级一个版本就可能改名或移走。这个脚本逐行扫描八本手册的 Markdown：

- 路径：反引号里以 `vllm/`、`csrc/`、`rust/`（vLLM）或 `srt/`、`sglang/`（SGLang）开头的路径，必须存在（缺失算错误）；
- 符号：和路径出现在同一行的 `Class.method`、`function_name`、`xxx.py` 这类名字，要在这一行引用的文件里出现，
  否则在整个仓库里找；整个仓库都找不到说明它已经改名或删掉了（警告）。`--strict` 时"别的文件里有"也算警告；
- 参数与环境变量：`--some-flag`、`VLLM_*` / `SGLANG_*`，要在 vLLM 或 SGLang 的源码里出现过（找不到算警告）。

用法（源码不在仓库里，先各自解压好对应版本）：

    python tools/check_sources.py --vllm ~/src-reading/vllm-0.30.0 --sglang ~/src-reading/sglang-0.5.20
    python tools/check_sources.py ... --only serving --warnings      # 只看一本书，并列出警告

有错误时返回码为 1。升级书里引用的框架版本时跑一遍，按报告逐条修改正文。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ["python", "cpp", "llm", "cuda", "train", "serving", "minisgl", "cs"]
TICK = re.compile(r"`([^`\n]+)`")
PATH = re.compile(r"^(?:python/)?(vllm|csrc|rust|srt|sglang)/[A-Za-z0-9_./-]*$")
SYMBOL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*(?:\(\))?$")
FLAG = re.compile(r"^--[a-z][a-z0-9-]+$")
ENV = re.compile(r"^(?:VLLM|SGLANG)_[A-Z0-9_]+$")
CODE_EXT = (".py", ".rs", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".c")
# 书里提到的、不在 vLLM / SGLang 仓库里的外部文件（比如 DeepSeek 开源的 EPLB 仓库里的 eplb.py）
EXTERNAL = {"eplb.py"}
# 常见的普通词、类型名和 Python 内置名：出现在路径旁边也不当作源码符号
COMMON = {"None", "True", "False", "self", "cls", "torch", "Tensor", "int", "float", "str", "bool", "list", "dict", "tuple",
          "numpy", "np", "json", "msgpack", "async", "await", "yield", "return", "import", "from", "def", "class", "fp8",
          "bf16", "fp16", "fp32", "cuda", "cpu", "gpu", "rank", "tp", "ep", "dp", "pp", "cp", "kv", "http", "grpc"}


class Repo:
    def __init__(self, name: str, root: Path, prefixes: dict[str, str]):
        self.name, self.root, self.prefixes = name, root, prefixes
        self._grep_cache: dict[str, bool] = {}
        self._files: set[str] | None = None

    def resolve(self, ref: str) -> Path | None:
        """把书里写的路径换算成仓库里的路径；不属于这个仓库时返回 None"""
        ref = ref.removeprefix("python/")
        for prefix, sub in self.prefixes.items():
            if ref == prefix.rstrip("/") or ref.startswith(prefix):
                return self.root / sub / ref[len(prefix):]
        return None

    def contains(self, needle: str, word: bool = False) -> bool:
        """整个仓库的代码里是否出现过这个字符串（word=True 时按整词匹配；结果缓存）"""
        key = ("w:" if word else "s:") + needle
        if key not in self._grep_cache:
            cmd = ["grep", "-rqs" + ("w" if word else "") + "F", "--include=*.py", "--include=*.rs", "--include=*.cu",
                   "--include=*.cuh", "--include=*.cpp", "--include=*.h", "--include=*.toml", "--include=*.md", needle, str(self.root)]
            self._grep_cache[key] = subprocess.run(cmd).returncode == 0
        return self._grep_cache[key]

    def has_file(self, name: str) -> bool:
        """仓库里有没有叫这个名字的文件"""
        if self._files is None:
            self._files = {p.name for p in self.root.rglob("*") if p.is_file()}
        return name in self._files


def defines(text: str, name: str) -> bool:
    """文件里有没有定义这个名字（Python 的 def / class / 赋值，Rust 的 fn / struct / enum，C++ 的函数和类）"""
    n = re.escape(name)
    return re.search(rf"(^|\W)(def|class|fn|struct|enum|trait|impl|type|const|static)\s+{n}\b|^\s*{n}\s*[:=]|\b{n}\s*\(|\b{n}\s*=", text,
                     re.M) is not None


def book_definitions() -> tuple[set[str], set[str]]:
    """书里自己的代码定义了哪些名字（代码块和仓库里的 .py 文件中的 class / def / 赋值），以及有哪些文件（代码块的 title 和真实文件）"""
    names, files = set(), set()
    sources = [p.read_text(encoding="utf-8", errors="replace") for p in ROOT.glob("*/python/**/*.py")]
    for book in BOOKS:
        for md in (ROOT / book / "docs").rglob("*.md"):
            text = md.read_text(encoding="utf-8")
            files.update(re.findall(r'^\s*`{3,}\w*[^\n]*title="([^"]+)"', text, re.M))
            sources += re.findall(r"^\s*`{3,}[^\n]*\n(.*?)^\s*`{3,}\s*$", text, re.M | re.S)
    for src in sources:
        names.update(re.findall(r"\b(?:class|def)\s+([A-Za-z_]\w*)", src))
        names.update(re.findall(r"^\s*(?:self\.)?([A-Za-z_]\w*)\s*[:=]", src, re.M))
    files.update(p.name for p in ROOT.glob("*/python/**/*.py"))
    return names, files


def scan(md: Path):
    """逐行给出 (行号, 反引号里的内容列表)；跳过代码块"""
    fence = None
    for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"^\s*(```+|~~~+)", line)
        if m:
            fence = None if fence and m.group(1)[0] == fence[0] else (fence or m.group(1))
            continue
        if fence is None:
            yield i, line, TICK.findall(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vllm", type=Path, required=True, help="vLLM 源码的根目录（含 vllm/、csrc/）")
    ap.add_argument("--sglang", type=Path, required=True, help="SGLang 源码中含 sglang/ 包的目录")
    ap.add_argument("--only", choices=BOOKS, help="只检查一本书")
    ap.add_argument("--warnings", action="store_true", help="列出所有警告（默认只列错误和警告的数量）")
    ap.add_argument("--strict", action="store_true", help="符号在同一行引用的文件里找不到、但仓库别处有时也警告")
    args = ap.parse_args()

    vllm = Repo("vLLM", args.vllm.expanduser(), {"vllm/": "vllm", "csrc/": "csrc", "rust/": "rust"})
    sgl = Repo("SGLang", args.sglang.expanduser(), {"sglang/": "sglang", "srt/": "sglang/srt"})
    repos = [vllm, sgl]
    own_names, own_files = book_definitions()                   # 书里自己的代码（迷你引擎、mini_llm 等）定义的名字和文件
    for r in repos:
        if not r.root.is_dir():
            print(f"找不到 {r.name} 源码目录：{r.root}")
            return 2

    errors, warnings = defaultdict(list), defaultdict(list)
    counts = {"路径": 0, "符号": 0, "参数": 0, "环境变量": 0}
    text_cache: dict[Path, str] = {}

    def read(p: Path) -> str:
        if p not in text_cache:
            text_cache[p] = p.read_text(encoding="utf-8", errors="replace")
        return text_cache[p]

    books = [args.only] if args.only else BOOKS
    for book in books:
        for md in sorted((ROOT / book / "docs").rglob("*.md")):
            page = md.relative_to(ROOT).as_posix()
            for lineno, line, ticks in scan(md):
                files = []                                      # 这一行引用的源码文件（用来核对同一行的符号）
                for t in ticks:
                    t = t.strip()
                    if not PATH.match(t):
                        continue
                    hits = [(r, r.resolve(t)) for r in repos if r.resolve(t) is not None]
                    if not hits:
                        continue
                    counts["路径"] += 1
                    found = [(r, p) for r, p in hits if p.exists() or p.with_suffix("").exists()]
                    if not found:
                        errors[page].append(f"{lineno}: 路径 `{t}` 在 {'、'.join(r.name for r, _ in hits)} 里不存在")
                    else:
                        files += [(r, p) for r, p in found if p.is_file() and p.suffix in CODE_EXT]
                if files:
                    for t in ticks:
                        t = t.strip().removesuffix("()")
                        if PATH.match(t) or ENV.match(t) or FLAG.match(t):
                            continue
                        if t.endswith(CODE_EXT) and re.match(r"^[A-Za-z0-9_.-]+$", t):   # 只写了文件名：仓库里要有这个文件
                            counts["符号"] += 1
                            if t not in EXTERNAL and t not in own_files and not any(r.has_file(t) for r in repos):
                                warnings[page].append(f"{lineno}: 文件 `{t}` 在 vLLM、SGLang 里都找不到")
                            continue
                        if not SYMBOL.match(t) or t in COMMON or len(t) < 4 or "." not in t and "_" not in t \
                                and not re.search(r"[a-z][A-Z]|^[A-Z][a-z]", t):
                            continue                           # 只看像代码符号的名字：带点号、带下划线或驼峰
                        parts = t.split(".")
                        if parts[0] in ("torch", "vllm", "sglang", "np", "os", "sys"):
                            continue
                        counts["符号"] += 1
                        if any(all(defines(read(p), part) or part in read(p) for part in parts) for _, p in files):
                            continue
                        missing = [part for part in parts if part not in own_names and not any(r.contains(part, word=True) for r in repos)]
                        where = "、".join(p.relative_to(r.root).as_posix() for r, p in files)
                        if missing:
                            warnings[page].append(f"{lineno}: `{t}` 在 vLLM、SGLang 里都找不到"
                                                  f"（{'、'.join(missing)}；这一行引用的文件：{where}）")
                        elif args.strict:
                            warnings[page].append(f"{lineno}: `{t}` 不在 {where} 里，但仓库别处有（检查文件是否写对）")
                for t in ticks:
                    t = t.strip()
                    if FLAG.match(t) and (book == "serving" or re.search(r"vLLM|SGLang|vllm|sglang", line)):
                        counts["参数"] += 1
                        flag = t[2:]
                        if not any(r.contains(flag) or r.contains(flag.replace("-", "_")) for r in repos):
                            warnings[page].append(f"{lineno}: 参数 `{t}` 在 vLLM、SGLang 的源码里都找不到")
                    elif ENV.match(t):
                        counts["环境变量"] += 1
                        repo = vllm if t.startswith("VLLM_") else sgl
                        if not repo.contains(t):
                            warnings[page].append(f"{lineno}: 环境变量 `{t}` 在 {repo.name} 的源码里找不到")

    for title, bag in (("错误", errors), ("警告", warnings)):
        n = sum(len(v) for v in bag.values())
        if not n:
            continue
        print(f"{title}：{n} 条")
        if title == "警告" and not args.warnings:
            print("  （加 --warnings 列出）")
            continue
        for page in sorted(bag):
            print(f"  {page}")
            for msg in bag[page]:
                print(f"    {msg}")
    print("检查了 " + "、".join(f"{v} 处{k}" for k, v in counts.items()) +
          f"；{sum(len(v) for v in errors.values())} 个错误、{sum(len(v) for v in warnings.values())} 个警告")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
