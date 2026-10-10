"""一章一步地把 minisgl 搭起来：每一步只有"到这一章为止"的文件，那一章的 main 必须能跑。

    python tools/steps.py derive     # 从各章声明的文件出发，反复试跑，把缺的文件补成"提前引入"，写出 steps.json
    python tools/steps.py check      # 按 steps.json 逐步搭包、跑每一章的 main（CI 用）
    python tools/steps.py tree 7     # 打印第 7 步的文件树（@@tree@@ 用的就是它）

每一步的包放在 build/steps/NN/minisgl/。子包的 __init__.py 只保留当前已存在的模块的导入
（对应读者在那一章写的那个更短的 __init__.py），其余文件原样复制。
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
PKG = BOOK / "python" / "minisgl"
OUT = BOOK / "build" / "steps"
MANIFEST = BOOK / "steps.json"
PY = os.environ.get("MINISGL_PY", sys.executable)

# 各章：页面、这一步的 main、这一章声明要写的文件（新建）、这一章改动的已有文件
CHAPTERS = [
    ("overview/tiny-engine.md", "ch00_tiny_engine", [], []),
    ("compute/core.md", "ch01_req", ["__init__.py", "core.py"], []),
    ("compute/layers.md", "ch02_layers",
     ["layers/__init__.py", "layers/base.py", "layers/linear.py", "layers/embedding.py", "layers/norm.py",
      "layers/rotary.py", "layers/activation.py", "layers/attention.py", "kernel/__init__.py", "kernel/torch_ops.py",
      "utils/__init__.py", "utils/misc.py", "utils/hf.py", "utils/torch_utils.py", "utils/registry.py", "utils/logger.py",
      "models/config.py", "models/base.py", "models/utils.py", "models/decoder.py"], []),
    ("compute/models.md", "ch03_models", ["models/__init__.py", "models/register.py", "models/weight.py"], []),
    ("compute/kvcache.md", "ch04_kvcache",
     ["kvcache/__init__.py", "kvcache/base.py", "kvcache/mha_pool.py", "kvcache/naive_cache.py", "scheduler/table.py"], []),
    ("compute/attention.md", "ch05_attention",
     ["attention/__init__.py", "attention/base.py", "attention/utils.py", "attention/torch_backend.py"], []),
    ("compute/engine.md", "ch06_engine",
     ["engine/__init__.py", "engine/config.py", "engine/engine.py", "engine/sample.py", "utils/device.py"], []),
    ("schedule/scheduler.md", "ch07_scheduler",
     ["scheduler/__init__.py", "scheduler/config.py", "scheduler/utils.py", "scheduler/decode.py",
      "scheduler/prefill.py", "scheduler/scheduler.py", "llm/__init__.py", "llm/llm.py", "env.py"], []),
    ("schedule/cache-manager.md", "ch08_cache_manager", ["scheduler/cache.py"], ["scheduler/prefill.py"]),
    ("schedule/radix-cache.md", "ch09_radix", ["kvcache/radix_cache.py"], ["kernel/torch_ops.py", "kvcache/__init__.py"]),
    ("schedule/chunked-prefill.md", "ch10_chunked", [], ["scheduler/prefill.py"]),
    ("schedule/overlap.md", "ch11_overlap", [], ["scheduler/scheduler.py"]),
    ("serve/message.md", "ch12_message",
     ["message/__init__.py", "message/utils.py", "message/backend.py", "message/tokenizer.py", "message/frontend.py",
      "utils/mp.py"], []),
    ("serve/tokenizer.md", "ch13_tokenizer",
     ["tokenizer/__init__.py", "tokenizer/tokenize.py", "tokenizer/detokenize.py", "tokenizer/server.py"], []),
    ("serve/scheduler-io.md", "ch14_scheduler_io", ["scheduler/io.py"], []),
    ("serve/api-server.md", "ch15_server",
     ["server/__init__.py", "server/args.py", "server/api_server.py", "server/launch.py", "__main__.py", "shell.py"], []),
    ("perf/tensor-parallel.md", "ch16_tp", ["distributed/__init__.py", "distributed/info.py", "distributed/impl.py"], []),
    ("perf/gpu-attention.md", "ch17_gpu_attention", ["attention/fi.py", "attention/fa.py"], []),
    ("perf/cuda-graph.md", "ch18_cuda_graph", ["engine/graph.py"], ["attention/torch_backend.py"]),
    ("perf/kernels.md", "ch19_kernels", ["kernel/cuda_ext.py"], ["kernel/__init__.py"]),
    ("perf/moe.md", "ch20_moe",
     ["layers/moe.py", "moe/__init__.py", "moe/base.py", "moe/torch_backend.py", "moe/fused.py"], ["models/utils.py"]),
    ("perf/benchmark.md", "ch21_benchmark", ["benchmark/__init__.py", "benchmark/offline.py", "benchmark/client.py"], []),
]
ALL_FILES = sorted(str(p.relative_to(PKG)) for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def declared_in(path: str) -> int | None:
    for i, (_, _, new, _) in enumerate(CHAPTERS):
        if path in new:
            return i
    return None


def files_until(k: int, preset: dict[int, list[str]]) -> list[str]:
    have: list[str] = []
    for i in range(k + 1):
        for f in CHAPTERS[i][2] + preset.get(i, []):
            if f not in have:
                have.append(f)
    for f in list(have):                       # 一个子包只要有文件，就得有 __init__.py（裁剪过的）
        d = str(Path(f).parent)
        init = f"{d}/__init__.py" if d != "." else "__init__.py"
        if init in ALL_FILES and init not in have:
            have.append(init)
    return have


def _trim_init(src: str, present: set[str], pkg_dir: str) -> str:
    """去掉 __init__.py 里对还不存在的兄弟模块的模块级导入（读者此时写的 __init__ 就是这个样子）。"""
    tree = ast.parse(src)
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            target = f"{pkg_dir}/{node.module.replace('.', '/')}.py" if pkg_dir else f"{node.module}.py"
            if target not in present and f"{pkg_dir}/{node.module}/__init__.py" not in present:
                continue
        body.append(node)
    tree.body = body
    return ast.unparse(tree) + "\n"


def build_step(k: int, preset: dict[int, list[str]]) -> Path:
    dest = OUT / f"{k:02d}" / "minisgl"
    if dest.exists():
        shutil.rmtree(dest)
    present = set(files_until(k, preset))
    for f in present:
        src = PKG / f
        d = dest / f
        d.parent.mkdir(parents=True, exist_ok=True)
        if f.endswith("__init__.py"):
            pkg_dir = str(Path(f).parent) if str(Path(f).parent) != "." else ""
            d.write_text(_trim_init(src.read_text(encoding="utf-8"), present, pkg_dir), encoding="utf-8")
        else:
            shutil.copy2(src, d)
    csrc = PKG / "kernel" / "csrc"
    if (dest / "kernel").exists() and csrc.exists():
        shutil.copytree(csrc, dest / "kernel" / "csrc", dirs_exist_ok=True)
    return dest.parent


def run_step(k: int, preset: dict[int, list[str]], timeout: int = 900) -> tuple[bool, str]:
    page, main, _, _ = CHAPTERS[k]
    step = build_step(k, preset)
    env = dict(os.environ, PYTHONPATH=f"{step}{os.pathsep}{BOOK / 'tests'}", TOKENIZERS_PARALLELISM="false",
               LOG_LEVEL="WARNING")
    r = subprocess.run([PY, str(BOOK / "examples" / f"{main}.py")], cwd=BOOK, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode == 0, r.stdout + r.stderr


_MISSING = re.compile(r"No module named '(minisgl[\w.]*)'")
_NAME = re.compile(r"cannot import name '(\w+)' from '(minisgl[\w.]*)'")
# 裁剪过的 __init__.py 里用到了被裁掉的名字：NameError 或 "module has no attribute"
_INIT_FILE = re.compile(r"File \"[^\"]*/build/steps/\d+/minisgl/([\w/]*)__init__\.py\"")
_NAMEERR = re.compile(r"NameError: name '(\w+)' is not defined")
_ATTR = re.compile(r"module '(minisgl[\w.]*)' has no attribute '(\w+)'")


def _module_file(mod: str) -> str | None:
    rel = mod.removeprefix("minisgl").lstrip(".").replace(".", "/")
    for cand in (f"{rel}.py", f"{rel}/__init__.py") if rel else ("__init__.py",):
        if cand in ALL_FILES:
            return cand
    return None


def _defining_file(name: str, pkg: str) -> str | None:
    rel = pkg.removeprefix("minisgl").lstrip(".").replace(".", "/")
    for f in ALL_FILES:
        if (rel == "" and "/" not in f) or f.startswith(rel + "/"):
            if re.search(rf"^(class|def) {name}\b|^{name} = ", (PKG / f).read_text(encoding="utf-8"), re.M):
                return f
    return None


def derive() -> dict[int, list[str]]:
    preset: dict[int, list[str]] = {}
    for k in range(1, len(CHAPTERS)):
        for _ in range(60):
            ok, log = run_step(k, preset)
            if ok:
                # 这一步新出现、却不是本章声明的文件（包括顺带带进来的 __init__.py），都记成"提前引入"
                before = set(files_until(k - 1, preset))
                for f in files_until(k, preset):
                    if f not in before and f not in CHAPTERS[k][2] and f not in preset.get(k, []):
                        preset.setdefault(k, []).append(f)
                print(f"[ok] 第 {k} 步 {CHAPTERS[k][1]}" + (f"，提前引入 {preset[k]}" if preset.get(k) else ""))
                break
            m, n, at = _MISSING.search(log), _NAME.search(log), _ATTR.search(log)
            ne, inits = _NAMEERR.search(log), _INIT_FILE.findall(log)
            if m:
                f = _module_file(m.group(1))
            elif n:
                f = _defining_file(n.group(1), n.group(2))
            elif ne and inits:                      # 裁剪过的 __init__ 用到了被裁掉的名字：最后那个 __init__ 所在的包
                f = _defining_file(ne.group(1), "minisgl." + inits[-1].strip("/").replace("/", "."))
            elif at:
                f = _defining_file(at.group(2), at.group(1))
            else:
                f = None
            if f is None or f in files_until(k, preset):
                print(f"[FAIL] 第 {k} 步 {CHAPTERS[k][1]}：\n{log[-3000:]}")
                sys.exit(1)
            preset.setdefault(k, []).append(f)
        else:
            sys.exit(f"第 {k} 步补了 60 次还没跑通")
    return preset


def write_manifest(preset: dict[int, list[str]]) -> None:
    steps = []
    for k, (page, main, new, modified) in enumerate(CHAPTERS):
        steps.append({"page": page, "main": main, "new": new, "modified": modified,
                      "preset": [(f, declared_in(f)) for f in preset.get(k, [])],
                      "files": files_until(k, preset)})
    MANIFEST.write_text(json.dumps(steps, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"写出 {MANIFEST.relative_to(BOOK)}：{len(steps)} 步，{len(steps[-1]['files'])} / {len(ALL_FILES)} 个文件")
    missing = sorted(set(ALL_FILES) - set(steps[-1]["files"]))
    if missing:
        print("没有分到任何一章的文件：", missing)


def check() -> int:
    steps = json.loads(MANIFEST.read_text(encoding="utf-8"))
    preset = {k: [f for f, _ in s["preset"]] for k, s in enumerate(steps)}
    bad = 0
    for k in range(1, len(CHAPTERS)):
        ok, log = run_step(k, preset)
        print(f"[{'ok' if ok else 'FAIL'}] 第 {k} 步 {CHAPTERS[k][1]}（{len(files_until(k, preset))} 个文件）")
        if not ok:
            bad += 1
            print(log[-2000:])
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "derive":
        write_manifest(derive())
    elif cmd == "check":
        sys.exit(1 if check() else 0)
    elif cmd == "tree":
        steps = json.loads(MANIFEST.read_text(encoding="utf-8"))
        print("\n".join(steps[int(sys.argv[2])]["files"]))
