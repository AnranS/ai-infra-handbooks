"""环境自检：每本手册挑几个有代表性的例子跑一遍，确认本机（尤其是 Mac）能跑。

用法：先运行 env/setup-macos.sh，然后
    .venv/bin/python tools/mac_check.py            # 全部检查，约 10 分钟
    .venv/bin/python tools/mac_check.py cpp cuda   # 只查某几项
每一项的完整日志在 build/mac_check/ 下；有失败时把终端里的汇总发给维护者即可。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "build" / "mac_check"
VENV = ROOT / ".venv" / "bin" / "python"
PY = str(VENV) if VENV.exists() else sys.executable
PY314 = ROOT / "python" / ".venv-check" / "bin" / "python"

# (名字, 说明, 工作目录, 命令, 超时秒数)
CHECKS = [
    ("python", "Python 手册：迭代器与生成器一章", "python",
     [str(PY314), "tools/check_examples.py", "docs/core/iterators.md"], 300),
    ("cpp", "C++ 手册：移动语义、线程与 std::jthread（ThreadSanitizer）", "cpp",
     [PY, "tools/check_code.py", "docs/basics/move.md", "docs/concurrency/threads.md"], 900),
    ("cpp-ext", "C++ 手册：pybind11 与 PyTorch 扩展", "cpp",
     [PY, "tools/check_code.py", "docs/engineering/python-binding.md"], 1200),
    ("llm", "大模型原理：从零组装 LLaMA 并加载 Qwen3-0.6B", "llm",
     [PY, "tools/check_code.py", "docs/transformer/build-llm.md"], 1200),
    ("math", "数学基础：浮点误差与量化噪声（用大模型原理手册的环境）", ".",
     [PY, "math/tools/check_code.py", "math/docs/floating-point.md"], 600),
    ("serving", "推理系统：NCCL 算法估算、张量并行（torchrun + gloo）、Qwen3.5 的线性注意力", "serving",
     [PY, "tools/check_code.py", "docs/comm/nccl.md", "docs/distributed/tensor-parallel.md",
      "docs/frontier/linear-attn.md"], 1500),
    ("scratch", "从零训练一个小模型：语料与分词器一章", ".",
     [PY, "scratch/tools/check_code.py", "scratch/docs/data.md"], 900),
    ("train", "分布式训练：集合通信（torchrun，4 个进程）", "train",
     [PY, "tools/check_code.py", "docs/basics/collectives.md"], 900),
    ("cuda-emu", "CUDA 手册：在 CPU 模拟器上跑 kernel", "cuda",
     [PY, "tools/emu_run.py", "vector_add.cu", "reduction.cu", "transpose.cu"], 900),
    ("cuda-torch", "CUDA 手册：PyTorch 的 autograd 与 torch.compile", "cuda",
     [PY, "tools/check_torch.py", "docs/framework/autograd.md", "docs/framework/compile.md"], 1200),
    ("minisgl", "手写 mini-sglang：ZMQ 消息、Radix Cache、HTTP 服务、TP=2/4", "minisgl",
     [PY, "-m", "pytest", "-q", "tests/test_ch12_message.py", "tests/test_ch09_radix.py",
      "tests/test_ch15_server.py", "tests/test_ch16_tp.py"], 1800),
    ("practice", "练习题：Python、C++（sanitizer）、CUDA（模拟器）各一题", ".",
     [PY, "practice/judge.py", "check", "py-lru-cache", "cpp-spsc-ring", "cu-reduction"], 900),
]


def env_summary() -> str:
    lines = [f"系统 {platform.system()} {platform.release()}（{platform.machine()}），Python {platform.python_version()}，解释器 {PY}"]
    try:
        out = subprocess.run([PY, "-c", "import torch; print(torch.__version__, torch.backends.mps.is_available(), "
                              "torch.distributed.is_available())"], capture_output=True, text=True, timeout=120).stdout.split()
        lines.append(f"PyTorch {out[0]}，MPS {'可用' if out[1] == 'True' else '不可用'}，torch.distributed {'可用' if out[2] == 'True' else '不可用'}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"PyTorch：导入失败（{e}）")
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx:
        ver = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
        lines.append(f"C++ 编译器 {cxx}：{ver[0] if ver else '?'}")
    for tool in ("cmake", "nvcc"):
        lines.append(f"{tool}：{shutil.which(tool) or '没有'}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    print(env_summary(), "\n", flush=True)
    env = dict(os.environ, PYTHON=PY, PYTHONPATH=str(ROOT / "minisgl" / "python") + os.pathsep + str(ROOT / "minisgl" / "tests"),
               OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    if platform.system() == "Darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    results = []
    for name, desc, cwd, cmd, timeout in CHECKS:
        if argv and name not in argv:
            continue
        if cmd[0] == str(PY314) and not PY314.exists():
            results.append((name, desc, "跳过", 0.0, "没有 python/.venv-check（Python 3.14），先运行 env/setup-macos.sh"))
            print(f"-  {desc}：跳过（没有 Python 3.14 的检查环境）", flush=True)
            continue
        log = LOGS / f"{name}.log"
        start = time.time()
        try:
            r = subprocess.run(cmd, cwd=ROOT / cwd, env=env, capture_output=True, text=True, timeout=timeout)
            text, ok = r.stdout + r.stderr, r.returncode == 0
        except subprocess.TimeoutExpired as e:
            text, ok = f"超时（{timeout} 秒）\n{e.stdout or ''}{e.stderr or ''}", False
        log.write_text(text, encoding="utf-8")
        took = time.time() - start
        tail = "\n".join(l for l in text.strip().splitlines()[-12:])
        results.append((name, desc, "通过" if ok else "失败", took, tail))
        print(f"{'✓' if ok else '✗'}  {desc}（{took:.0f} 秒）", flush=True)
        if not ok:
            print("   " + tail.replace("\n", "\n   "), flush=True)
    failed = [r for r in results if r[2] == "失败"]
    print(f"\n{len(results) - len(failed)} / {len(results)} 项没有失败；完整日志在 {LOGS.relative_to(ROOT)}/", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
