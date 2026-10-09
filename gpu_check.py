"""学习环境自检：把手册里本机该能跑的例子跑一遍，并打印一张结果表。

主要补的是 CI 跑不到的那一半：标了 `run="no"` 的 GPU 脚本、CUDA 手册的 .cu 示例、
需要显存的冒烟测试。另外三项（cpp / cs / python）不需要显卡，但要本机有 g++、gcc
和 Python 3.14——缺了会自动跳过，不算失败。

用法：先运行 bash setup-gpu.sh，然后
    .venv-gpu/bin/python gpu_check.py              # 全部 14 项，约 30 分钟
    .venv-gpu/bin/python gpu_check.py bench cuda   # 只跑某几项
    .venv-gpu/bin/python gpu_check.py cpp cs python   # 只跑不需要显卡的那三项
每一项的完整日志在 build/gpu_check/ 下。把 bench 那一项的输出贴给维护者，
就能把《第一个 CUDA 程序》里「让测出来的数字可信」那一节的数字补齐。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "build" / "gpu_check"
VENV = ROOT / ".venv-gpu" / "bin" / "python"
PY = str(VENV) if VENV.exists() else sys.executable
EX = ROOT / "cuda" / "examples"          # 正文里的 .cu 示例，由 tools/export_examples.py 导出
ARCH = os.environ.get("CUDA_ARCH", "native")

# (名字, 说明, 工作目录, 命令, 超时秒数, 需要先存在的东西)
CHECKS = [
    ("bench", "基准测试：三种计时口径、工作集与有效带宽、空 kernel 的启动开销", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 bench_timing.cu -o /tmp/bench_timing && /tmp/bench_timing"], 600, None),
    ("cuda-basic", "CUDA 手册：向量加法与访存模式", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 vector_add.cu -o /tmp/vector_add && /tmp/vector_add && "
                    f"nvcc -O3 -arch={ARCH} -std=c++17 access_pattern.cu -o /tmp/access_pattern && /tmp/access_pattern"], 900, None),
    ("cuda-reduce", "CUDA 手册：归约的七个版本，测有效带宽", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 reduction.cu -o /tmp/reduction && /tmp/reduction"], 900, None),
    ("cuda-gemm", "CUDA 手册：GEMM 优化之路", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 gemm.cu -o /tmp/gemm && /tmp/gemm"], 1200, None),
    ("scratch-gpu", "从零训练一个小模型：GPU 版训练脚本（29M 模型，冒烟跑 20 步）", "scratch/examples",
     # train_gpu.py 读 prepare.py 产出的 tokens.pt，所以先跑一遍第一章的脚本（约 4 秒）；
     # 正式训练是 2000 步，自检只要确认跑得通，所以用把 STEPS 改成 20 的那份副本
     [PY, "-c", "import runpy, pathlib; "
                "runpy.run_path('prepare.py', run_name='__main__') if not pathlib.Path('tokens.pt').exists() else None; "
                "runpy.run_path('train_gpu_smoke.py', run_name='__main__')"], 1800, None),
    ("llm", "大模型原理：从零组装 LLaMA 并加载 Qwen3-0.6B", "llm",
     [PY, "tools/check_code.py", "docs/transformer/build-llm.md"], 1800, "llm/models/Qwen3-0.6B/config.json"),
    ("minisgl", "手写 mini-sglang：Radix Cache 与 HTTP 服务（单卡）", "minisgl",
     [PY, "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider",
      "tests/test_ch09_radix.py", "tests/test_ch15_server.py"], 1800,
     ("all", "minisgl/models/Qwen3-0.6B/config.json", ("module", "flashinfer"))),
    ("minisgl-tp", "手写 mini-sglang：张量并行（要 2 张以上的卡）", "minisgl",
     [PY, "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider", "tests/test_ch16_tp.py"], 1800, ("gpus", 2)),
    ("triton", "CUDA 手册：Triton 的四个 kernel 在真卡上跑（本机校验只用解释器）", "cuda/examples",
     [PY, "-c", "import runpy; [runpy.run_path(f, run_name='__main__') for f in "
                "('triton_add.py', 'triton_softmax.py', 'triton_rmsnorm.py', 'triton_matmul.py')]"],
     1800, ("module", "triton")),
    # 下面三项不需要显卡，但同一台机器上也该能跑：这三本书的代码校验在 CI 里跑的是另一套环境
    ("cpp", "C++ 进阶：全书程序在 ASan / UBSan / TSan 下编译运行", "cpp",
     [PY, "tools/check_code.py"], 3600, ("cmd", "g++")),
    ("cs", "计算机基础：操作系统、网络那几章的 C 和 Python 程序", "cs",
     [PY, "tools/check_code.py"], 3600, ("cmd", "gcc")),
    ("python", "Python 进阶：正文代码块与 doctest（要 Python 3.14）", "python",
     [str(ROOT / "python" / ".venv-check" / "bin" / "python"), "tools/check_examples.py"], 1800,
     "python/.venv-check/bin/python"),
    ("practice-cuda", "练习题：CUDA 题在真卡上判题（没有 GPU 时自动退回模拟器）", ".",
     [PY, "practice/judge.py", "check", "cu-reduction", "cu-transpose-smem"], 900, None),
    ("practice-bench", "练习题：实测本机的带宽、算力与 all-reduce（性能档位的分母）", ".",
     [PY, "practice/judge.py", "bench", "--quick"], 900, None),
]


def env_summary() -> tuple[str, int]:
    """环境摘要 + 卡数。import torch 要初始化 CUDA，有时要十几秒，所以只做一次，结果给后面复用。"""
    lines = []
    gpus = 0
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        lines += [f"GPU：{l.strip()}" for l in out.splitlines()]
    else:
        lines.append("GPU：没有找到 nvidia-smi")
    nvcc = shutil.which("nvcc")
    if nvcc:
        ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
        lines.append("nvcc：" + " ".join(v.strip() for v in ver))
    else:
        lines.append("nvcc：没有（CUDA 手册的 .cu 示例会失败）")
    try:
        out = subprocess.run([PY, "-c", "import torch; print(torch.__version__, torch.version.cuda, "
                              "torch.cuda.is_available(), torch.cuda.device_count())"],
                             capture_output=True, text=True, timeout=180).stdout.split()
        lines.append(f"PyTorch {out[0]}（CUDA {out[1]}），可用 {out[2]}，{out[3]} 张卡")
        gpus = int(out[3])
    except Exception as e:  # noqa: BLE001
        lines.append(f"PyTorch：导入失败（{e}）")
    lines.append(f"编译目标：-arch={ARCH}（改环境变量 CUDA_ARCH 可以指定，比如 sm_90）")
    return "\n".join(lines), gpus


def _why(need, gpus: int) -> str:
    """这一项为什么跑不了：needs 里没满足的那一条翻译成一句人话"""
    if isinstance(need, str):
        tail = "下模型，别带 SKIP_MODELS" if "models" in need else ""
        return f"缺 {need}，重新运行 bash setup-gpu.sh {tail}".rstrip()
    if need[0] == "module":
        return f"没装 {need[1]}，装法：uv pip install --python {PY} {need[1]}"
    if need[0] == "cmd":
        return f"本机没有 {need[1]}，装法：apt install -y build-essential cmake"
    return f"本机 {gpus} 张卡，这一项要 {need[1]} 张"


def _have(need, gpus: int) -> bool:
    """needs 里的一项是否已经满足：文件路径、可执行文件、Python 模块，或者卡数"""
    if isinstance(need, str):
        return (ROOT / need).exists()
    if need[0] == "module":
        return subprocess.run([PY, "-c", f"import {need[1]}"], capture_output=True).returncode == 0
    if need[0] == "cmd":
        return shutil.which(need[1]) is not None
    return gpus >= need[1]


def main(argv: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    print("正在检查环境（第一次 import torch 要初始化 CUDA，可能要十几秒）…", flush=True)
    summary, gpus = env_summary()                  # 卡数顺带拿到，下面不再单独跑一次 torch
    print(summary, "\n", flush=True)
    if not EX.exists():
        print(f"没有 {EX.relative_to(ROOT)}/：先运行 python3 tools/export_examples.py", file=sys.stderr)
        return 1
    env = dict(os.environ, PYTHON=PY,
               PYTHONPATH=str(ROOT / "minisgl" / "python") + os.pathsep + str(ROOT / "minisgl" / "tests"))
    env.setdefault("HF_HUB_OFFLINE", "1")          # 模型都在本地 models/ 下，别去连 huggingface
    env["PYTHONUNBUFFERED"] = "1"                  # 日志要能一边跑一边看（tail -f），不能攒到最后才写
    work = LOGS / "work"                           # examples/ 是仓库里的文件，跑的时候复制一份，不往里写产物
    if work.exists():
        shutil.rmtree(work)
    for book in ("cuda", "scratch"):
        src = ROOT / book / "examples"
        if src.exists():
            shutil.copytree(src, work / book)
    smoke = work / "scratch" / "train_gpu.py"      # 自检只跑 20 步，正式训练仍然是页面上的 2000 步
    if smoke.exists():
        text = smoke.read_text(encoding="utf-8")
        line = "MICRO, ACCUM, STEPS, LR, WARMUP = 32, 4, 2000, 1e-3, 100"
        assert text.count(line) == 1, "train_gpu.py 的超参数那一行改过了，更新 gpu_check.py"
        (work / "scratch" / "train_gpu_smoke.py").write_text(
            text.replace(line, line.replace(", 2000,", ", 20,")), encoding="utf-8")
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    results = []
    skipped = []
    for name, desc, cwd, cmd, timeout, needs in CHECKS:
        if argv and name not in argv:
            continue
        if isinstance(needs, tuple) and needs[0] == "all":        # 一组条件：挑出第一个没满足的
            needs = next((n for n in needs[1:] if not _have(n, gpus)), None)
        elif needs is not None and _have(needs, gpus):
            needs = None
        if needs is not None:                                      # 到这儿 needs 就是没满足的那一条
            skipped.append(name)
            print(f"-  {desc}：跳过（{_why(needs, gpus)}）", flush=True)
            continue
        start = time.monotonic()   # 不用 time.time()：系统时钟被 NTP 往回拨过，耗时会算成负数
        log = LOGS / f"{name}.log"
        here = work / cwd.split("/")[0] if cwd.endswith("/examples") else ROOT / cwd
        print(f"   跑着呢，进度看 {log.relative_to(ROOT)}（tail -f）", flush=True)
        try:                                       # 日志边跑边写，长任务可以在另一个终端 tail -f
            with open(log, "w", encoding="utf-8") as f:
                ok = subprocess.run(cmd, cwd=here, env=env, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"\n超时（{timeout} 秒）\n")
            ok = False
        except FileNotFoundError as e:
            log.write_text(str(e), encoding="utf-8")
            ok = False
        text = log.read_text(encoding="utf-8", errors="replace")
        took = time.monotonic() - start
        tail = "\n".join(text.strip().splitlines()[-12:])
        results.append((name, desc, ok, took, tail))
        print(f"{'✓' if ok else '✗'}  {desc}（{took:.0f} 秒）", flush=True)
        if not ok:
            print("   " + tail.replace("\n", "\n   "), flush=True)
    failed = [r for r in results if not r[2]]
    tail_note = f"，跳过 {len(skipped)} 项（{'、'.join(skipped)}）" if skipped else ""
    print(f"\n{len(results) - len(failed)} / {len(results)} 项通过{tail_note}；完整日志在 {LOGS.relative_to(ROOT)}/", flush=True)
    if any(r[0] == "bench" and r[2] for r in results):
        print(f"\n基准测试的完整输出在 {(LOGS / 'bench.log').relative_to(ROOT)}，"
              "可以贴到《第一个 CUDA 程序》的「让测出来的数字可信」一节里。", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
