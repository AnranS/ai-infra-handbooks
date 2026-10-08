"""有 NVIDIA 显卡时的自检：把手册里**必须真卡才能跑**的例子跑一遍，并打印一张结果表。

正文里的绝大多数代码在 CPU 上就能验证（CI 每次推送都在跑）；这个脚本补的是另一半：
标了 `run="no"` 的 GPU 脚本、CUDA 手册的 .cu 示例、以及需要显存的冒烟测试。

用法：先运行 bash setup-gpu.sh，然后
    .venv-gpu/bin/python gpu_check.py              # 全部，约 15 分钟
    .venv-gpu/bin/python gpu_check.py bench cuda   # 只跑某几项
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

# (名字, 说明, 工作目录, 命令, 超时秒数)
CHECKS = [
    ("bench", "基准测试：三种计时口径、工作集与有效带宽、空 kernel 的启动开销", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 bench_timing.cu -o /tmp/bench_timing && /tmp/bench_timing"], 600),
    ("cuda-basic", "CUDA 手册：向量加法与访存模式", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 vector_add.cu -o /tmp/vector_add && /tmp/vector_add && "
                    f"nvcc -O3 -arch={ARCH} -std=c++17 access_pattern.cu -o /tmp/access_pattern && /tmp/access_pattern"], 900),
    ("cuda-reduce", "CUDA 手册：归约的七个版本，测有效带宽", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 reduction.cu -o /tmp/reduction && /tmp/reduction"], 900),
    ("cuda-gemm", "CUDA 手册：GEMM 优化之路", "cuda/examples",
     ["bash", "-c", f"nvcc -O3 -arch={ARCH} -std=c++17 gemm.cu -o /tmp/gemm && /tmp/gemm"], 1200),
    ("scratch-gpu", "从零训练一个小模型：GPU 版训练脚本（只跑几十步）", "scratch/examples",
     [PY, "train_gpu.py"], 1800),
    ("llm", "大模型原理：从零组装 LLaMA 并加载 Qwen3-0.6B", "llm",
     [PY, "tools/check_code.py", "docs/transformer/build-llm.md"], 1800),
    ("minisgl", "手写 mini-sglang：Radix Cache、HTTP 服务、TP=2", "minisgl",
     [PY, "-m", "pytest", "-q", "tests/test_ch09_radix.py", "tests/test_ch15_server.py", "tests/test_ch16_tp.py"], 1800),
    ("practice-cuda", "练习题：CUDA 题在真卡上判题（没有 GPU 时自动退回模拟器）", ".",
     [PY, "practice/judge.py", "check", "cu-reduction", "cu-transpose"], 900),
    ("practice-bench", "练习题：实测本机的带宽、算力与 all-reduce（性能档位的分母）", ".",
     [PY, "practice/judge.py", "bench", "--quick"], 900),
]


def env_summary() -> str:
    lines = []
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
    except Exception as e:  # noqa: BLE001
        lines.append(f"PyTorch：导入失败（{e}）")
    lines.append(f"编译目标：-arch={ARCH}（改环境变量 CUDA_ARCH 可以指定，比如 sm_90）")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    print(env_summary(), "\n", flush=True)
    if not EX.exists():
        print(f"没有 {EX.relative_to(ROOT)}/：先运行 python3 tools/export_examples.py", file=sys.stderr)
        return 1
    env = dict(os.environ, PYTHON=PY,
               PYTHONPATH=str(ROOT / "minisgl" / "python") + os.pathsep + str(ROOT / "minisgl" / "tests"))
    env["PATH"] = str(Path(PY).parent) + os.pathsep + env.get("PATH", "")
    results = []
    for name, desc, cwd, cmd, timeout in CHECKS:
        if argv and name not in argv:
            continue
        start = time.time()
        try:
            r = subprocess.run(cmd, cwd=ROOT / cwd, env=env, capture_output=True, text=True, timeout=timeout)
            text, ok = r.stdout + r.stderr, r.returncode == 0
        except subprocess.TimeoutExpired as e:
            text, ok = f"超时（{timeout} 秒）\n{e.stdout or ''}{e.stderr or ''}", False
        except FileNotFoundError as e:
            text, ok = str(e), False
        (LOGS / f"{name}.log").write_text(text, encoding="utf-8")
        took = time.time() - start
        tail = "\n".join(text.strip().splitlines()[-12:])
        results.append((name, desc, ok, took, tail))
        print(f"{'✓' if ok else '✗'}  {desc}（{took:.0f} 秒）", flush=True)
        if not ok:
            print("   " + tail.replace("\n", "\n   "), flush=True)
    failed = [r for r in results if not r[2]]
    print(f"\n{len(results) - len(failed)} / {len(results)} 项通过；完整日志在 {LOGS.relative_to(ROOT)}/", flush=True)
    if any(r[0] == "bench" and r[2] for r in results):
        print(f"\n基准测试的完整输出在 {(LOGS / 'bench.log').relative_to(ROOT)}，"
              "可以贴到《第一个 CUDA 程序》的「让测出来的数字可信」一节里。", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
