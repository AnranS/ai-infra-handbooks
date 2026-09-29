"""测本机的性能基准：显存带宽、矩阵乘可达算力、多卡 all-reduce 总线带宽。

用法：python practice/judge.py bench          （完整测一遍，约 1～2 分钟）
      python practice/judge.py bench --quick  （缩小规模，十几秒）

结果写到 practice/workspace/baseline.json。性能门槛（冲刺计划里的 L3 验收、CUDA 题的性能档位）都应该拿"实测"而不是规格表上的峰值做分母：
规格峰值往往达不到，拿它做分母会把一个其实已经很好的 kernel 判成"才 70%"。
支持 NVIDIA GPU（CUDA）、Apple GPU（MPS）和 CPU；多张 NVIDIA GPU 时再测 NCCL all-reduce。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "workspace" / "baseline.json"

# 常见 GPU 的规格（显存带宽 GB/s、稠密 BF16 TFLOPS），只用来和实测对比
SPECS = [
    (("h100", "pcie"), "H100 PCIe", 2000, 756),
    (("h100",), "H100 SXM", 3350, 989),
    (("h800",), "H800 SXM", 3350, 989),
    (("h20",), "H20", 4000, 148),
    (("a100", "pcie", "80gb"), "A100 80GB PCIe", 1935, 312),
    (("a100", "80gb"), "A100 80GB SXM", 2039, 312),
    (("a100",), "A100 40GB", 1555, 312),
    (("l40s",), "L40S", 864, 362),
    (("4090",), "RTX 4090", 1008, 165),
]


def spec_for(name: str):
    low = name.lower()
    for keys, label, bw, tf in SPECS:
        if all(k in low for k in keys):
            return {"name": label, "bandwidth_gbs": bw, "bf16_tflops": tf}
    return None


def pick_device(torch):
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", "Apple GPU (MPS)"
    import platform

    return "cpu", f"{platform.processor() or platform.machine()} CPU（{torch.get_num_threads()} 线程）"


def sync(torch, dev):
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def timeit(torch, dev, fn, iters, warmup=2) -> float:
    """平均每次的秒数。"""
    for _ in range(warmup):
        fn()
    sync(torch, dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(torch, dev)
    return (time.perf_counter() - t0) / iters


def bandwidth_gbs(torch, dev, nbytes: int, iters: int) -> float:
    x = torch.empty(nbytes // 4, dtype=torch.float32, device=dev).fill_(1.0)
    y = torch.empty_like(x)
    t = timeit(torch, dev, lambda: y.copy_(x), iters)
    return 2 * nbytes / t / 1e9            # 读一遍、写一遍


def matmul_tflops(torch, dev, dtype, sizes, iters: int) -> float:
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device=dev).to(dtype)
        b = torch.randn(n, n, device=dev).to(dtype)
        t = timeit(torch, dev, lambda: a @ b, iters)
        best = max(best, 2 * n ** 3 / t / 1e12)
    return best


def _allreduce_worker(rank, world, backend, sizes, iters, port, q):
    import torch
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(backend, rank=rank, world_size=world)
    dev = "cpu"
    if backend == "nccl":
        torch.cuda.set_device(rank)
        dev = f"cuda:{rank}"
    res = {}
    for nbytes in sizes:
        x = torch.ones(nbytes // 4, dtype=torch.float32, device=dev)
        for _ in range(2):
            dist.all_reduce(x)
        if backend == "nccl":
            torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        if backend == "nccl":
            torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / iters
        algbw = nbytes / t / 1e9
        res[nbytes] = algbw * 2 * (world - 1) / world       # 总线带宽：和单卡链路带宽直接可比
    if rank == 0:
        q.put(res)
    dist.destroy_process_group()


def allreduce_busbw(world: int, backend: str, sizes, iters: int) -> dict:
    import random

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = random.randint(20000, 40000)
    procs = [ctx.Process(target=_allreduce_worker, args=(r, world, backend, sizes, iters, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    res = q.get(timeout=600)
    for p in procs:
        p.join()
    return res


def fmt_bytes(n: int) -> str:
    return f"{n / 2 ** 20:.0f} MiB" if n >= 2 ** 20 else f"{n / 2 ** 10:.0f} KiB"


def run(quick: bool = False, cpu_procs: int = 0) -> dict:
    try:
        import torch
    except ImportError:
        raise SystemExit("基准测试需要 PyTorch：先按 practice/README.md 安装本地环境")
    dev, name = pick_device(torch)
    print(f"设备：{name}（{dev}）")
    small = quick or dev == "cpu"
    result = {"device": dev, "name": name, "time": datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__}

    nbytes = (64 if small else 1024) * 2 ** 20
    result["bandwidth_gbs"] = bandwidth_gbs(torch, dev, nbytes, iters=5 if small else 20)
    print(f"  {'内存' if dev == 'cpu' else '显存'}带宽（{fmt_bytes(nbytes)} 拷贝，读 + 写）：{result['bandwidth_gbs']:.0f} GB/s")

    sizes = [512, 1024] if small else [2048, 4096, 8192]
    dtypes = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    if dev == "cpu":
        dtypes.pop("fp16")
    result["matmul_tflops"] = {}
    for label, dt in dtypes.items():
        if dev == "cuda" and label == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
        try:
            v = matmul_tflops(torch, dev, dt, sizes, iters=3 if small else 10)
        except (RuntimeError, TypeError) as e:          # 例如老版本 macOS 的 MPS 不支持 bf16
            print(f"  矩阵乘 {label}：跳过（{str(e).splitlines()[0][:60]}）")
            continue
        result["matmul_tflops"][label] = v
        print(f"  矩阵乘 {label}（方阵 {'/'.join(map(str, sizes))} 中最快的）：{v:.2f} TFLOPS")
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        v = matmul_tflops(torch, dev, torch.float32, sizes, iters=3 if small else 10)
        result["matmul_tflops"]["tf32"] = v
        print(f"  矩阵乘 tf32：{v:.2f} TFLOPS")
        torch.backends.cuda.matmul.allow_tf32 = False

    world, backend = 0, ""
    if dev == "cuda" and torch.cuda.device_count() >= 2:
        world, backend = torch.cuda.device_count(), "nccl"
    elif cpu_procs >= 2:
        world, backend = cpu_procs, "gloo"
    if world:
        ar_sizes = [2 ** 20, 64 * 2 ** 20] if small else [2 ** 20, 64 * 2 ** 20, 512 * 2 ** 20]
        res = allreduce_busbw(world, backend, ar_sizes, iters=5 if small else 20)
        result["allreduce_busbw_gbs"] = {fmt_bytes(k): v for k, v in res.items()}
        result["allreduce"] = {"world": world, "backend": backend}
        for k, v in res.items():
            print(f"  all-reduce（{world} 路 {backend}，{fmt_bytes(k)}）：总线带宽 {v:.1f} GB/s")

    spec = spec_for(name) if dev == "cuda" else None
    if spec:
        result["spec"] = spec
        bf16 = result["matmul_tflops"].get("bf16", 0)
        print(f"  对照规格（{spec['name']}）：带宽达到 {result['bandwidth_gbs'] / spec['bandwidth_gbs']:.0%}，"
              f"BF16 矩阵乘达到 {bf16 / spec['bf16_tflops']:.0%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存到 {OUT.relative_to(HERE.parent)}：性能门槛以这里的实测值为分母")
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cpu-procs", type=int, default=0, help="没有多张 GPU 时，用 N 个 CPU 进程（gloo）走一遍 all-reduce 流程")
    a = ap.parse_args()
    run(a.quick, a.cpu_procs)
