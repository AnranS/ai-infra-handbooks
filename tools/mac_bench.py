"""Apple 芯片推理实测：一条命令测出统一内存的带宽，以及几种模型的 prefill 和 decode 速度。

用法（Apple 芯片的 Mac，先跑过 env/setup-macos.sh）：
    uv pip install --python .venv/bin/python mlx mlx-lm        # 第一次用先装 MLX（setup 脚本没装它）
    .venv/bin/python tools/mac_bench.py                        # 带宽 + 默认的五个模型；第一次要下载约 45 GB 权重
    .venv/bin/python tools/mac_bench.py --quick                # 只测带宽和 Qwen3-0.6B，几分钟
    .venv/bin/python tools/mac_bench.py --models 8b 30b-a3b    # 只测某几个；也可以直接写 Hugging Face 仓库名或本地路径
    .venv/bin/python tools/mac_bench.py --list                 # 列出预设的模型
结果打印成表格，同时存到 build/mac_bench/<时间>.json，数字可以直接用在「Apple 芯片推理」一章里。

测的是什么：
- 带宽：在默认设备上反复读一个 1 GiB 的数组（求和）和读写一遍（逐元素乘），取中位数。decode 每生成一个 token
  都要把权重读一遍，所以"带宽 ÷ 每 token 读的字节数"就是 decode 速度的上限；
- 模型：每个模型在两个提示词长度下各生成一段，记录 prefill（处理提示词）和 decode（逐个生成）的 token/s 与峰值内存。
  生成时去掉结束符，保证每次都生成满长度；MoE 模型每 token 只读激活的那部分专家，上限按"激活参数 / 总参数"折算。
没有 Apple GPU 的机器（Linux 上装 mlx[cpu]）也能跑，测的是 CPU，只用来检查脚本本身。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "mac_bench"

# 预设：键 -> (仓库, 说明, 总参数量 B, 激活参数量 B)
PRESETS = {
    "0.6b": ("mlx-community/Qwen3-0.6B-bf16", "Qwen3-0.6B，bf16", 0.6, 0.6),
    "8b": ("mlx-community/Qwen3-8B-bf16", "Qwen3-8B，bf16", 8.2, 8.2),
    "8b-4bit": ("mlx-community/Qwen3-8B-4bit", "Qwen3-8B，4 bit", 8.2, 8.2),
    "30b-a3b": ("mlx-community/Qwen3-30B-A3B-4bit", "Qwen3-30B-A3B（MoE），4 bit", 30.5, 3.3),
    "32b": ("mlx-community/Qwen3-32B-4bit", "Qwen3-32B，4 bit", 32.8, 32.8),
    "32b-bf16": ("mlx-community/Qwen3-32B-bf16", "Qwen3-32B，bf16（约 65 GB，先用 sysctl 调高 iogpu.wired_limit_mb）", 32.8, 32.8),
}
DEFAULT = ["0.6b", "8b", "8b-4bit", "30b-a3b", "32b"]
TEXT = ("推理引擎每生成一个 token 都要把模型的权重从内存里读一遍，所以 decode 的速度主要由内存带宽决定；"
        "处理提示词时一次算很多个 token，矩阵乘法的算力才是瓶颈。 The decode phase is bound by memory bandwidth, "
        "while prefill is bound by compute. ")


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def env_info() -> dict:
    import mlx.core as mx
    import mlx_lm

    info = {
        "系统": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}（{platform.machine()}）",
        "芯片": sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "内存 GB": round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1) or None,
        "GPU 可锁定内存上限 MB（0 = 系统默认）": sh(["sysctl", "-n", "iogpu.wired_limit_mb"]) or None,
        "MLX": mx.__version__,
        "mlx-lm": mlx_lm.__version__,
        "设备": str(mx.default_device()),
        "Python": platform.python_version(),
    }
    if mx.metal.is_available():
        d = mx.metal.device_info()
        info["GPU 架构"] = d.get("architecture")
        info["建议的 GPU 工作集上限 GB"] = round(d.get("max_recommended_working_set_size", 0) / 2**30, 1)
    return info


def bandwidth(n_bytes: int = 1 << 30, repeats: int = 10) -> dict:
    """读：对 1 GiB 的 fp16 数组求和；读写：逐元素乘一个数（读一遍、写一遍）。各跑 repeats 次取中位数，单位 GB/s"""
    import mlx.core as mx

    x = mx.ones((n_bytes // 2,), dtype=mx.float16)
    mx.eval(x)

    def timed(fn):
        mx.eval(fn())                                   # 预热：编译 kernel、分配输出
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    read = n_bytes / timed(lambda: x.sum()) / 1e9
    copy = 2 * n_bytes / timed(lambda: x * 1.5) / 1e9
    return {"读 GB/s": round(read, 1), "读写 GB/s": round(copy, 1)}


def make_prompt(tokenizer, n: int) -> list[int]:
    ids = tokenizer.encode(TEXT)
    return (ids * (n // len(ids) + 1))[:n]


def bench_model(key: str, prompt_lens: list[int], gen: int, read_gbs: float) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate

    repo, desc, total_b, active_b = PRESETS.get(key, (key, key, None, None))
    t = time.perf_counter()
    model, tokenizer = load(repo)
    load_s = time.perf_counter() - t
    weight_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    active = active_b / total_b if total_b else 1.0     # MoE：每 token 只读激活的那部分（按参数量折算）
    bound = read_gbs * 1e9 / (weight_bytes * active)    # decode 速度的上限：带宽 ÷ 每 token 读的字节数（batch=1，忽略 KV）
    tokenizer.eos_token_ids = set()                     # 不在结束符处停下，保证每次都生成满 gen 个 token

    def run(n_prompt, n_gen):
        mx.reset_peak_memory()
        last = None
        for last in stream_generate(model, tokenizer, make_prompt(tokenizer, n_prompt), max_tokens=n_gen):
            pass
        return last

    run(32, 8)                                          # 预热：编译 kernel
    rows = []
    for n in prompt_lens:
        r = run(n, gen)
        rows.append({"提示词": n, "prefill tok/s": round(r.prompt_tps, 1), "decode tok/s": round(r.generation_tps, 1),
                     "生成": r.generation_tokens, "峰值内存 GB": round(r.peak_memory, 2)})
    del model
    mx.clear_cache()
    return {"模型": desc, "仓库": repo, "权重 GB": round(weight_bytes / 1e9, 2), "加载秒": round(load_s, 1),
            "decode 上限 tok/s": round(bound, 1), "结果": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", help="预设的键、Hugging Face 仓库名或本地路径；默认 " + " ".join(DEFAULT))
    ap.add_argument("--quick", action="store_true", help="只测带宽和 0.6b")
    ap.add_argument("--prompt-lens", nargs="*", type=int, default=[512, 4096], help="提示词长度，默认 512 4096")
    ap.add_argument("--gen", type=int, default=128, help="每次生成多少个 token，默认 128")
    ap.add_argument("--list", action="store_true", help="列出预设的模型")
    args = ap.parse_args()
    if args.list:
        for k, (repo, desc, total, active) in PRESETS.items():
            print(f"{k:<9} {repo:<36} {desc}（总参数 {total}B，激活 {active}B）{'  默认' if k in DEFAULT else ''}")
        return
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        sys.exit("没装 MLX：先运行 uv pip install --python .venv/bin/python mlx mlx-lm")

    report = {"时间": time.strftime("%Y-%m-%d %H:%M:%S"), "环境": env_info()}
    for k, v in report["环境"].items():
        print(f"{k}：{v}")
    report["带宽"] = bandwidth()
    print(f"带宽：读 {report['带宽']['读 GB/s']} GB/s，读写 {report['带宽']['读写 GB/s']} GB/s")
    if not mlx.core.metal.is_available():
        print("（没有 Apple GPU，测的是 CPU：MLX 的 CPU 后端做 fp16 运算很慢，带宽和上限都不准，只用来检查脚本能不能跑通）")

    models = ["0.6b"] if args.quick else (args.models or DEFAULT)
    report["模型"] = []
    for key in models:
        print(f"\n== {PRESETS.get(key, (key, key))[1]} ==", flush=True)
        try:
            res = bench_model(key, args.prompt_lens, args.gen, report["带宽"]["读 GB/s"])
        except Exception as e:                          # 一个模型失败（下载、内存不够）不影响其他模型
            print(f"失败：{type(e).__name__}: {e}")
            report["模型"].append({"模型": key, "错误": f"{type(e).__name__}: {e}"})
            continue
        report["模型"].append(res)
        print(f"权重 {res['权重 GB']} GB，加载 {res['加载秒']} 秒，decode 上限约 {res['decode 上限 tok/s']} tok/s")
        for r in res["结果"]:
            eff = r["decode tok/s"] / res["decode 上限 tok/s"] if res["decode 上限 tok/s"] else 0
            print(f"  提示词 {r['提示词']:>5}：prefill {r['prefill tok/s']:>8.1f} tok/s，decode {r['decode tok/s']:>6.1f} tok/s"
                  f"（上限的 {eff:.0%}），峰值内存 {r['峰值内存 GB']} GB")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / time.strftime("%Y%m%d-%H%M%S.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已存到 {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
