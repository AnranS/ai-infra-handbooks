"""Run the extracted CUDA examples on the CPU emulator (tools/emu) to check kernel logic.

Run tools/check_code.py first (it extracts the sources into build/examples).
Each program is source-translated (<<<>>> launches, __shared__, time_ms), compiled with g++
against the emulator headers, shrunk to small problem sizes and executed. A program passes when
it exits with 0, prints no FAIL and the emulator detects no deadlock.

Usage: python tools/emu_run.py [name.cu ...]      examples from the book (extracted from docs/ on first use)
       python tools/emu_run.py path/to/mine.cu      any .cu file of your own
On macOS (no NVIDIA GPU) this is the way to run the book's kernels: it checks correctness, not speed.
The compiler is $CXX, else g++ (Linux) or clang++ (macOS).
"""

import os
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "build" / "examples"
MACOS = platform.system() == "Darwin"
CXX = os.environ.get("CXX") or (shutil.which("clang++") if MACOS else shutil.which("g++")) or shutil.which("g++") or "c++"
# macOS 上 <ucontext.h> 只有在定义了 _XOPEN_SOURCE 时才能用（虽然已标为过时，但仍然可用）
PLATFORM_FLAGS = ["-D_XOPEN_SOURCE=600", "-D_DARWIN_C_SOURCE"] if MACOS else []
OUT = ROOT / "build" / "emu"
UNSUPPORTED_HEADERS = ("cuda_bf16.h", "cuda_fp8.h", "cooperative_groups", "cuda/pipeline",
                       "cuda/barrier", "nccl.h", "cuda.h", "cublas_v2.h", "cute/", "torch/extension.h")
UNSUPPORTED_CODE = ("asm(", "asm volatile", "cudaLaunchCooperativeKernel", "__cluster_dims__",
                    "__nv_bfloat16", "__halves2half2")

# Shrink problem sizes so the (very slow) emulator finishes quickly: file -> [(old, new), ...]
SHRINK = {
    "vector_add.cu": [("1 << 24", "1 << 12")],
    "grid_stride.cu": [("1 << 25", "1 << 12")],
    "matrix_add.cu": [("rows = 3000, cols = 5000", "rows = 70, cols = 90")],
    "access_pattern.cu": [("1 << 22", "1 << 10")],
    "bank_conflict.cu": [("iters = 4096", "iters = 16")],
    "conv1d.cu": [("(1 << 22) + 123", "(1 << 11) + 123")],
    "divergence.cu": [("iters = 2000", "iters = 20")],
    "block_sum.cu": [("50'000'000", "100'000")],
    "histogram.cu": [("1 << 26", "1 << 14")],
    "atomic_max_float.cu": [("1 << 22", "1 << 12")],
    "reduction.cu": [("(1 << 26) + 3", "(1 << 14) + 3")],
    "argmax.cu": [("10'000'019", "100'019"), ("7'654'321", "76'543")],
    "transpose.cu": [("4096 + 17", "64 + 17"), ("8192 + 5", "96 + 5")],
    "gemm.cu": [("check_n = 512", "check_n = 256"), (": 4096;", ": 128;")],
    "softmax.cu": [("{{4096, 1000}, {64, 50000}}", "{{16, 1000}, {3, 5000}}")],
    "rmsnorm.cu": [("tokens = 512", "tokens = 8")],
    "layernorm.cu": [("rows = 256", "rows = 6")],
    "scan.cu": [("10'000'000 + 7", "300'000 + 7")],
    "compact.cu": [("1 << 22", "1 << 14")],
    "wmma_gemm.cu": [(": 2048, N = M", ": 128, N = M"), ("check_rows = 64", "check_rows = 128")],
    "gemm_cp_async.cu": [(": 4096;", ": 128;")],
    "flash_attn.cu": [("batch_heads = 8, N = 1000", "batch_heads = 2, N = 150")],
    "paged_decode.cu": [("num_seqs = 4, num_heads = 32, num_kv_heads = 8", "num_seqs = 4, num_heads = 8, num_kv_heads = 2"),
                        ("{1, 17, 1000, 4096}", "{1, 17, 100, 300}")],
    "gemv_w4.cu": [("N = 4096, K = 4096", "N = 64, K = 512")],
    "nvtx_demo.cu": [("1 << 24", "1 << 12")],
    "overlap.cu": [("1 << 26", "1 << 14")],
    "cuda_graph.cu": [("n = 4096, kernels_per_step = 200, steps = 20", "n = 512, kernels_per_step = 20, steps = 3")],
    "managed_prefetch.cu": [("1 << 24", "1 << 12")],
    "pdl_chain.cu": [("n = 1 << 14, layers = 200, iters = 20", "n = 1000, layers = 6, iters = 2")],
}


def split_top_level(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def translate_launches(src):
    out, pos = [], 0
    pattern = re.compile(r"([A-Za-z_][\w:.]*(?:<[^;<>]*>)?)\s*<<<")
    while True:
        m = pattern.search(src, pos)
        if not m:
            out.append(src[pos:])
            break
        cfg_end = src.index(">>>", m.end())
        cfg = src[m.end():cfg_end]
        i = src.index("(", cfg_end)
        depth, j = 0, i
        while True:
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        args = src[i + 1:j]
        parts = split_top_level(cfg)
        call = f"::emu::launch({', '.join(parts)}, [&]() {{ {m.group(1)}({args}); }})"
        out.append(src[pos:m.start()] + call)
        pos = j + 1
    return "".join(out)


def translate(name, src):
    for old, new in SHRINK.get(name, []):
        if old not in src:
            raise SystemExit(f"{name}: shrink pattern {old!r} not found")
        src = src.replace(old, new)
    src = re.sub(r"extern\s+__shared__\s+(?:__align__\(\d+\)\s+)?([\w:]+(?:\s+[\w:]+)*)\s+(\w+)\[\];",
                 r"\1* \2 = reinterpret_cast<\1*>(::emu::dyn_smem());", src)
    src = re.sub(r"\b__shared__\b", "static", src)
    src = re.sub(r"\btime_ms\(", "emu_time_ms(", src)
    src = re.sub(r"(\.|->)(gridDim|blockDim)\b", r"\1emu_\2", src)   # cudaLaunchConfig_t 的成员（gridDim 在模拟器里是宏）
    return translate_launches(src)


def run_one(path: Path):
    name = path.name
    src = path.read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*", "", src)                      # ignore comments
    code = re.sub(r"#ifdef WITH_CUBLAS.*?#endif", "", code, flags=re.S)
    includes = re.findall(r"#include\s*[<\"]([^>\"]+)", code)
    if any(h in inc for inc in includes for h in UNSUPPORTED_HEADERS) or any(t in code for t in UNSUPPORTED_CODE):
        return name, "skip", "uses features the emulator does not model"
    OUT.mkdir(parents=True, exist_ok=True)
    cpp = OUT / (path.stem + ".cpp")
    cpp.write_text(translate(name, src), encoding="utf-8")
    exe = OUT / path.stem
    c = subprocess.run([CXX, "-std=c++17", "-O1", "-w", *PLATFORM_FLAGS, f"-I{ROOT / 'tools/emu/include'}", f"-I{SRC}",
                        f"-I{path.parent}", str(cpp), "-o", str(exe)], capture_output=True, text=True)
    if c.returncode:
        return name, "FAIL", "compile error:\n" + c.stderr[-2500:]
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600, cwd=OUT)
    except subprocess.TimeoutExpired:
        return name, "FAIL", "timeout"
    text = (r.stdout + r.stderr).strip()
    bad = r.returncode != 0 or "FAIL" in r.stdout or "DEADLOCK" in text
    return name, "FAIL" if bad else "ok", text[-1500:]


def extract():
    """没有 nvcc 时 check_code.py 跑不了：直接从正文里取出 .cu / .cuh 文件"""
    from check_code import blocks
    SRC.mkdir(parents=True, exist_ok=True)
    for md in sorted((ROOT / "docs").rglob("*.md")):
        for name, _, code in blocks(md):
            if name.endswith((".cu", ".cuh")):
                (SRC / name).write_text(code, encoding="utf-8")


def main(argv):
    if not SRC.exists() or not any(SRC.glob("*.cu")):
        extract()
    files = [Path(a).resolve() if Path(a).exists() else SRC / a for a in argv] if argv else sorted(SRC.glob("*.cu"))
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for name, status, text in pool.map(run_one, files):
            print(f"[{status}] emu {name}")
            if status != "ok":
                print("    " + text.replace("\n", "\n    "))
            failed += status == "FAIL"
    print(f"\n{len(files)} programs, {failed} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
