"""CUDA C++ 练习题的判题：有 NVIDIA GPU 和 nvcc 时编译到真卡上运行（并报告耗时、带宽），
否则用 CUDA 手册自带的 CPU 模拟器（cuda/tools/emu，g++ 或 clang++ 编译）只检查正确性。

题目目录里的 test.cu 以 `#include "user.cu"` 引入你的代码，输出 CASE / PERF / TIER 行（见 runtime/cuda/judge.cuh）。
TIER 行是性能档位：达到参照（本机实测带宽、同形状 cuBLAS）的比例，分铜、银、金三档，只在真 GPU 上给出。
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EMU_INCLUDE = REPO / "cuda" / "tools" / "emu" / "include"
BUILD = HERE / "workspace" / ".build"


def _emu_translate():
    spec = importlib.util.spec_from_file_location("emu_run", REPO / "cuda" / "tools" / "emu_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.translate


def nvcc_path():
    return os.environ.get("PRACTICE_NVCC") or shutil.which("nvcc") or \
        ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)


TIERS = ("金", "银", "铜")


def tier_label(ratio: float, bronze: float, silver: float, gold: float) -> str:
    for name, bar in zip(TIERS, (gold, silver, bronze)):
        if ratio >= bar:
            return name
    return "未达铜档"


def parse(output: str) -> dict:
    cases, perf, tiers = [], {}, {}
    for line in output.splitlines():
        if line.startswith("TIER "):
            vals = line.split(" ", 6)
            if len(vals) >= 6:
                ratio, bronze, silver, gold = map(float, vals[2:6])
                tiers[vals[1]] = {"ratio": ratio, "bronze": bronze, "silver": silver, "gold": gold,
                                  "ref": vals[6] if len(vals) > 6 else "参照", "label": tier_label(ratio, bronze, silver, gold)}
            continue
        parts = line.split(" ", 3)
        if parts[0] == "CASE" and len(parts) >= 3:
            status = {"PASS": "pass", "FAIL": "fail", "ERROR": "error"}.get(parts[2], "error")
            cases.append({"name": parts[1], "doc": "", "status": status, "message": parts[3] if len(parts) > 3 else "",
                          "stdout": "", "ms": 0.0})
        elif parts[0] == "PERF" and len(parts) >= 3:
            vals = line.split()
            perf[vals[1]] = vals[2:]
    for c in cases:
        if c["name"] in perf:
            ms, gbps, tflops = (perf[c["name"]] + ["0", "0", "0"])[:3]
            c["ms"] = float(ms)
            extra = f"GPU 耗时 {float(ms):.3f} ms，带宽 {float(gbps):.1f} GB/s" + \
                (f"，{float(tflops):.2f} TFLOPS" if float(tflops) > 0 else "")
            c["message"] = (c["message"] + "\n" if c["message"] else "") + extra
        if c["name"] in tiers:
            t = tiers[c["name"]]
            c["tier"] = t
            sep = " " if t["ref"][-1:].isascii() else ""
            c["message"] += (f"\n性能档位：{t['label']}（达到{t['ref']}{sep}的 {t['ratio']:.0%}；"
                             f"铜 ≥ {t['bronze']:.0%}、银 ≥ {t['silver']:.0%}、金 ≥ {t['gold']:.0%}）")
    return {"cases": cases, "perf": perf, "tiers": tiers}


def judge(p, code: str, info: dict | None = None, force_emulator: bool = False) -> dict:
    if not p.cuda:
        return {"status": "compile_error", "cases": [], "passed": 0, "total": 0,
                "error": "这道题没有 CUDA C++ 版本，请提交 .py 文件"}
    info = info or {}
    work = BUILD / p.slug
    work.mkdir(parents=True, exist_ok=True)
    nvcc = nvcc_path()
    real = not force_emulator and info.get("device") == "cuda" and nvcc
    exe = work / ("test_gpu" if real else "test_emu")
    if real:
        (work / "user.cu").write_text(code, encoding="utf-8")
        (work / "test.cu").write_text(p.cuda["tests"], encoding="utf-8")
        cmd = [nvcc, "-O2", "-std=c++17", "-arch=native", "-lineinfo", f"-I{HERE / 'runtime' / 'cuda'}", f"-I{work}",
               str(work / "test.cu"), "-o", str(exe)]
        if "judge_cublas.cuh" in p.cuda["tests"]:
            cmd.append("-lcublas")
        mode = "nvcc（真 GPU）"
    else:
        cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            return {"status": "compile_error", "cases": [], "passed": 0, "total": 0,
                    "error": "没有找到 nvcc（NVIDIA GPU）也没有找到 g++ / clang++（CPU 模拟器），无法判题"}
        src = p.cuda["tests"].replace('#include "user.cu"', "// ---- 你的代码 ----\n" + code + "\n// ---- 测试 ----")
        src = _emu_translate()(p.slug + ".cu", src)
        (work / "emu.cpp").write_text(src, encoding="utf-8")
        cmd = [cxx, "-std=c++17", "-O1", "-w", "-DPRACTICE_EMU", f"-I{EMU_INCLUDE}", f"-I{HERE / 'runtime' / 'cuda'}",
               str(work / "emu.cpp"), "-o", str(exe)]
        if platform.system() == "Darwin":
            cmd[1:1] = ["-D_XOPEN_SOURCE=600", "-D_DARWIN_C_SOURCE"]
        mode = "CPU 模拟器（只检查正确性）"
    c = subprocess.run(cmd, capture_output=True, text=True)
    if c.returncode:
        return {"status": "compile_error", "cases": [], "passed": 0, "total": 0,
                "error": f"编译失败（{mode}）：\n" + c.stderr[-4000:]}
    try:
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=300, cwd=work)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cases": [], "passed": 0, "total": 0, "error": "运行超过 300 秒（死循环或死锁？）"}
    out = r.stdout + r.stderr
    res = parse(out)
    cases = res["cases"]
    passed = sum(x["status"] == "pass" for x in cases)
    status = "accepted" if cases and passed == len(cases) and r.returncode == 0 else \
        ("runtime_error" if r.returncode != 0 and not any(x["status"] == "fail" for x in cases) else "wrong_answer")
    extra = [l for l in out.splitlines() if not l.startswith(("CASE ", "PERF "))]
    err = None
    if r.returncode != 0:
        err = f"程序异常退出（返回码 {r.returncode}）" + ("\n" + "\n".join(extra[-30:]) if extra else "")
        if "DEADLOCK" in out:
            err += "\n模拟器检测到死锁：__syncthreads() 没有被 block 内所有线程执行？"
    note = f"判题方式：{mode}"
    if res["tiers"]:
        note += "；性能档位：" + "、".join(f"{k} {v['label']}" for k, v in res["tiers"].items())
    elif not real:
        note += "；性能档位只在 NVIDIA GPU 上评定"
    return {"status": status, "cases": cases, "passed": passed, "total": len(cases), "error": err,
            "note": note, "stdout": "\n".join(extra[-20:]), "tiers": res["tiers"]}
