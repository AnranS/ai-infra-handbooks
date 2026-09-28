"""在 CUDA 手册的 CPU 模拟器上运行 tests/cuda/*.cu 的自检程序（没有 GPU 也能验证 kernel 逻辑）。"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMU = ROOT.parent / "cuda" / "tools"
sys.path.insert(0, str(EMU))
import emu_run  # noqa: E402  复用 CUDA 手册的源码翻译（<<<>>> 启动、__shared__ 等）


def run(cu: Path) -> bool:
    src = emu_run.translate(cu.name, cu.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        cpp, exe = Path(tmp) / (cu.stem + ".cpp"), Path(tmp) / cu.stem
        cpp.write_text(src, encoding="utf-8")
        c = subprocess.run(["g++", "-std=c++17", "-O1", "-w", f"-I{EMU / 'emu' / 'include'}",
                            f"-I{ROOT / 'python/minisgl/kernel/csrc'}", str(cpp), "-o", str(exe)],
                           capture_output=True, text=True)
        if c.returncode:
            print(c.stderr[-2000:])
            return False
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
        print(r.stdout.strip())
        return r.returncode == 0 and "FAIL" not in r.stdout


if __name__ == "__main__":
    files = sorted((ROOT / "tests" / "cuda").glob("*.cu"))
    ok = all([run(f) for f in files])
    print("emulator:", "all passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)
