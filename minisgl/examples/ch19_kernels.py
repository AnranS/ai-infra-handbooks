"""第 19 章：用两个版本的 nvcc 编译 kernel，并在 CPU 模拟器上运行自检。"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "tools")
import emu_kernels  # noqa: E402

csrc = Path("python/minisgl/kernel/csrc")
for nvcc in sorted(Path.home().glob("cuda-handbook/.toolkit*/bin/nvcc")):
    version = re.search(r"release ([\d.]+)", subprocess.run([str(nvcc), "--version"], capture_output=True,
                                                            text=True).stdout).group(1)
    r = subprocess.run([str(nvcc), "-std=c++17", "-O2", "-Wno-deprecated-gpu-targets", f"-I{csrc}", "-c",
                        "tests/cuda/test_kv_kernels.cu", "-o", "/dev/null"], capture_output=True, text=True)
    print(f"nvcc {version}: 编译 {'通过' if r.returncode == 0 else '失败'}")
print("CPU 模拟器上运行自检：")
ok = emu_kernels.run(Path("tests/cuda/test_kv_kernels.cu"))
print("全部通过" if ok else "有失败")
