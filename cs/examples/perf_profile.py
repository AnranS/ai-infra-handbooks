import re
import shutil
import subprocess

if not shutil.which("perf"):
    print("没有 perf，跳过（Ubuntu 上装 linux-tools-$(uname -r)）")
    raise SystemExit
# 按时间采样（cpu-clock 是软件事件，虚拟机里没有硬件计数器也能用），每秒 999 次
subprocess.run(["perf", "record", "-q", "-e", "cpu-clock", "-F", "999", "-g", "-o", "perf.data", "./hot"],
               check=True, capture_output=True)
report = subprocess.run(["perf", "report", "-i", "perf.data", "--stdio", "--no-children", "--sort", "symbol"],
                        capture_output=True, text=True).stdout
for name in ("hot_loop", "cold_loop"):
    m = re.search(rf"([\d.]+)%.*\b{name}\b", report)
    print(f"{name} 占了 {m.group(1)}% 的采样" if m else f"没有找到 {name}")
