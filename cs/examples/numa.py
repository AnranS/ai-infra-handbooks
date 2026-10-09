import glob
import shutil
import subprocess

if not shutil.which("numactl"):
    print("没有 numactl，跳过（Ubuntu 上装 numactl）")
    raise SystemExit
for node in sorted(glob.glob("/sys/devices/system/node/node[0-9]*")):
    cpus = open(f"{node}/cpulist").read().strip()
    print(f"{node.rsplit('/', 1)[1]}：CPU {cpus}")

# 进程固定跑在节点 0 的 CPU 上，内存分别放在本地（节点 0）和远端（节点 1）
for mem in (0, 1):
    out = subprocess.run(["numactl", "--cpunodebind=0", f"--membind={mem}", "./numabw"],
                         capture_output=True, text=True, check=True).stdout.strip()
    print(f"CPU 在节点 0、内存在节点 {mem}（{'本地' if mem == 0 else '远端'}）：{out}")
