import os
import shutil
import subprocess
import sys

with open("data.bin", "wb") as f:                   # 准备一个 64 MiB 的文件
    f.write(os.urandom(64 << 20))

reader = """
import os, sys
fd = os.open("data.bin", os.O_RDONLY)
while os.read(fd, int(sys.argv[1])):
    pass
"""
if not shutil.which("strace"):
    print("没有 strace，跳过")
    raise SystemExit
for chunk in (4096, 1 << 20):
    # -c 汇总系统调用次数，-P 只统计涉及这个文件的调用
    out = subprocess.run(["strace", "-c", "-P", "data.bin", "-e", "trace=read", sys.executable, "-c", reader, str(chunk)],
                         capture_output=True, text=True).stderr
    calls = next(line.split()[3] for line in out.splitlines() if line.rstrip().endswith("read"))
    print(f"每次读 {chunk // 1024} KiB：{calls} 次 read 系统调用")
