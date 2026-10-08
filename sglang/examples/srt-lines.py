import subprocess
REF0 = "22085081bb"
files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF0, "--", "python/sglang/srt"], capture_output=True, text=True).stdout.split()
rows = []
for f in files:
    if f.endswith(".py"):
        rows.append((subprocess.run(["git", "show", f"{REF0}:{f}"], capture_output=True, text=True).stdout.count("\n"), f.removeprefix("python/sglang/srt/")))
for n, f in sorted(rows, reverse=True):
    print(f"{n:5d}  {f}")
print(f"{sum(n for n, _ in rows):5d}  合计（{len(rows)} 个文件）")
