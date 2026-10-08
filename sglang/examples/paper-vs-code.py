import collections, subprocess
REF0 = "22085081bb"
files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF0, "--", "python/sglang"], capture_output=True, text=True).stdout.split()
GROUPS = [("lang/ 前端语言", "python/sglang/lang/"), ("backend/ 外部后端", "python/sglang/backend/"),
          ("srt/managers/router/ 调度与执行", "python/sglang/srt/managers/router/"), ("srt/managers/ 分词与进程", "python/sglang/srt/managers/"),
          ("srt/constrained/ 约束解码", "python/sglang/srt/constrained/"), ("srt/layers/ 注意力算子", "python/sglang/srt/layers/"),
          ("srt/models/ 模型", "python/sglang/srt/models/"), ("srt/ 其他（内存池、参数、服务）", "python/sglang/srt/"), ("顶层（api、test、utils）", "python/sglang/")]
count = collections.Counter(); lines = collections.Counter()
for f in files:
    if not f.endswith(".py"):
        continue
    n = subprocess.run(["git", "show", f"{REF0}:{f}"], capture_output=True, text=True).stdout.count("\n")
    group = next(name for name, prefix in GROUPS if f.startswith(prefix))
    count[group] += 1; lines[group] += n
print(f"{'模块':<34}{'文件':>4}{'行数':>7}")
for name, _ in GROUPS:
    print(f"{name:<34}{count[name]:>4}{lines[name]:>7}")
print(f"{'合计':<34}{sum(count.values()):>4}{sum(lines.values()):>7}")
