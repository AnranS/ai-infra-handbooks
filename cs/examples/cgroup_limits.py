import os


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


v2 = os.path.exists("/sys/fs/cgroup/cgroup.controllers")
print("cgroup 版本：", "v2" if v2 else "v1")
if v2:
    rel = open("/proc/self/cgroup").read().strip().split("::")[-1]
    base = "/sys/fs/cgroup" + rel
    quota, period = (read(f"{base}/cpu.max") or "max 100000").split()
    mem = read(f"{base}/memory.max")
else:
    quota, period = read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us"), read("/sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us")
    quota = "max" if quota in (None, "-1") else quota
    mem = read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
limit = None if quota == "max" else int(quota) / int(period)
print("CPU 配额：", "不限" if limit is None else f"{limit:g} 个 CPU")
print("内存上限：", "不限" if mem in (None, "max") or int(mem) > 1 << 60 else f"{int(mem) / 2**30:.1f} GiB")

affinity = len(os.sched_getaffinity(0))
usable = affinity if limit is None else min(affinity, limit)
print(f"os.cpu_count() = {os.cpu_count()}，可调度的 CPU = {affinity}，按配额真正能用的约 {usable:g} 个")

st = os.statvfs("/dev/shm")
print(f"/dev/shm 大小：{st.f_blocks * st.f_frsize / 2**30:.1f} GiB")
