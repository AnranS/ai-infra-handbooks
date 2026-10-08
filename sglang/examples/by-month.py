import collections, os, subprocess
REF = os.environ.get("REF", "29f6d408c0")
dates = subprocess.run(["git", "log", "--date=short", "--format=%ad", REF], capture_output=True, text=True).stdout.split()
by_month = collections.Counter(d[:7] for d in dates)
by_year = collections.Counter(d[:4] for d in dates)
print("按年：", dict(sorted(by_year.items())))
peak = max(by_month.items(), key=lambda kv: kv[1])
print(f"按月：共 {len(by_month)} 个月，最多的是 {peak[0]}（{peak[1]} 个）")
for m in ("2024-01", "2024-07", "2025-01", "2025-07", "2026-01", "2026-07"):
    print(f"  {m}: {by_month.get(m, 0):5d} {'#' * (by_month.get(m, 0) // 40)}")
