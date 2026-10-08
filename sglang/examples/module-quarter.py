import collections, os, re, subprocess
REF = os.environ.get("REF", "29f6d408c0")
MODS = [("managers", r"^python/sglang/srt/managers/"), ("mem_cache", r"^python/sglang/srt/mem_cache/"), ("layers/attention", r"^python/sglang/srt/layers/attention"),
        ("speculative", r"^python/sglang/srt/speculative/"), ("disaggregation", r"^python/sglang/srt/disaggregation/"), ("sgl-kernel/kernels", r"^sgl-kernel/|^python/sglang/kernels/"),
        ("router/gateway", r"^rust/|^sgl-router/|^sgl-model-gateway/"), ("multimodal_gen", r"^python/sglang/multimodal_gen/")]
pats = [(n, re.compile(p)) for n, p in MODS]
table, cur, touched = collections.defaultdict(collections.Counter), None, set()
out = subprocess.run(["git", "log", "--date=short", "--format=@%ad", "--name-only", REF], capture_output=True, text=True).stdout
for line in out.splitlines():
    if line.startswith("@"):
        for m in touched: table[cur][m] += 1
        d = line[1:]; cur = f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"; touched = set()
    elif line.strip():
        for n, p in pats:
            if p.search(line): touched.add(n)
for m in touched: table[cur][m] += 1
qs = sorted(q for q in table if q >= "2024Q3")
print("模块 \\ 季度".ljust(20) + "".join(q.rjust(7) for q in qs))
for n, _ in MODS:
    print(n.ljust(20) + "".join(str(table[q][n] or "·").rjust(7) for q in qs))
