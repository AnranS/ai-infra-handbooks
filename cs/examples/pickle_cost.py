import array
import pickle
import time

# 一步调度的结果：256 个请求，每个请求这一步要算的 token（大部分在 decode，只有 1 个 token）
batch = [{"req_id": f"req-{i}", "token_ids": [1000 + i] if i % 16 else list(range(512)), "block_ids": list(range(i, i + 8))}
         for i in range(256)]

t = time.perf_counter()
for _ in range(100):
    blob = pickle.dumps(batch)
    pickle.loads(blob)
pk = (time.perf_counter() - t) / 100

# 同样的信息压成扁平的整数数组：每个请求的 token 数 + 所有 token + 块号
flat = array.array("i")
for r in batch:
    flat.append(len(r["token_ids"]))
    flat.extend(r["token_ids"])
    flat.extend(r["block_ids"])
t = time.perf_counter()
for _ in range(100):
    raw = flat.tobytes()
    array.array("i").frombytes(raw)
fl = (time.perf_counter() - t) / 100
print(f"pickle：{len(blob) / 1024:.0f} KiB，序列化 + 反序列化 {pk * 1e6:.0f} µs")
print(f"扁平数组：{len(raw) / 1024:.0f} KiB，{fl * 1e6:.0f} µs")
