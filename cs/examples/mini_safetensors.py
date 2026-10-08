import json
import mmap
import struct

import numpy as np


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


# 写一个 safetensors 格式的文件：8 字节（小端）的头长度 + JSON 头 + 所有张量的原始字节
tensors = {f"layers.{i}.weight": np.full((1024, 1024), i, dtype=np.float16) for i in range(16)}   # 16 个 2 MiB 的张量
header, off = {}, 0
for name, t in tensors.items():
    header[name] = {"dtype": "F16", "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
    off += t.nbytes
raw = json.dumps(header).encode()
with open("model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(raw)) + raw)
    for t in tensors.values():
        f.write(t.tobytes())

# 读：mmap 整个文件，按头里记录的偏移直接"看"某个张量，不拷贝
with open("model.safetensors", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
n = struct.unpack("<Q", mm[:8])[0]
meta = json.loads(mm[8:8 + n])


def load(name):
    m = meta[name]
    start, end = m["data_offsets"]
    return np.frombuffer(mm, dtype=np.float16, count=(end - start) // 2, offset=8 + n + start).reshape(m["shape"])


before = rss_mib()
w = load("layers.7.weight")
print("文件里有", len(meta), "个张量，layers.7.weight 的形状", w.shape, "，值", float(w[0, 0]))
print("数组直接指向 mmap 的内存，没有拷贝：", not w.flags.owndata)
total = float(w.sum(dtype=np.float32))              # 真正读这 2 MiB：缺页把对应的文件页映射进来
print("32 MiB 的文件只读了 2 MiB，RSS 增加不到 3 MiB：", rss_mib() - before < 3)
