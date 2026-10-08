# 取模分片：加一台机器，多少个键要换地方？
import hashlib


def h(key):
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")


keys = [f"session-{i}" for i in range(100000)]
print("机器数变化    需要迁移的键")
for n in (4, 8, 16, 32):
    moved = sum(1 for k in keys if h(k) % n != h(k) % (n + 1))
    print(f"{n:3d} -> {n + 1:3d}    {moved / len(keys):6.1%}")
print("\n理想情况：加一台机器只需要迁移 1/(n+1)：", ", ".join(f"{1 / (n + 1):.1%}" for n in (4, 8, 16, 32)))
