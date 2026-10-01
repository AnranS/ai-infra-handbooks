# 文件与 I/O：一次写入如何落盘

<p class="lead">"描述一次文件写入从调用开始到落盘的全过程"是系统岗面试的经典题，它一路穿过系统调用、文件系统、页缓存、块设备层、驱动和盘本身。推理系统里到处是这条路径的影子：模型加载第二次为什么快得多，从网络存储 mmap 权重为什么慢，KV Cache 落到 SSD 上要不要 fsync，一个日志写入为什么偶尔卡住几百毫秒。这一章沿着一次 write 和一次 read 走一遍，再讲 mmap（safetensors 为什么适合它）、O_DIRECT，以及 vLLM 加载权重时的几种策略。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `write()` 返回时，数据在哪里？这时断电会丢吗？
    2. `fsync()` 做了什么？为什么"每次写一点、每次都 fsync"很慢？
    3. 同一个模型第二次加载为什么快得多？怎么让这个效果失效？
    4. 用 mmap 读文件和用 read 读文件有什么区别？safetensors 为什么适合 mmap？
    5. O_DIRECT 是什么？什么场景会用它？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 在内核的页缓存里：内核把数据拷进对应文件页、标记为脏页就返回了，写回磁盘由后台线程稍后完成。这时断电或内核崩溃，数据会丢（进程自己崩溃不会丢，数据已经在内核里了）。
    2. fsync 把这个文件所有的脏页写回磁盘，提交文件系统日志里的元数据，再让盘把自己的写缓存刷到持久介质（FLUSH / FUA），全部完成才返回。每次都要等一次盘的往返，小写入的代价被固定开销主导，本机约 0.8 ms 一次，每秒一千多次；把多次写入攒在一起再 fsync（组提交）才能提高吞吐。
    3. 第一次加载时文件内容被读进了页缓存，只要内存没被别的用途挤掉，第二次读直接命中页缓存，不用碰盘，本机快了几十倍。让它失效：机器重启、内存紧张时被回收、`posix_fadvise(DONTNEED)` 丢掉某个文件的缓存、`echo 3 > /proc/sys/vm/drop_caches`（需要 root）。
    4. read 把数据从页缓存拷进用户缓冲区；mmap 把页缓存里的页直接映射进进程的地址空间，访问时靠缺页按需读入，不拷贝，多个进程映射同一个文件共享同一份物理页。safetensors 的布局是"头 + 各张量连续的原始字节"，头里记着每个张量的偏移，mmap 之后可以零拷贝地直接得到任意一个张量，只读用到的部分。
    5. 打开文件时加 O_DIRECT，读写绕过页缓存，直接在用户缓冲区和盘之间 DMA；要求缓冲区地址、长度和文件偏移都按块大小对齐。适合自己管理缓存的系统（数据库、KV 存储、把 KV Cache 放到 SSD 上的缓存层），避免数据在页缓存里多存一份、挤掉别的缓存，也让延迟更可预测。

## 一次 write 的旅程

![图：一次 write 的旅程——用户缓冲区、页缓存、块层、设备；O_DIRECT 与 mmap](../assets/figures/write-path.svg){.aig-svg}

调用 `write(fd, buf, n)` 之后：

1. **系统调用**：从用户态陷入内核，参数检查；
2. **VFS**：通过文件描述符找到打开的文件，交给具体的文件系统（ext4、xfs……）；
3. **文件系统**：必要时为文件分配磁盘块（延迟分配的文件系统会推迟到写回时），更新 inode 里的大小和时间；
4. **页缓存**：把数据从用户缓冲区拷进文件对应的页（页不在缓存里就新分配一个，写不满一页时还要先读出原内容），标记为**脏页**，`write` 就此返回；
5. **写回**：后台的写回线程（flusher）按时间（脏了超过 `dirty_expire_centisecs`，默认 30 秒）和数量（超过 `dirty_background_ratio`）把脏页写到盘上。脏页太多时（超过 `dirty_ratio`），写入的进程自己会被拖住、被迫参与写回——这就是"写日志偶尔卡住几百毫秒"的常见原因；
6. **块设备层**：写回的页被组织成 bio 请求，经过 I/O 调度器合并、排序，交给驱动；
7. **驱动与设备**：NVMe 驱动把命令放进提交队列、敲一下门铃寄存器，盘上的控制器通过 DMA 读走数据，写进自己的缓存，完成后通过完成队列和中断通知内核；
8. **真正持久化**：盘自己也有写缓存，只有收到 FLUSH 命令或者带 FUA（强制写入介质）标志的写，数据才保证落到闪存或磁片上。

**`fsync` 做的是把第 5～8 步立刻、完整地做一遍**：写回这个文件的所有脏页，提交文件系统日志（元数据），再发 FLUSH 让盘把缓存刷到介质，全部完成才返回。

```python title="write_fsync.py"
import os
import time

MiB = 1 << 20


def dirty_mib():                                   # 整个系统里还没写回磁盘的脏页
    for line in open("/proc/meminfo"):
        if line.startswith("Dirty:"):
            return int(line.split()[1]) / 1024


data = os.urandom(MiB)
with open("big.bin", "wb") as f:
    t0 = time.perf_counter()
    for _ in range(256):
        f.write(data)
    f.flush()                                      # 从 Python 的缓冲区交给内核：write 系统调用返回
    t1 = time.perf_counter()
    print(f"write 256 MiB：{(t1 - t0) * 1e3:.0f} ms（数据在页缓存里，脏页约 {dirty_mib():.0f} MiB）")
    os.fsync(f.fileno())                           # 等数据和元数据真正写到盘上
    t2 = time.perf_counter()
    print(f"fsync：{(t2 - t1) * 1e3:.0f} ms（之后脏页约 {dirty_mib():.0f} MiB）")

# 像数据库的预写日志那样：每次写 4 KiB 就 fsync 一次
block = os.urandom(4096)
with open("wal.log", "wb") as f:
    t0 = time.perf_counter()
    for _ in range(200):
        f.write(block)
        f.flush()
        os.fsync(f.fileno())
    per = (time.perf_counter() - t0) / 200
print(f"4 KiB 写入 + fsync：每次 {per * 1e3:.2f} ms，每秒最多约 {1 / per:,.0f} 次")
```

```text title="输出（本机示例）"
write 256 MiB：130 ms（数据在页缓存里，脏页约 256 MiB）
fsync：974 ms（之后脏页约 0 MiB）
4 KiB 写入 + fsync：每次 0.82 ms，每秒最多约 1,218 次
```

256 MiB 的 `write` 只是内存拷贝，快；`fsync` 要等盘真正写完，慢得多。最后一段像数据库的预写日志那样每写 4 KiB 就 fsync 一次：每次都要付一次"写回 + 日志提交 + 刷盘缓存"的往返，吞吐被这个固定开销卡死。数据库的对策是**组提交**：把同一时刻多个事务的日志攒在一起，一次 fsync。

## 读与页缓存

读的路径是反过来的：`read` 先查页缓存，命中就直接拷给用户；不命中就发起磁盘读，读完填进页缓存再拷贝。内核还会做**预读**（readahead）：发现是顺序读，就提前把后面的页读进来。

```python title="cold_warm.py"
import os
import time

fd = os.open("big.bin", os.O_RDONLY)               # 上一个脚本写好并 fsync 过的 256 MiB 文件
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # 把这个文件的页从页缓存里丢掉（只能丢干净页，所以前面要 fsync）


def read_all():
    os.lseek(fd, 0, os.SEEK_SET)
    t = time.perf_counter()
    while os.read(fd, 8 << 20):
        pass
    return time.perf_counter() - t


cold, warm = read_all(), read_all()
print(f"第一次读（从盘上读）：{256 / cold:,.0f} MiB/s")
print(f"第二次读（页缓存命中）：{256 / warm:,.0f} MiB/s，快 {cold / warm:.0f} 倍")
```

```text title="输出（本机示例）"
第一次读（从盘上读）：143 MiB/s
第二次读（页缓存命中）：4,731 MiB/s，快 33 倍
```

页缓存是"空闲"内存的主要用途：`free` 里的 buff/cache 就是它，内存紧张时内核会回收干净的缓存页。它对推理的影响：

- **模型第二次加载快得多**：权重文件还在页缓存里，加载时间只剩反序列化和拷到 GPU；机器重启或者内存被别的进程用掉以后又变回慢的那次；
- **多个进程共享**：张量并行的多个 worker 读同一个文件，页缓存里只有一份；
- **容器里计入内存用量**：页缓存算在 cgroup 的内存用量里，看起来"内存快满了"未必是泄漏，见[容器](containers.md)一章。

## mmap：把文件映射进地址空间

`mmap` 把文件的一段映射进进程的虚拟地址空间，访问某个地址时，如果对应的文件页不在页缓存里就触发主缺页、读盘，在的话就是次缺页、直接映射（缺页的两种类型见[虚拟内存](virtual-memory.md)）。和 `read` 相比：不用把数据拷进用户缓冲区；只读真正访问到的部分；多个进程映射同一个文件时共享同一份物理页。

权重文件的 safetensors 格式就是为这种用法设计的：文件开头 8 个字节是头的长度，接着是一个 JSON 头，记录每个张量的数据类型、形状和在数据区里的偏移，后面是所有张量连续存放的原始字节。mmap 之后，任意一个张量都可以零拷贝地"看"出来：

```python title="mini_safetensors.py"
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
```

```text title="输出"
文件里有 16 个张量，layers.7.weight 的形状 (1024, 1024) ，值 7.0
数组直接指向 mmap 的内存，没有拷贝： True
32 MiB 的文件只读了 2 MiB，RSS 增加不到 3 MiB： True
```

对比 PyTorch 的 `.bin`（pickle）格式：必须整个反序列化，反序列化时还可能执行任意代码。safetensors 既安全又能按需读取，所以张量并行的每个 rank 只需要读自己那一片。

mmap 的代价在于按需读取是一页一页地缺页：在本地 NVMe 上有预读帮忙，问题不大；在网络文件系统上，每次缺页都是一次网络往返，随机读又打乱了预读，加载会慢得离谱。vLLM 的 `safetensors_load_strategy`（`vllm/config/load.py`）就是在处理这个问题：

- 默认（不设置）：mmap 按需读取；检测到 NFS、并且整个检查点不超过可用内存的 90% 时，自动先把文件预取进页缓存；
- `lazy`：只用 mmap，不自动预取，适合本地盘；
- `eager`：先把整个文件读进 CPU 内存再加载，推荐用于 Lustre、NFS 这类网络文件系统，用顺序大块读代替随机的小读；
- `prefetch`：加载前用多个线程（`safetensors_prefetch_num_threads`）把文件顺序读进页缓存，之后的 mmap 就都命中缓存。

`load_format` 还提供了其他加载器：`runai_streamer` 用 Run:ai 的 Model Streamer 并发地从对象存储或文件流式读取，`sharded_state` 直接加载按张量并行预先切好的检查点（每个 rank 只读自己的文件），`tensorizer` 用 CoreWeave 的 tensorizer 格式，`ipc_cache` 通过 CUDA IPC 映射本机权重缓存守护进程里已经量化好的权重，让引擎重启几乎不用重新加载。

## O_DIRECT：绕过页缓存

打开文件时加 `O_DIRECT`，读写就绕过页缓存，在用户缓冲区和盘之间直接 DMA。条件是对齐：缓冲区地址、长度、文件偏移都要是逻辑块大小（通常 512 字节或 4 KiB）的整数倍：

```python title="odirect.py" ci="no"
import mmap
import os

fd = os.open("direct.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
try:
    os.write(fd, b"x" * 1000)                       # 长度不是块大小的整数倍，缓冲区也没有对齐
except OSError as e:
    print("不对齐的 O_DIRECT 写入失败：", e.strerror)
buf = mmap.mmap(-1, 4096)                          # mmap 分配的内存按页对齐
buf.write(b"y" * 4096)
print("对齐的写入成功：", os.write(fd, buf), "字节")
os.close(fd)
```

```text title="输出"
不对齐的 O_DIRECT 写入失败： Invalid argument
对齐的写入成功： 4096 字节
```

谁会用它：自己管理缓存的系统。数据库和 KV 存储有自己的缓存池，数据再在页缓存里存一份是浪费，还会把别的有用缓存挤出去；把 KV Cache 放到本地 SSD 的缓存层也常用 `O_DIRECT`（加上 io_uring 做异步，见[下一章](io-models.md)），让延迟更可预测。代价是没有了预读和写合并，所有优化都要自己做。

## 模型加载慢，从哪里查

把前面的内容串起来，模型加载时间可以按路径拆开看：

| 环节 | 典型问题 | 怎么看 |
| --- | --- | --- |
| 存储 | 网络文件系统随机读慢、对象存储单连接带宽低 | `iostat -x`、网卡流量；换 `eager` / `prefetch` / 流式加载器 |
| 页缓存 | 第一次加载冷启动，或者缓存被挤掉 | `free` 的 buff/cache；`vmtouch`、`fincore` 看文件有多少页在缓存里 |
| 反序列化 | pickle 格式要整个解析 | 换成 safetensors |
| CPU → GPU | 可换页内存要中转，没有并行 | 锁页缓冲区、多个 rank 并行读（见[锁页内存](pinned-numa.md)） |
| 加载后处理 | 量化、重排、切分 | 预先处理好（`sharded_state`、`ipc_cache`） |

!!! interview "面试怎么答"
    被问"描述一次文件写入从调用开始到落盘的全过程"：按层讲。`write` 陷入内核，VFS 找到文件交给文件系统，文件系统必要时分配块、更新 inode，数据拷进页缓存、页标记为脏就返回——此时数据只在内存里。之后后台写回线程按时间和脏页比例把脏页交给块设备层，bio 经过 I/O 调度器合并排序后交给驱动，NVMe 驱动把命令放进提交队列、敲门铃，盘通过 DMA 取走数据、写进自己的缓存，经完成队列和中断通知完成。要保证持久化必须 `fsync`：写回脏页、提交文件系统日志、再让盘把缓存刷到介质（FLUSH / FUA）。最后讲瓶颈：写入慢往往是脏页超过 `dirty_ratio` 被迫同步写回，或者频繁的小 fsync，被每次盘往返的固定开销卡住，对策是组提交；如果不需要页缓存（自己有缓存的存储系统），用 `O_DIRECT` 加异步 I/O。

## 练习

**1. cp 完就能拔电源吗。** 用 `cp` 把一个 10 GB 的检查点拷到本地盘，命令几秒钟就返回了，这时马上断电会怎样？训练框架保存检查点时，怎么保证"保存成功"之后断电也不会得到一个损坏的文件？

??? success "参考答案"
    `cp` 返回只说明数据进了页缓存，大量脏页可能还没写回，断电后文件可能不完整甚至全是零。可靠的保存流程是：写到一个临时文件，`fsync` 这个文件，再 `rename` 成正式的文件名（rename 在同一个文件系统内是原子的），最后 `fsync` 所在的目录，让目录项的修改也落盘。这样断电后看到的要么是旧的完整文件，要么是新的完整文件，不会看到写了一半的。

**2. KV Cache 的 SSD 层要不要 fsync。** 一个推理服务把淘汰出显存和 CPU 内存的 KV 块写到本地 NVMe 上，以便以后命中前缀时再读回来。每写一个块要不要 fsync？要不要走页缓存？

??? success "参考答案"
    不需要 fsync：KV Cache 是缓存，丢了可以重算，断电后整个缓存层直接作废即可，不需要持久化保证，fsync 只会让每次写入多付一次盘的往返。也不建议走页缓存：KV 块本来就是从内存里淘汰下来的，再在页缓存里存一份等于没淘汰，还会挤掉权重文件等更有用的缓存，读写延迟也受写回线程影响。常见的做法是 `O_DIRECT` 加 io_uring 异步读写，块大小按盘的块大小对齐，自己维护索引；一致性问题（读到写了一半的块）用校验和或者写完再更新索引来解决。

**3. NFS 上加载慢。** 模型放在 NFS 上，vLLM 默认策略加载一个 140 GB 的模型花了 20 分钟，同样的文件拷到本地 NVMe 以后 2 分钟。可能的原因是什么？除了拷到本地，还能怎么改？

??? success "参考答案"
    默认策略用 mmap 按需读取，每个张量、每个 rank 访问到哪里才缺页读到哪里，在 NFS 上变成大量随机的小读，每次都是一次网络往返，预读也帮不上忙；多个 rank 同时这样读，NFS 服务器的并发和延迟就成了瓶颈。另外，140 GB 可能超过了可用内存的 90%，不会触发自动预取。可以改用 `--safetensors-load-strategy eager`（每个文件顺序整块读进内存）或者 `prefetch`（多线程顺序预读进页缓存），用顺序大块读代替随机小读；也可以用 `runai_streamer` 这类并发流式加载器，或者用 `sharded_state` 预先按 rank 切好，每个 rank 只读自己的那一份。

## 小结

- [x] `write` 返回时数据只在页缓存里；写回线程按时间和脏页比例写盘，脏页过多会拖住写入者。
- [x] `fsync` = 写回脏页 + 提交文件系统日志 + 刷盘缓存，本机一次约 0.8 ms；频繁的小 fsync 要靠组提交。
- [x] 页缓存让第二次读快几十倍，多个进程共享，内存紧张时可回收；容器里计入内存用量。
- [x] mmap 零拷贝、按需读入、跨进程共享；safetensors 的"头 + 偏移"布局让任意张量都能零拷贝取出。网络文件系统上 mmap 随机读很慢，vLLM 提供 `eager`、`prefetch` 和流式加载器。
- [x] `O_DIRECT` 绕过页缓存，要求对齐，适合自己管理缓存的系统（包括 KV Cache 的 SSD 层）。
