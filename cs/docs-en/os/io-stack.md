# Files and I/O: how a write reaches the disk

<p class="lead">"Describe a file write's whole path from the call to the disk" is a classic systems interview question, and the path runs through the system call, the filesystem, the page cache, the block layer, the driver and the drive itself. An inference system is full of this path's shadow: why a model's second load is far faster, why mmaping weights from network storage is slow, whether a KV cache on an SSD needs an fsync, why a log write occasionally stalls for hundreds of milliseconds. This chapter follows one write and one read through, then covers mmap (why safetensors suits it), O_DIRECT and the strategies vLLM has for loading weights.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. When `write()` returns, where is the data? Is it lost if the power fails then?
    2. What does `fsync()` do? Why is "write a little and fsync every time" so slow?
    3. Why is the same model's second load far faster? What takes that away?
    4. How does reading a file with mmap differ from reading it with read? Why does safetensors suit mmap?
    5. What is O_DIRECT? Where is it used?

??? success "Answers (try it yourself first, then expand)"
    1. In the kernel's page cache: the kernel copied the data into the file's pages, marked them dirty and returned, with the write-back to disk done by a background thread later. A power failure or a kernel crash at that point loses the data (the process crashing does not, since the data is already in the kernel).
    2. fsync writes back all of that file's dirty pages, commits the metadata in the filesystem's journal, and makes the drive flush its own write cache to the persistent medium (FLUSH / FUA), returning only when all of it is done. Every call waits out a round trip to the drive, so the cost of a small write is dominated by the fixed overhead, measured here at about 0.8 ms each, a thousand-odd per second; accumulating several writes before one fsync (group commit) is what raises the throughput.
    3. The first load read the file's contents into the page cache, and as long as memory has not been taken for something else the second read hits the cache and never touches the disk, measured here at tens of times faster. What takes it away: a reboot, reclamation under memory pressure, `posix_fadvise(DONTNEED)` dropping a file's cache, or `echo 3 > /proc/sys/vm/drop_caches` (as root).
    4. read copies the data from the page cache into a user buffer; mmap maps the page cache's pages straight into the process's address space, reading them on demand through faults, with no copy, and several processes mapping the same file share the same physical pages. safetensors is laid out as "a header plus each tensor's contiguous raw bytes", with the header recording every tensor's offset, so after an mmap any one tensor comes out with no copy and only the parts used are read.
    5. Opening a file with O_DIRECT makes reads and writes bypass the page cache and DMA straight between the user buffer and the drive; it requires the buffer's address, the length and the file offset to be aligned to the block size. It suits systems that manage their own cache (databases, KV stores, a caching layer putting the KV cache on an SSD), avoiding a second copy in the page cache that would evict other caches, and making the latency more predictable.

## A write's journey {#一次-write-的旅程}

![Figure: a write's journey - the user buffer, the page cache, the block layer, the device; O_DIRECT and mmap](../assets/figures/write-path.svg){.aig-svg}

After a `write(fd, buf, n)`:

1. **the system call**: a trap from user space into the kernel, and the arguments are checked;
2. **the VFS**: the file descriptor finds the open file, which goes to a particular filesystem (ext4, xfs);
3. **the filesystem**: disk blocks are allocated for the file where needed (a filesystem with delayed allocation defers this to the write-back), and the size and the times in the inode are updated;
4. **the page cache**: the data is copied from the user buffer into the file's pages (allocating one when it is not cached, and reading the old contents first when the write does not fill a page), the pages are marked **dirty**, and `write` returns;
5. **write-back**: the background write-back threads (the flushers) write the dirty pages to the disk by age (dirty longer than `dirty_expire_centisecs`, 30 seconds by default) and by volume (past `dirty_background_ratio`). With too many dirty pages (past `dirty_ratio`), the writing process is itself held up and made to take part in the write-back, which is the common reason a log write "occasionally stalls for hundreds of milliseconds";
6. **the block layer**: the written-back pages are made into bio requests, merged and ordered by the I/O scheduler, and handed to the driver;
7. **the driver and the device**: the NVMe driver puts a command in the submission queue and rings a doorbell register, the drive's controller DMAs the data away into its own cache, and on finishing notifies the kernel through the completion queue and an interrupt;
8. **real persistence**: the drive has a write cache of its own, and only a FLUSH command or a write carrying the FUA flag (force unit access) guarantees the data has reached the flash or the platter.

**What `fsync` does is steps 5 to 8 at once and in full**: write back all of that file's dirty pages, commit the filesystem's journal (the metadata), and send a FLUSH for the drive to push its cache to the medium, returning only when all of it is done.

```python title="write_fsync.py"
import os
import time

MiB = 1 << 20


def dirty_mib():                                   # the dirty pages not yet written back anywhere in the system
    for line in open("/proc/meminfo"):
        if line.startswith("Dirty:"):
            return int(line.split()[1]) / 1024


data = os.urandom(MiB)
with open("big.bin", "wb") as f:
    t0 = time.perf_counter()
    for _ in range(256):
        f.write(data)
    f.flush()                                      # handed from Python's buffer to the kernel: the write system call returns
    t1 = time.perf_counter()
    print(f"write 256 MiB：{(t1 - t0) * 1e3:.0f} ms（数据在页缓存里，脏页约 {dirty_mib():.0f} MiB）")
    os.fsync(f.fileno())                           # wait for the data and the metadata to really reach the disk
    t2 = time.perf_counter()
    print(f"fsync：{(t2 - t1) * 1e3:.0f} ms（之后脏页约 {dirty_mib():.0f} MiB）")

# like a database's write-ahead log: an fsync every 4 KiB written
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

```text title="output (on this machine)"
write 256 MiB：130 ms（数据在页缓存里，脏页约 256 MiB）
fsync：974 ms（之后脏页约 0 MiB）
4 KiB 写入 + fsync：每次 0.82 ms，每秒最多约 1,218 次
```

The 256 MiB `write` is only a memory copy and is fast; the `fsync` waits for the drive and is far slower. The last part fsyncs every 4 KiB, like a database's write-ahead log: every one pays a round trip of "write back, commit the journal, flush the drive's cache", and the throughput is pinned by that fixed cost. A database's answer is **group commit**: accumulate several concurrent transactions' log records and fsync once.

## Reads and the page cache {#读与页缓存}

A read's path is the reverse: `read` checks the page cache first and copies straight to the user on a hit; on a miss it issues a disk read, fills the page cache and then copies. The kernel also does **readahead**: recognizing a sequential read, it reads the following pages in advance.

```python title="cold_warm.py"
import os
import time

fd = os.open("big.bin", os.O_RDONLY)               # the 256 MiB file the previous script wrote and fsynced
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # drop this file's pages from the page cache (only clean pages can be dropped, hence the fsync above)


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

```text title="output (on this machine)"
第一次读（从盘上读）：143 MiB/s
第二次读（页缓存命中）：4,731 MiB/s，快 33 倍
```

The page cache is what "free" memory is mostly used for: buff/cache in `free` is it, and the kernel reclaims the clean cached pages under memory pressure. How it affects inference:

- **a model's second load is far faster**: the weight file is still in the page cache and the load is only deserializing and copying to the GPU; a reboot or another process taking the memory makes it the slow one again;
- **several processes share it**: tensor-parallel workers reading the same file hold one copy in the page cache;
- **it counts towards a container's memory**: the page cache counts in a cgroup's memory usage, so "memory nearly full" is not necessarily a leak; see [containers](containers.md).

## mmap: mapping a file into the address space {#mmap把文件映射进地址空间}

`mmap` maps part of a file into the process's virtual address space, and touching an address takes a major fault and a disk read when that file page is not in the page cache, or a minor fault and a direct mapping when it is (the two kinds of fault are in [virtual memory](virtual-memory.md)). Against `read`: no copy into a user buffer; only the parts really touched are read; and several processes mapping one file share the same physical pages.

The safetensors format for weight files was designed for exactly this: 8 bytes of header length at the start, then a JSON header recording every tensor's data type, shape and offset into the data area, then every tensor's raw bytes laid out contiguously. After an mmap, any tensor can be "seen" with no copy:

```python title="mini_safetensors.py"
import json
import mmap
import struct

import numpy as np


def rss_mib():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024


# write a safetensors file: 8 bytes (little-endian) of header length + a JSON header + every tensor's raw bytes
tensors = {f"layers.{i}.weight": np.full((1024, 1024), i, dtype=np.float16) for i in range(16)}   # 16 tensors of 2 MiB
header, off = {}, 0
for name, t in tensors.items():
    header[name] = {"dtype": "F16", "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
    off += t.nbytes
raw = json.dumps(header).encode()
with open("model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(raw)) + raw)
    for t in tensors.values():
        f.write(t.tobytes())

# reading: mmap the whole file and "see" a tensor at the offset the header records, with no copy
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
total = float(w.sum(dtype=np.float32))              # really read these 2 MiB: the faults map the file's pages in
print("32 MiB 的文件只读了 2 MiB，RSS 增加不到 3 MiB：", rss_mib() - before < 3)
```

```text title="output"
文件里有 16 个张量，layers.7.weight 的形状 (1024, 1024) ，值 7.0
数组直接指向 mmap 的内存，没有拷贝： True
32 MiB 的文件只读了 2 MiB，RSS 增加不到 3 MiB： True
```

Against PyTorch's `.bin` (pickle) format, which has to be deserialized whole and may execute arbitrary code while doing so. safetensors is both safe and readable on demand, so each tensor-parallel rank need only read its own slice.

mmap's cost is that reading on demand faults page by page: with readahead helping on local NVMe this is no great problem; on a network filesystem every fault is a network round trip and the random reads defeat the readahead, making the load absurdly slow. vLLM's `safetensors_load_strategy` (`vllm/config/load.py`) exists for exactly this:

- the default (unset): mmap and read on demand; on detecting NFS, and when the whole checkpoint is under 90% of the available memory, it prefetches the file into the page cache first;
- `lazy`: mmap only with no automatic prefetch, which suits a local disk;
- `eager`: read the whole file into CPU memory before loading, recommended for network filesystems like Lustre and NFS, replacing random small reads with large sequential ones;
- `prefetch`: read the file sequentially into the page cache with several threads (`safetensors_prefetch_num_threads`) before loading, so the mmaps that follow all hit the cache.

`load_format` offers other loaders too: `runai_streamer` streams concurrently from object storage or a filesystem with Run:ai's Model Streamer, `sharded_state` loads a checkpoint already split by tensor parallelism (each rank reading only its own file), `tensorizer` uses CoreWeave's tensorizer format, and `ipc_cache` maps already-quantized weights from a local weight cache daemon through CUDA IPC, so restarting the engine barely reloads anything.

## O_DIRECT: bypassing the page cache {#o_direct绕过页缓存}

Opening a file with `O_DIRECT` makes reads and writes bypass the page cache and DMA straight between the user buffer and the drive. The condition is alignment: the buffer's address, the length and the file offset all have to be multiples of the logical block size (512 bytes or 4 KiB usually):

```python title="odirect.py" ci="no"
import mmap
import os

fd = os.open("direct.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
try:
    os.write(fd, b"x" * 1000)                       # the length is not a multiple of the block size and the buffer is not aligned
except OSError as e:
    print("不对齐的 O_DIRECT 写入失败：", e.strerror)
buf = mmap.mmap(-1, 4096)                          # memory from mmap is page-aligned
buf.write(b"y" * 4096)
print("对齐的写入成功：", os.write(fd, buf), "字节")
os.close(fd)
```

```text title="output"
不对齐的 O_DIRECT 写入失败： Invalid argument
对齐的写入成功： 4096 字节
```

Who uses it: systems that manage their own cache. A database or a KV store has its own buffer pool, and a second copy in the page cache is waste that evicts other useful cache; a caching layer putting the KV cache on a local SSD commonly uses `O_DIRECT` too (with io_uring for asynchrony, see [the next chapter](io-models.md)) to make the latency more predictable. The cost is losing readahead and write merging, so every optimization is yours to make.

## A slow model load, where to look {#模型加载慢从哪里查}

Putting all of this together, a model's load time can be taken apart along the path:

| Stage | The typical problem | How to see it |
| --- | --- | --- |
| storage | slow random reads on a network filesystem, low single-connection bandwidth from object storage | `iostat -x`, the network card's traffic; switch to `eager` / `prefetch` / a streaming loader |
| the page cache | a cold first load, or the cache evicted | buff/cache in `free`; `vmtouch` and `fincore` show how much of a file is cached |
| deserializing | the pickle format has to be parsed whole | switch to safetensors |
| CPU to GPU | pageable memory staging, with no parallelism | pinned buffers, several ranks reading in parallel (see [pinned memory](pinned-numa.md)) |
| post-processing | quantizing, reordering, sharding | do it in advance (`sharded_state`, `ipc_cache`) |

!!! interview "Answering in an interview"
    Asked "describe a file write's whole path from the call to the disk": go layer by layer. `write` traps into the kernel, the VFS finds the file and hands it to the filesystem, which allocates blocks where needed and updates the inode, and the data is copied into the page cache and the pages marked dirty, at which point it returns, with the data only in memory. The background write-back threads then hand the dirty pages to the block layer by age and by the dirty ratio, the bios are merged and ordered by the I/O scheduler and given to the driver, the NVMe driver puts a command in the submission queue and rings the doorbell, and the drive DMAs the data away into its own cache and signals completion through the completion queue and an interrupt. Persistence requires an `fsync`: write back the dirty pages, commit the filesystem journal, and make the drive flush its cache to the medium (FLUSH / FUA). Finish with the bottlenecks: a slow write is usually dirty pages past `dirty_ratio` forcing synchronous write-back, or frequent small fsyncs pinned by each round trip's fixed cost, for which the answer is group commit; and where the page cache is not wanted (a storage system with its own cache), `O_DIRECT` plus asynchronous I/O.

## Exercises {#练习}

**1. Can the power go out right after cp.** Copying a 10 GB checkpoint to a local disk with `cp` returns in a few seconds. What happens if the power fails right then? How does a training framework guarantee that "saved successfully" survives a power failure without a corrupt file?

??? success "Answer"
    `cp` returning only means the data reached the page cache, and a great many dirty pages may not be written back, so after a power failure the file may be incomplete or all zeros. The reliable sequence is: write to a temporary file, `fsync` it, `rename` it to the real name (a rename within one filesystem is atomic), and finally `fsync` the containing directory so the directory entry's change reaches the disk too. After a power failure you then see either the complete old file or the complete new one, never one half written.

**2. Does the KV cache's SSD tier need fsync.** An inference service writes the KV blocks evicted from device and CPU memory to a local NVMe so a later prefix hit can read them back. Does each block need an fsync? Should it go through the page cache?

??? success "Answer"
    No fsync is needed: the KV cache is a cache, what is lost can be recomputed, and after a power failure the whole tier can simply be discarded, so there is no durability requirement and an fsync only adds a round trip to the drive per write. The page cache is not advisable either: the KV blocks were evicted from memory to begin with, so a copy in the page cache undoes the eviction, evicts more useful cache like the weight files, and leaves the latency at the write-back threads' mercy. The usual approach is `O_DIRECT` with io_uring for asynchronous reads and writes, the block size aligned to the drive's, and an index of your own; consistency (reading a half-written block) is handled with a checksum or by updating the index after the write completes.

**3. A slow load over NFS.** With the model on NFS, vLLM's default strategy took 20 minutes to load a 140 GB model, while the same files on local NVMe took 2. What could it be? Besides copying it locally, what else can be changed?

??? success "Answer"
    The default strategy mmaps and reads on demand, so every tensor and every rank faults wherever it touches, which on NFS becomes a great many random small reads each costing a network round trip, with readahead no help; and with several ranks doing this at once, the NFS server's concurrency and latency become the bottleneck. Besides, 140 GB may exceed 90% of the available memory, so the automatic prefetch never triggers. Switch to `--safetensors-load-strategy eager` (reading each file into memory sequentially) or `prefetch` (several threads reading sequentially into the page cache), replacing random small reads with large sequential ones; or use a concurrent streaming loader like `runai_streamer`, or `sharded_state` split by rank in advance so each rank reads only its own part.

## Summary {#小结}

- [x] When `write` returns the data is only in the page cache; the write-back threads write it out by age and by the dirty ratio, and too many dirty pages hold the writer up.
- [x] `fsync` = write back the dirty pages + commit the filesystem journal + flush the drive's cache, about 0.8 ms each here; frequent small fsyncs need group commit.
- [x] The page cache makes a second read tens of times faster, is shared between processes, and is reclaimable under memory pressure; it counts towards a container's memory.
- [x] mmap copies nothing, reads on demand and shares across processes; safetensors' "header plus offsets" layout lets any tensor come out with no copy. mmap's random reads are very slow on a network filesystem, for which vLLM offers `eager`, `prefetch` and streaming loaders.
- [x] `O_DIRECT` bypasses the page cache and requires alignment, which suits systems that manage their own cache (including the KV cache's SSD tier).
