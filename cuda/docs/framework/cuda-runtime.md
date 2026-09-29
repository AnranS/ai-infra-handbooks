# PyTorch 的 CUDA 运行时：异步、stream 与显存

<p class="lead">在 GPU 上，PyTorch 的每个算子调用都只是把 kernel <strong>放进队列</strong>就返回了，Python 代码和 GPU 并行地往前跑。推理引擎的性能很大程度上取决于能不能保持这种异步：一次不经意的同步，就会让 CPU 停下来等 GPU、GPU 又停下来等 CPU。这一章讲 PyTorch 在 CUDA 之上的这层运行时：异步执行与同步点、stream、缓存分配器和 CUDA Graph。</p>

!!! note "本章的代码需要 NVIDIA GPU"
    本章的脚本在书的校验环境里只做了语法检查（那里没有 GPU），在 WSL2 或 Linux + NVIDIA GPU 上可以直接运行。概念对应 CUDA 手册的[流、并发与 CUDA Graphs](../tools/streams.md)，那一章用 CUDA C++ 讲了同样的机制。

!!! question "自测：能答上来就可以跳过本章"
    1. `y = x @ w` 在 GPU 上执行完这一行时，`y` 算好了吗？
    2. 哪些常见操作会隐式地同步 CPU 和 GPU？
    3. 怎样正确地测量一段 GPU 代码的耗时？
    4. 在另一个 stream 上使用一个张量时，为什么要调用 `record_stream`？
    5. `torch.cuda.memory_allocated()` 和 `memory_reserved()` 有什么区别？

## 异步执行与同步点

CUDA 算子的调用只做三件事：检查参数、分配输出张量（从缓存分配器里拿，很快）、把 kernel 放进当前 stream 的队列。返回的时候 kernel 很可能还没开始执行。只要 CPU 不去**读**结果，CPU 就可以一直往前提交，GPU 在后面追——CPU 提交的开销被完全藏在 GPU 的执行时间后面。

下面这些操作需要知道 GPU 上的值，会让 CPU **等到 GPU 把前面的工作全部做完**：

| 操作 | 为什么要同步 |
| --- | --- |
| `.item()`、`.tolist()`、`.cpu()`、`print(tensor)` | 把值拷回 CPU |
| `torch.cuda.synchronize()` | 显式等待 |
| `nonzero()`、布尔掩码索引 `x[mask]`、`torch.unique` | 输出的形状取决于数据，要先知道结果才能分配输出 |
| `if tensor > 0:`、`while not done.all():` | Python 的控制流需要一个具体的布尔值 |
| 从可分页（非锁页）内存拷到 GPU | 驱动要先把数据拷到临时的锁页缓冲区 |

推理引擎的调度循环里，每一步都要知道"采样出的 token 是什么、哪些请求结束了"，这就是一次同步点；重叠调度（mini-sglang 的[重叠调度](minisgl://schedule/overlap/)）的核心就是把这次同步推迟到下一步的 kernel 已经提交之后。

**测量耗时**要用 CUDA event（或者在计时前后 `synchronize`），否则测到的只是"提交的时间"：

```python title="timing.py" run="no"
import torch

x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
w = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):                     # 预热：第一次调用有 cuBLAS 初始化、选择算法的开销
    x @ w

start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record()                         # 事件也被放进队列，GPU 执行到这里时记下时间
for _ in range(10):
    y = x @ w
end.record()
end.synchronize()                      # 等 end 事件完成
ms = start.elapsed_time(end) / 10
print(f"每次 {ms:.3f} ms，{2 * 8192**3 / ms / 1e9:.0f} TFLOPS")
```

## stream：让拷贝和计算重叠

所有算子默认放在**当前 stream**上，同一个 stream 里的 kernel 按顺序执行。要让两件事并行（最典型的是"把下一批数据拷到 GPU"和"计算这一批"），就把它们放到不同的 stream 上，并在需要的地方显式建立依赖：

```python title="overlap_copy.py" run="no"
import torch

compute = torch.cuda.current_stream()
copy = torch.cuda.Stream()
w = torch.randn(4096, 4096, device="cuda")
batches = [torch.randn(4096, 4096).pin_memory() for _ in range(4)]   # 锁页内存：才能真正异步地拷贝

next_gpu = batches[0].to("cuda", non_blocking=True)
for i in range(len(batches)):
    cur = next_gpu
    if i + 1 < len(batches):
        with torch.cuda.stream(copy):                   # 在拷贝 stream 上预取下一批
            next_gpu = batches[i + 1].to("cuda", non_blocking=True)
    y = cur @ w                                         # 在计算 stream 上算这一批
    compute.wait_stream(copy)                           # 下一轮用 next_gpu 之前，等它拷完
    next_gpu.record_stream(compute)                     # 告诉分配器：这块显存还会被计算 stream 用
torch.cuda.synchronize()
```

两个容易出错的地方：

- **依赖要显式建立**：`wait_stream` 或者 `event.record()` + `stream.wait_event(event)`。漏掉了，计算可能读到还没拷完的数据，而且只会偶尔出错；
- **`record_stream`**：缓存分配器按 stream 管理显存。一个在拷贝 stream 上分配的张量，被释放后分配器会认为"拷贝 stream 上的工作完成了就可以复用"，而不知道计算 stream 还在读它。`record_stream(compute)` 让分配器等计算 stream 上当前的工作完成后才复用这块显存。

## 缓存分配器

`cudaMalloc` / `cudaFree` 很慢而且会同步，所以 PyTorch 用**缓存分配器**：释放的显存不还给驱动，按大小留在池子里复用（C++ 手册的[分配器与内存池](cpp://memory/allocators/)实现过一个简化版）。于是有两个数字：

- `torch.cuda.memory_allocated()`：当前被张量占用的字节数；
- `torch.cuda.memory_reserved()`：分配器从驱动拿到的总字节数（`nvidia-smi` 看到的占用主要是它）。

两者的差是缓存着的空闲块。"明明 reserved 还有很多空余却 OOM"通常是碎片：空闲块的大小凑不出一块连续的大内存。常用的排查和缓解手段：

```python title="memory_debug.py" run="no"
import torch

torch.cuda.memory._record_memory_history(max_entries=100_000)   # 记录每次分配和释放的调用栈
torch.cuda.reset_peak_memory_stats()

x = torch.randn(4096, 4096, device="cuda")
y = x @ x
del x

print(f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB，"
      f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB，"
      f"峰值 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
torch.cuda.memory._dump_snapshot("mem_snapshot.pickle")        # 拖到 https://pytorch.org/memory_viz 里查看时间线
```

- 环境变量 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 让分配器用虚拟内存映射扩展段，大幅减少碎片；
- 推理引擎在启动时就按 `gpu_memory_utilization` 把 KV Cache 一次性分配走，运行时几乎不再向分配器要大块内存。

## CUDA Graph

decode 一步有几百个 kernel，每个 kernel 的计算只有几微秒到几十微秒，CPU 提交 kernel 的开销（每个几微秒）会占掉相当大的比例。**CUDA Graph** 把一整串 kernel 录制下来，之后一次提交、整体重放：

```python title="cuda_graph_decode.py" run="no"
import torch

model = torch.nn.Sequential(torch.nn.Linear(4096, 11008), torch.nn.SiLU(), torch.nn.Linear(11008, 4096)).cuda().half()
static_in = torch.zeros(8, 4096, device="cuda", dtype=torch.half)   # 录制和重放都用这块固定的显存

side = torch.cuda.Stream()                    # 在非默认 stream 上预热，让分配器、cuBLAS 完成初始化
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side), torch.inference_mode():
    for _ in range(3):
        model(static_in)
torch.cuda.current_stream().wait_stream(side)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.inference_mode():
    static_out = model(static_in)             # 只录制，不执行

for step in range(5):
    new_input = torch.randn(8, 4096, device="cuda", dtype=torch.half)
    static_in.copy_(new_input)                # 把新输入拷进固定的地址
    g.replay()                                # 一次提交整张图
    token_logits = static_out.clone()         # 结果在固定的输出地址上
```

录制下来的是**固定的 kernel 序列和固定的显存地址**，所以：

- 输入要拷进录制时用的那块显存，输出从固定的地址读——漏拷一个输入，重放用的就是旧数据；
- 形状必须固定，推理引擎为几个常见的 batch 大小各录一张图，实际 batch 向上补齐到最近的一档；
- 录制期间不能有同步点、不能有依赖数据的控制流，也不能在图外释放图里用到的显存。

`torch.compile(mode="reduce-overhead")` 会自动做这些事；vLLM、SGLang 则在模型外面自己管理多张图（见推理系统手册的 [CUDA Graphs 与 torch.compile](serving://engine/graphs-compile/)）。

!!! interview "面试怎么答"
    PyTorch 的 CUDA 运行时题：GPU 算子只是入队就返回，`.item()`、`.cpu()`、依赖数据的形状和控制流都会同步 CPU，推理的热路径要避免；计时用 CUDA event 或前后同步，并先预热；跨 stream 使用张量要建立依赖并 `record_stream`，异步拷贝要用锁页内存。显存看两个数：allocated 是张量在用的，reserved 是缓存分配器从驱动拿到的，reserved 很多却 OOM 通常是碎片，看内存快照、开 `expandable_segments`。CUDA Graph 用固定地址和形状录制，输入要拷进固定的缓冲区。

## 练习

1. 下面的采样代码每一步都会让 CPU 等 GPU 一次。找出同步点，并说明怎样改写才能不同步（提示：结束判断可以留在 GPU 上）。

    ```python
    next_tokens = torch.argmax(logits, dim=-1)
    for i, tok in enumerate(next_tokens.tolist()):
        if tok == eos_id:
            finished[i] = True
    ```

??? success "参考答案"
    `.tolist()` 把结果拷回 CPU，是一次同步；之后的 Python 循环也只能在拷贝完成后开始。改写：在 GPU 上算出结束标志 `done = next_tokens == eos_id`，和 token 一起用 `non_blocking=True` 拷到锁页内存的缓冲区里，并记录一个 event；
    调度器先提交下一步的 kernel，再在需要用到结果的时候 `event.synchronize()`。这样拷贝和同步都藏在下一步的 GPU 执行时间后面——这就是重叠调度的做法。

2. 为什么用 CUDA Graph 重放时，输出要写成 `static_out.clone()`，而不能直接把 `static_out` 放进结果列表里？

??? success "参考答案"
    `static_out` 指向录制时固定的那块显存，下一次 `replay()` 会覆盖它。直接保存引用，列表里所有元素最后都是同一块显存、同一个（最后一次的）结果。要保存结果就拷贝出来；推理引擎通常在同一步里就把需要的部分（采样出的 token）拷到别处。

## 小结

- [x] GPU 上的算子只是入队就返回；读取结果（`.item()`、`.cpu()`、依赖数据的形状和控制流）会同步，推理的热路径要避免同步。
- [x] 计时用 CUDA event 或者前后同步，并先预热。
- [x] 不同 stream 可以并行，依赖要显式建立；跨 stream 使用张量要 `record_stream`；异步拷贝要用锁页内存。
- [x] 缓存分配器：allocated 是在用的，reserved 是拿到手的；碎片导致 OOM 时看内存快照、开 `expandable_segments`。
- [x] CUDA Graph 用固定的地址和形状录制、一次提交整串 kernel；输入要拷进固定地址，输出要及时拷走。
