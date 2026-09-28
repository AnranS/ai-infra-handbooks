# 自定义 CUDA kernel

<p class="lead">mini-sglang 自己写的 CUDA kernel 只有三个：把新 K/V 写进 KV 池（<code>store_cache</code>）、词表并行的嵌入查表（<code>indexing</code>）、以及基数树里比较两段 token 的 <code>fast_compare_key</code>（C++，在 CPU 上）。前两个是纯粹的"按下标搬运整行数据"，逻辑简单，但每层每步都要执行，值得写成高效的 kernel。这一章写出它们的 CUDA 版本，用两个版本的 nvcc 编译，并在 CUDA 手册的 CPU 模拟器上运行自检。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 写 KV 缓存的 kernel，瓶颈在计算还是访存？理论上最快能多快？
    2. 为什么"一个 warp 负责一行"，而不是"一个线程负责一行"？
    3. 用 `uint4` 拷贝有什么好处？什么时候不能用？
    4. PyTorch 的 `k_cache[idx] = k` 在 GPU 上也能工作，为什么还要自己写？

**本章要写的文件**：`kernel/csrc/kv_kernels.cuh`、`kernel/csrc/ext.cu`、`kernel/cuda_ext.py`，以及 `kernel/__init__.py` 中的分派；自检程序 `tests/cuda/test_kv_kernels.cu`。

## 按下标搬运整行

写 KV 缓存要做的事是：对本轮每个新 token `i`，把 `k[i]`（一行 `头数 × head_dim` 个元素）拷到 `k_cache[out_loc[i]]`，`v` 同理。嵌入查表是反过来：`out[i] = weight[ids[i]]`。两者都没有任何计算，瓶颈完全是显存带宽——理想情况下，搬运的字节数除以显存带宽就是耗时。

@@code python/minisgl/kernel/csrc/kv_kernels.cuh@@

设计要点（原理见 [CUDA 手册的内存层次与访存优化](cuda://basics/memory/)）：

- **一个 warp 负责一行**：Qwen3-0.6B 的一行 K 是 8 × 128 × 2 = 2 KB。一个线程搬 2 KB 太慢，而一个 warp 的 32 个线程各搬相邻的 16 字节，一次就能搬 512 字节，访存是**合并**的；
- **向量化**：按 `uint4`（16 字节）为单位拷贝，一条指令搬 16 字节。要求行字节数和指针都是 16 的倍数；否则退回按元素大小（2 或 4 字节）拷贝。选择由主机端根据形状和对齐决定，kernel 本身按类型 `V` 模板化；
- **词表并行的掩码**：不在本 rank 词表范围内的 token，整行写 0（`V{}`），配合之后的 all-reduce。

PyTorch 的 `k_cache[idx] = k` 在 GPU 上当然也能工作，但它是一个通用的 `index_put` kernel，要处理任意形状和步长；专用 kernel 按固定的行大小生成代码、向量化拷贝，K、V 在同一个 kernel 里完成，更接近带宽上限。官方还加了 PDL（Programmatic Dependent Launch，Hopper 起支持）的开关，让它与前后的 kernel 部分重叠。

## 接进 PyTorch

@@code python/minisgl/kernel/csrc/ext.cu@@

扩展由 `torch.utils.cpp_extension.load` 在第一次使用时即时编译。CUDA stream 由 Python 端传入（`torch.cuda.current_stream().cuda_stream`），这样扩展只依赖 PyTorch 的 CPU 头文件和 CUDA runtime——也让我们在没有 CUDA 版 PyTorch 的开发机上能够编译检查它。

@@code python/minisgl/kernel/__init__.py:store_cache@@

GPU 上且扩展编译成功时用自定义 kernel，否则（CPU、没有 nvcc）用 PyTorch 参考实现。

!!! diff "与官方的差异：tvm-ffi 与 torch 扩展"
    官方用 [tvm-ffi](https://github.com/apache/tvm-ffi) 做 JIT 编译和 Python 绑定：kernel 的行大小、线程数等作为 C++ 模板参数，按实际形状即时生成专门的版本（`kernel/utils.py` 的 `load_jit`）。我们用更常见的 PyTorch 扩展，行大小作为运行时参数。kernel 的思路（一个 warp 一行、向量化拷贝、词表掩码）相同。

## 在没有 GPU 的机器上验证

自检程序用随机数据分别测试两个 kernel 的向量化路径和逐元素路径，与 CPU 上的参考结果逐字节比较：

@@code tests/cuda/test_kv_kernels.cu@@

它用 CUDA 13.4 和 12.9 两个版本的 nvcc 编译，然后在 [CUDA 手册](cuda://)的 CPU 模拟器上运行——模拟器为每个 CUDA 线程开一个协程，真实地执行 warp 内的并行逻辑：

@@code examples/ch19_kernels.py@@

@@output ch19_kernels@@

PyTorch 扩展 `ext.cu` 同样用两个版本的 nvcc 编译检查（`tools/check.py`），但它依赖 CUDA 版的 PyTorch 运行时，只能在 GPU 上执行。

!!! upstream "官方实现"
    - 写 KV：`kernel/csrc/jit/store.cu`，Python 端 @@upstream kernel/store.py:store_cache@@
    - 查表：`kernel/csrc/jit/index.cu`，Python 端 @@upstream kernel/index.py:indexing@@（行字节数是 2048、1024 的倍数时，用 4 个或 2 个 warp 分担一行）
    - 基数树的比较：`kernel/csrc/src/radix.cpp`（CPU 上的 `std::mismatch`）

## 测试

@@code tests/test_ch19_kernels.py:test_cuda_kernels_on_cpu_emulator@@

## 练习

1. 在 H100（显存带宽约 3.35 TB/s）上，Qwen3-0.6B 一次 decode、批大小 64，每层写 KV 要搬多少字节？理论耗时是多少？这个 kernel 值得用 CUDA Graph 吗？
2. 官方的 `indexing` 在行很大时用多个 warp 分担一行。为 `embedding_kernel` 加上这个优化，并在模拟器上验证。
3. 把 `store_kv_kernel` 改成"一个线程块负责多行、每个 warp 一行"的写法，和现在"每 4 个 warp 一个块"的写法有什么区别？

??? success "参考答案"
    1. 每个 token 的 K、V 各 2 KB，64 个 token 共 256 KB（读 k、v 再写入池子，读写合计约 512 KB），约 0.15 µs——远小于 kernel 的启动开销（几微秒）。所以它必须放进 CUDA Graph，否则启动开销会是实际工作的几十倍。
    2. `num_splits` 个 warp 负责一行：`warp_id / num_splits` 是行号，`warp_id % num_splits` 是这一行里的第几段，每段 `row / num_splits` 个元素。
    3. 现在的写法本来就是每个线程块 128 个线程（4 个 warp），每个 warp 一行；改变每块的 warp 数只影响调度粒度和 occupancy，对这种纯访存的 kernel 影响不大，关键仍是每个 warp 内的合并访问。

## 小结

- [x] 写 KV 和嵌入查表都是"按下标搬运整行"，瓶颈是带宽：一个 warp 一行、`uint4` 向量化、合并访问。
- [x] PyTorch 扩展即时编译，stream 从 Python 传入；GPU 上用 kernel，其他情况退回参考实现。
- [x] kernel 用两个版本的 nvcc 编译，并在 CPU 模拟器上逐字节自检。
