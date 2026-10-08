# Custom CUDA kernels

<p class="lead">mini-sglang writes only three CUDA kernels of its own: writing new K/V into the KV pool (<code>store_cache</code>), the vocabulary-parallel embedding lookup (<code>indexing</code>), and <code>fast_compare_key</code> for comparing two stretches of tokens in the radix tree (C++, on the CPU). The first two are pure "move whole rows by index" with simple logic, but they run every layer of every step, so they are worth writing as efficient kernels. This chapter writes their CUDA versions, compiles them with two versions of nvcc, and self-checks them on the CUDA handbook's CPU simulator.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Is the KV-cache write kernel compute-bound or memory-bound? How fast can it be in theory?
    2. Why "one warp per row" rather than "one thread per row"?
    3. What does copying with `uint4` buy? When can it not be used?
    4. PyTorch's `k_cache[idx] = k` works on a GPU too. Why write your own?

??? success "Answers (try it yourself first, then expand)"
    1. Memory-bound: it only moves rows of K and V from one place to another by index, with no computation. The theoretical best is the bytes read and written divided by memory bandwidth.
    2. A row has hundreds to thousands of elements: a warp's 32 threads read neighbouring elements of the same row, so the accesses coalesce and saturate the bandwidth. With one thread per row, neighbouring threads touch different rows, nothing coalesces, and with few rows there are too few threads.
    3. One instruction moves 16 bytes, cutting the number of memory instructions to a quarter or less and keeping more data in flight. It requires the addresses to be 16-byte aligned and the row's byte count to be a multiple of 16; otherwise fall back to a smaller type or handle the tail separately.
    4. PyTorch's advanced indexing launches a generic kernel with extra overhead (index checks, metadata handling) and can introduce synchronization; it also cannot be fused with anything else. A hand-written kernel is specialized to this shape, can go into a CUDA Graph, and shares one pattern with the embedding lookup.

**Files you will write**: `kernel/csrc/kv_kernels.cuh`, `kernel/csrc/ext.cu`, `kernel/cuda_ext.py`, and the dispatch in `kernel/__init__.py`; plus the self-check program `tests/cuda/test_kv_kernels.cu`.

## Moving whole rows by index {#按下标搬运整行}

Writing the KV cache means: for each new token `i` this step, copy `k[i]` (one row of `heads × head_dim` elements) to `k_cache[out_loc[i]]`, and the same for `v`. The embedding lookup is the reverse: `out[i] = weight[ids[i]]`. Neither computes anything, so the bottleneck is purely memory bandwidth: at best the bytes moved divided by the bandwidth.

@@code python/minisgl/kernel/csrc/kv_kernels.cuh@@

The design (the background is in the [CUDA handbook's memory hierarchy and access optimization](cuda://basics/memory/)):

- **one warp per row**: one row of K for Qwen3-0.6B is 8 × 128 × 2 = 2 KB. One thread moving 2 KB is far too slow, while a warp's 32 threads each moving a neighbouring 16 bytes move 512 bytes at once, with **coalesced** accesses;
- **vectorization**: copy in `uint4` units (16 bytes), one instruction per 16 bytes. This needs both the row's byte count and the pointers to be multiples of 16; otherwise it falls back to copying by element size (2 or 4 bytes). The host side chooses based on the shape and alignment, and the kernel itself is templated on the type `V`;
- **the vocabulary-parallel mask**: a token outside this rank's vocabulary slice gets a whole row of zeros (`V{}`), which pairs with the all-reduce that follows.

PyTorch's `k_cache[idx] = k` works on a GPU, of course, but it is a generic `index_put` kernel that has to handle any shape and stride; a dedicated kernel generates code for a fixed row size, vectorizes the copy and does K and V in one kernel, landing much closer to the bandwidth limit. Upstream also adds a switch for PDL (Programmatic Dependent Launch, supported from Hopper on) so it can partly overlap with the kernels around it.

## Hooking into PyTorch {#接进-pytorch}

@@code python/minisgl/kernel/csrc/ext.cu@@

The extension is compiled just in time on first use by `torch.utils.cpp_extension.load`. The CUDA stream is passed in from Python (`torch.cuda.current_stream().cuda_stream`), so the extension depends only on PyTorch's CPU headers and the CUDA runtime, which also lets us compile-check it on a development machine without a CUDA build of PyTorch.

@@code python/minisgl/kernel/__init__.py:store_cache@@

On a GPU with the extension compiled it uses the custom kernel; otherwise (CPU, no nvcc) it falls back to the PyTorch reference implementation.

!!! diff "Difference from upstream: tvm-ffi versus torch extensions"
    Upstream uses [tvm-ffi](https://github.com/apache/tvm-ffi) for JIT compilation and Python bindings: the kernel's row size, thread count and so on are C++ template parameters, and a specialized version is generated on the fly for the actual shape (`load_jit` in `kernel/utils.py`). We use the more familiar PyTorch extension with the row size as a runtime argument. The kernel's ideas (one warp per row, vectorized copies, the vocabulary mask) are the same.

## Verifying on a machine with no GPU {#在没有-gpu-的机器上验证}

The self-check program tests both kernels' vectorized and element-wise paths on random data and compares byte for byte against a CPU reference:

@@code tests/cuda/test_kv_kernels.cu@@

It is compiled with both CUDA 13.4 and 12.9 nvcc and then run on the [CUDA handbook's](cuda://) CPU simulator, which gives every CUDA thread a coroutine and really executes the parallel logic within a warp:

@@code examples/ch19_kernels.py@@

@@output ch19_kernels@@

The PyTorch extension `ext.cu` is compile-checked with both nvcc versions too (`tools/check.py`), but it depends on a CUDA build of PyTorch at runtime and can only execute on a GPU.

!!! upstream "The official implementation"
    - writing KV: `kernel/csrc/jit/store.cu`, with the Python side at @@upstream kernel/store.py:store_cache@@
    - the lookup: `kernel/csrc/jit/index.cu`, with the Python side at @@upstream kernel/index.py:indexing@@ (when the row's byte count is a multiple of 2048 or 1024, 4 or 2 warps share a row)
    - the radix tree's comparison: `kernel/csrc/src/radix.cpp` (`std::mismatch` on the CPU)

## Tests {#测试}

@@code tests/test_ch19_kernels.py:test_cuda_kernels_on_cpu_emulator@@

!!! interview "How to explain it"
    On custom kernels: writing the KV cache and the embedding lookup are both "move whole rows by index", the bottleneck is bandwidth, and the theoretical best is bytes moved divided by memory bandwidth. One warp handles a row so its 32 threads read contiguous addresses and coalesce (one thread per row would put a whole row between neighbouring threads' addresses); `uint4` moves 16 bytes at a time and cuts the memory instructions, given 16-byte alignment and a divisible row length. PyTorch's `k_cache[idx] = k` is a generic `index_put` that handles any shape and stride and may launch several kernels, while a dedicated kernel generated for a fixed row size is faster and can be captured into a CUDA Graph. Without a GPU, a CPU simulator self-checks it byte for byte.

## Exercises {#练习}

1. On an H100 (about 3.35 TB/s of memory bandwidth), for one Qwen3-0.6B decode step at batch size 64, how many bytes does the per-layer KV write move? What is the theoretical time? Is this kernel worth putting in a CUDA Graph?
2. Upstream's `indexing` has several warps share a row when the row is large. Add that optimization to `embedding_kernel` and verify it on the simulator.
3. Rewrite `store_kv_kernel` as "one block handles several rows, one warp per row". How does that differ from the current "4 warps per block"?

??? success "Answers"
    1. Each token's K and V are 2 KB each, so 64 tokens are 256 KB (reading k and v and writing the pool makes roughly 512 KB of traffic), about 0.15 µs, which is far below a kernel launch's few microseconds. So it has to go into a CUDA Graph, or the launch overhead would be tens of times the actual work.
    2. Have `num_splits` warps share a row: `warp_id / num_splits` is the row and `warp_id % num_splits` is the segment within it, each segment covering `row / num_splits` elements.
    3. The current version already uses 128 threads per block (4 warps) with one warp per row; changing the warps per block only affects scheduling granularity and occupancy, which matters little for a purely memory-bound kernel where coalescing within each warp is what counts.

## Summary {#小结}

- [x] Writing KV and the embedding lookup are both "move whole rows by index" and bandwidth-bound: one warp per row, `uint4` vectorization, coalesced accesses.
- [x] The PyTorch extension compiles just in time with the stream passed in from Python; the kernel is used on a GPU and the reference implementation everywhere else.
- [x] The kernels are compiled with two nvcc versions and self-checked byte for byte on the CPU simulator.
