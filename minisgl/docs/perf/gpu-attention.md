# GPU 注意力后端：FlashInfer 与 FlashAttention

<p class="lead">参考后端用 Python 循环逐个请求算注意力，语义清楚但慢。GPU 上，mini-sglang 用两个专门的库：FlashInfer（decode 很强，prefill 也好用）和 FlashAttention 3（Hopper 上 prefill 最快）。它们的接口不同——FlashInfer 要先 plan 再 run，FlashAttention 直接吃页号表——但都能表达第 5 章那些元数据。这一章按官方的写法实现两个后端，并用同接口的 PyTorch 假实现在 CPU 上验证参数语义。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. FlashInfer 的 `plan` 和 `run` 分别做什么？为什么 plan 每个 batch 只做一次？
    2. mini-sglang 用 FlashInfer 时把 KV 池当成 page size = 1，`indices` 里存的是什么？`last_page_len` 为什么恒为 1？
    3. FlashAttention 的 `page_table` 里存的是 token 位置还是页号？怎样从 mini-sglang 的全局 page table 得到它？
    4. 在 CPU 上怎样验证一个只能在 GPU 上运行的后端？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `plan` 根据这个 batch 的元数据（每个请求的长度、页表）做调度规划：划分工作、分配临时缓冲区、把元数据拷到 GPU；`run` 在每一层按规划执行计算。同一个 batch 的所有层元数据相同，所以只 plan 一次（推迟到第一层时才做）。
    2. `indices` 里存的是每个 token 的 KV 在池中的位置（page size 为 1 时页号就是 token 位置），也就是 page table 的一行；每一页只有一个 token，最后一页总是满的，`last_page_len` 恒为 1。
    3. 页号。从全局 page table（按 token 存位置）每隔 page size 取一个位置，再除以 page size，就得到每一页的页号。
    4. 写一个接口与真实库相同的假实现：把两个库的参数语义翻译成同一个参考实现（Python + SDPA），在 CPU 上跑后端代码，结果要和参考后端完全一致。

**本章要写的文件**：`attention/fi.py`、`attention/fa.py`，以及测试用的 `tests/fakes/flashinfer/`、`tests/fakes/sgl_kernel/`。

## FlashInfer：plan 与 run

FlashInfer 为分页 KV 缓存提供了"批量 prefill"和"批量 decode"两种 wrapper。用法分两步：

- `plan(...)`：传入这个 batch 的形状信息——每个请求的 query 范围（`qo_indptr`）、每个请求的 KV 由哪些页组成（`kv_indptr` 划分 `kv_indices`）、最后一页用了多少（`last_page_len`）。FlashInfer 在 CPU 上据此算出工作划分（哪个线程块处理哪段 KV），异步拷到 GPU；
- `run(q, paged_kv_cache)`：每层调用一次，按 plan 好的方案计算。

28 层的形状完全相同，plan 只需要一次。mini-sglang 把 plan 推迟到第一层的 `forward` 里（`_initialize_metadata_once`），而不是在 `prepare_metadata` 里：

@@code python/minisgl/attention/fi.py:FlashInferBackend.forward@@

@@code python/minisgl/attention/fi.py:FlashInferBackend.prepare_metadata@@

几个值得注意的设计：

- **把 KV 池当成 page size = 1**：`forward` 里把 KV 池 view 成 `[token 数, 1, 头, 维]`，于是全局 page table 的一行就是 `kv_indices`，`last_page_len` 恒为 1。这样无论系统的 page size 是多少，FlashInfer 后端都不用做任何换算（第 4 章"page table 按 token 存位置"的好处）；
- **两个快路径**：纯 decode 时 `qo_indptr = [0, 1, …, bs]`；没有前缀命中的 prefill 时每个请求的 query 数等于 KV 长度，`qo_indptr` 直接复用 `kv_indptr`；
- **锁页内存与事件**：plan 会复用一块锁页内存做异步拷贝，所以在下一次 plan 修改它之前，要先等上一次拷贝完成（`last_event.synchronize()`）；
- **Tensor Core**：GQA 组大于等于 4 时，decode 实际上是一个个小矩阵乘，用 Tensor Core 版本的 decode kernel 更快。

@@code python/minisgl/attention/fi.py:FlashInferBackend._initialize_metadata_once@@

## FlashAttention：页号表

`sgl_kernel` 里的 `flash_attn_with_kvcache`（FlashAttention 3）直接读分页的 KV 池，需要：`page_table`（每个请求的**页号**）、`cache_seqlens`（每个请求的 KV 长度）、`cu_seqlens_q`（变长 query 的划分）。没有 plan 步骤。

@@code python/minisgl/attention/fa.py:FlashAttentionBackend.prepare_metadata@@

全局 page table 按 token 存位置，每隔 page size 取一个位置就是每页的起点，再除以 page size 就是页号：`page_table[r, :max_len:page_size] // page_size`。

## 在 CPU 上验证

这两个库都只能在 NVIDIA GPU 上运行。为了验证后端代码传给它们的参数是对的，我们为每个库写了一个"同接口的假实现"，放在 `tests/fakes/` 下，测试时加到 import 路径最前面：

@@code tests/fakes/flashinfer/__init__.py:BatchPrefillWithPagedKVCacheWrapper@@

@@code tests/fakes/sgl_kernel/flash_attn.py:flash_attn_with_kvcache@@

两者都把参数翻译成"每个请求的 query 范围、KV 页列表、KV 长度"，交给同一个参考实现：

@@code tests/fakes/_fake_ref.py:paged_attention@@

这样，后端代码里任何一个参数传错（比如 `indptr` 少了开头的 0、页号没有除以 page size、decode 时 `qo_indptr` 写错），假实现算出的结果就会与参考后端不同，测试失败。这套假实现本身也是理解 FlashInfer 和 FlashAttention 接口语义的最短路径。

@@code examples/ch17_gpu_attention.py@@

@@output ch17_gpu_attention@@

两个请求：请求 0 命中前 3 个 token、本轮算 2 个；请求 1 前 5 个在缓存里、本轮算 1 个。FlashInfer 拿到的是 token 级的 `kv_indices`（16～20 和 40～45），`last_page_len` 全为 1；FlashAttention 在 page size 为 4 时拿到的是页号（位置 16 在第 4 页，位置 40 在第 10 页）。两个后端的 logits 与参考后端完全相同。

!!! upstream "官方实现"
    - @@upstream attention/fi.py:FlashInferBackend@@
    - @@upstream attention/fa.py:FlashAttentionBackend@@
    - 官方还有 TensorRT-LLM 的后端 `attention/trtllm.py`（Blackwell 上默认使用，只支持 page size 16、32、64），我们没有实现。

    我们的两个后端与官方的逻辑一致，只把 `pin_memory=True` 和 `torch.cuda.Event` 换成了设备抽象里的函数。

## 测试

@@code tests/test_ch17_gpu_backends.py:test_backend_logits_match_torch_backend@@

`tests/test_ch17_gpu_backends.py` 还检查了"每个 batch 只 plan 一次"，并用 FlashInfer（page size 1）、FlashAttention（page size 4）、`fa,fi` 组合三种配置，在重叠调度和分块 prefill 下端到端地与 Hugging Face 对比。

## 练习

1. 如果要让 FlashInfer 使用真正的 page size（例如 16）而不是展平成 1，`prepare_metadata` 要怎么改？`last_page_len` 怎么算？
2. 在假 FlashInfer 里故意把 `qo_indptr` 的解释改错（比如忽略开头的 0），哪个测试会失败？
3. 为什么 CUDA Graph 只和 decode 后端打交道，而 FlashAttention 的 prefill 不需要 CUDA Graph？

??? success "参考答案"
    1. `kv_indices` 改为页号：`page_table[r, :device_len:ps] // ps`；`kv_indptr` 按页数累加（`ceil(device_len / ps)`）；`last_page_len = device_len - (页数 - 1) * ps`；`forward` 里不再 view 成 page size 1，而是直接传 `[页数, ps, 头, 维]` 的 KV 池。
    2. `test_backend_logits_match_torch_backend[fi]` 和所有用 `fi` 的端到端测试都会失败——这正是假实现的价值。
    3. prefill 的 batch 形状（token 数、各请求长度）千变万化，无法为每种形状录制 graph；prefill 的计算量大，kernel 启动开销占比小，也不需要。decode 的形状只由批大小决定，且每步计算很少，启动开销占比大。

## 小结

- [x] FlashInfer：plan（每个 batch 一次，推迟到第一层）+ run（每层）；mini-sglang 把 KV 池当成 page size 1，page table 的一行就是 `kv_indices`。
- [x] FlashAttention：直接吃页号表、KV 长度和 `cu_seqlens_q`，页号从全局 page table 每隔 page size 取一个再除以 page size。
- [x] 同接口的假实现把两个库的参数语义翻译成同一个参考实现，在 CPU 上验证后端代码，结果与参考后端完全一致。
