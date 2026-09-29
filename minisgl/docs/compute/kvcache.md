# KV 池、page table 与 token pool

<p class="lead">引擎启动时一次性分配三块内存：存放所有请求 K、V 的 KV 池，记录"每个请求的每个 token 的 KV 存在池中哪个位置"的 page table，以及记录"每个请求的每个位置是哪个 token"的 token pool。后面的注意力后端、调度器、重叠调度都围绕这三块内存工作。这一章把它们建出来，并讲清 mini-sglang 一个独特的设计：page table 永远按 token 存位置，与 page size 无关。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Qwen3-0.6B 在 bf16 下，每个 token 的 KV 占多少字节？40 GiB 能放多少个 token？
    2. 分页（page size > 1）解决了什么问题？page size 越大越好吗？
    3. 为什么模型的输入 token 也要在 GPU 上保存一份（token pool），而不是每轮从 CPU 传过去？
    4. KV 池为什么要多分配一页？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 2 × 28 层 × 8 个 KV 头 × 128 维 × 2 字节 = 114688 字节（112 KiB）；40 GiB 能放约 37.4 万个 token（40 × 1024 × 1024 ÷ 112）。
    2. 以页为单位分配和管理，块表更短、分配释放的次数更少，也方便按页共享前缀、按页传输。page size 不是越大越好：最后一页的内部碎片更多，前缀缓存只能缓存完整的页、命中粒度更粗。
    3. 采样出的 token 直接写回 GPU 上的 token pool，下一轮的输入直接从那里取，CPU 不在关键路径上——这也是重叠调度的前提。
    4. 给 CUDA Graph 补齐 batch 用的 dummy 请求：它也会写 KV，多出来的最后一页专门给它写，不会踩到真实请求的数据。

**本章要写的文件**：`kvcache/base.py`、`kvcache/mha_pool.py`、`kvcache/naive_cache.py`、`kvcache/__init__.py`、`scheduler/table.py`。

## KV 池：一整块预分配的显存

推理时每个请求的 KV 缓存随生成不断增长，而且请求来来去去。如果每个请求各自申请显存，很快就会碎片化。所有现代推理引擎的做法都一样：**启动时把剩余显存一次性分配成一个大池子，按固定大小的"页"管理**，谁需要就分给谁几页，用完还回来。这就是 PagedAttention 的思想（原理见[推理系统手册的分页 KV Cache](serving://engine/paged-kv/)）。

@@code python/minisgl/kvcache/mha_pool.py:MHAKVCache@@

布局是 `[2, 层数, 页数, 页大小, 本 rank 的 KV 头数, head_dim]`：

- 第 0 维区分 K 和 V，`k_cache(i)`、`v_cache(i)` 取出第 i 层的 `[页数, 页大小, 头数, head_dim]`；
- 把"页数 × 页大小"展平后，第 j 行就是池中第 j 个 token 的位置，这就是 `_storage_shape`；
- `store_kv` 把本轮新算出的 K、V 按 `out_loc`（每个新 token 的池内位置）写进去。它由注意力后端在计算注意力之前调用。

@@code python/minisgl/kernel/torch_ops.py:store_cache@@

这是个纯粹的"按下标搬运"，GPU 上由第 19 章的自定义 kernel 完成：一个 warp 负责一个 token，每次搬 16 字节。

@@code examples/ch04_kvcache.py@@

@@output ch04_kvcache@@

Qwen3-0.6B 用了 GQA（16 个 q 头、8 个 KV 头），每个 token 的 KV 是 112 KiB。模型权重只有约 1.2 GB（bf16），一张 80 GB 的卡除去权重和激活，几乎全部显存都给了 KV 池——能同时服务多少请求、每个请求能多长，都由这个池子的大小决定。

**为什么多一页？** 引擎为 CUDA Graph 准备了一个 dummy 请求（第 18 章），它补齐 batch 时也会写 KV。多出来的最后一页专门给它写，不会踩到真实请求的数据。

## page table：按 token 存位置

@@diagram kv-layout KV 池、page table 与 token pool：两个请求的 KV 散落在不同的页里@@

page table 是一个二维 int32 张量，形状 `[max_running_req + 1, max_seq_len 向上对齐到 32]`，每行对应一个运行中的请求（最后一行给 dummy 请求），`page_table[r, j]` 是这个请求第 j 个 token 的 KV 在池中的位置。

mini-sglang 的一个特别之处：**无论 page size 是多少，page table 都按 token 存位置，而不是按页存页号**。上面的例子里 page size 是 16，请求 0 占了第 2 页和第 5 页，它的 page table 行是 `[32, 33, …, 47, 80, 81, 82, 83]`，而不是 `[2, 5]`。

这样做的好处是简单：

- 写 KV 时，`out_loc = page_table[行, cached_len:device_len]`，直接得到每个新 token 的位置，不用做"页号 × 页大小 + 页内偏移"的换算；
- FlashInfer 后端把 KV 池当成 page size = 1 来用，page table 的一行直接就是它需要的 `indices`；
- FlashAttention 需要页号，就每隔 page size 取一个位置再除以 page size：`page_table[r, ::page_size] // page_size`（第 17 章）。

代价是 page table 大了 page size 倍，但它只是 int32，最多几十 MB。对齐到 32 个 int32（128 字节）是为了让每一行的起始地址对齐，GPU 读取时效率更高。

page size 大于 1 的意义在于**前缀缓存和分配的粒度**：Radix Cache 只缓存完整的页（第 9 章），分配器每次分配一整页（第 8 章），页越大管理的开销越小，但最后一页平均浪费半页，前缀匹配也只能按页对齐。一些 kernel（例如 TensorRT-LLM 的注意力）只支持特定的 page size。

## token pool：输入也放在设备上

@@code python/minisgl/scheduler/table.py:TableManager@@

`TableManager` 管理 page table 的行：请求被接纳时分配一行，结束时归还。它还持有 **token pool**：与 page table 形状相同的 int32 张量，`token_pool[r, j]` 是这个请求第 j 个位置的 token id。

为什么要在设备上再存一份 token？因为本轮的输入是**上一轮采样的输出**。引擎采样得到 `next_tokens` 后，调度器直接在设备上执行 `token_pool[行, device_len] = next_tokens`；下一轮组 batch 时，`input_ids = token_pool[行, 位置]` 也在设备上完成。整个过程 CPU 不需要知道新 token 是什么。这正是第 11 章重叠调度的前提：GPU 在算第 N+1 轮时，CPU 可能还没收到第 N 轮的结果。

dummy 请求也从 token pool 取输入，所以 token pool 要初始化为合法的 token id（0），而不是随机值。

## 前缀缓存的接口

KV 池只管"存"，不管"哪个位置属于谁"。决定哪些位置可以复用的是**前缀缓存**，接口在 `kvcache/base.py`：

@@code python/minisgl/kvcache/base.py:BasePrefixCache@@

本章先实现最简单的 `NaivePrefixCache`：永远不命中，插入什么都不保留（请求结束后，它的页全部释放）。

@@code python/minisgl/kvcache/naive_cache.py:NaivePrefixCache@@

真正的 Radix Cache 在第 9 章实现。两者通过注册表 `SUPPORTED_CACHE_MANAGER` 按名字（`naive`、`radix`）创建。

!!! upstream "官方实现"
    - KV 池：@@upstream kvcache/mha_pool.py:MHAKVCache@@，写 KV 由自定义 CUDA kernel 完成（`kernel/csrc/jit/store.cu`）
    - page table 的创建与对齐：@@upstream engine/engine.py:Engine.__init__@@ 中 "Page table initialization" 一段，注释写着 "aligned to 128 bytes; store raw locations instead of pages"；`core.py` 的 `Context` 里也注明 "this table always treat page_size = 1"
    - token pool：@@upstream scheduler/table.py:TableManager@@

## 测试

@@code tests/test_ch04_kvcache.py:test_kv_pool_layout_and_store@@

第二个测试检查张量并行时的 KV 头复制：2 个 KV 头分给 4 个 rank，每个 rank 保存 1 个头（第 16 章）。

## 练习

1. Llama-3.1-8B（32 层、8 个 KV 头、head_dim 128）在 bf16 下每个 token 的 KV 多大？80 GB 的卡放完 16 GB 权重后，最多能容纳多少个 token？
2. 如果 page table 按页存页号，写 KV 时 `out_loc` 要怎么算？写出 PyTorch 表达式。
3. page size 为 16 时，一个 17 个 token 的请求占几页？平均每个请求浪费多少个 token 的空间？

??? success "参考答案"
    1. 2 × 32 × 8 × 128 × 2 = 131072 字节（128 KiB）；约 64 GiB 可用时能放约 52 万个 token（实际还要扣掉激活和 CUDA Graph 的显存）。
    2. `page_ids[pos // page_size] * page_size + pos % page_size`，其中 `pos` 是 `[cached_len, device_len)`。
    3. 2 页；最后一页平均用一半，平均浪费约 page_size / 2 = 8 个 token。

## 小结

- [x] KV 池是启动时一次性分配的大张量 `[2, 层, 页, 页大小, 头, 维]`，按页分给请求；多一页给 dummy 请求。
- [x] page table 每行对应一个请求，按 token 存 KV 位置，与 page size 无关；需要页号的后端自己换算。
- [x] token pool 在设备上保存每个请求每个位置的 token，采样结果直接写回，下一轮输入直接取出，CPU 不在关键路径上。
- [x] 前缀缓存决定哪些位置可以复用；先用永远不命中的 naive 版本。
