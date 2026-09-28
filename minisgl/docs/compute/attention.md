# 注意力后端与参考实现

<p class="lead">注意力是引擎里唯一需要"知道 batch 结构"的计算：每个请求长度不同、KV 散落在池子的不同页里、有的请求前缀已经缓存。mini-sglang 把这些复杂性收进"注意力后端"：每个 batch 调用一次 <code>prepare_metadata</code> 把请求信息整理成 kernel 需要的格式，每层调用一次 <code>forward</code> 写 KV、算注意力。官方有 FlashInfer、FlashAttention、TensorRT-LLM 三个 GPU 后端；这一章先写一个 PyTorch 参考后端，把语义讲透，第 17 章再接 GPU 后端。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `cu_seqlens_q = [0, 6, 9, 10]` 表示什么？
    2. 一个请求 KV 总长 7、本轮 3 个 query，因果掩码怎样对齐？第一个 query 能看到哪些 key？
    3. 为什么元数据每个 batch 准备一次，而不是每层准备一次？
    4. 写 KV 必须在算注意力之前还是之后？为什么？

**本章要写的文件**：`attention/base.py`、`attention/utils.py`、`attention/torch_backend.py`、`attention/__init__.py`。

## 接口

@@code python/minisgl/attention/base.py:BaseAttnBackend@@

- `prepare_metadata(batch)`：调度器在每个 batch 开始前调用一次，结果放在 `batch.attn_metadata`。模型有 28 层，每层的注意力形状完全相同，元数据只需准备一次；GPU 后端还会在这里把数据从 CPU 拷到 GPU、或者做 FlashInfer 的 plan。
- `forward(q, k, v, layer_id, batch)`：每层调用。**先写后读**：把本轮新 token 的 k、v 写进 KV 池，再从池子里读出每个请求的全部 KV 算注意力。顺序不能反，因为本轮的 query 也要看到本轮的 key（因果掩码的对角线）。
- 最后三个方法服务于 CUDA Graph，第 18 章再讲。

元数据还要提供 `get_last_indices(bs)`：prefill 时每个请求最后一个 token 在展平后的 q 里的下标，给 LM head 用（第 2 章）。

## 变长、分页、带前缀的注意力

batch 里的张量是各请求首尾相接的一维数组。用一个例子说明注意力需要哪些信息：请求 A 从头 prefill 6 个 token；请求 B 前 4 个 token 命中前缀缓存，本轮算 3 个；请求 C 已经有 8 个 token，本轮 decode 1 个。

- **每个请求本轮的 query 在哪**：`cu_seqlens_q = [0, 6, 9, 10]`（cumulative sequence lengths），第 i 个请求是 `q[cu[i]:cu[i+1]]`；
- **每个请求的 KV 有多长**：`cache_seqlens = [6, 7, 9]`，即各请求的 `device_len`；
- **每个请求的 KV 在池中哪里**：page table 的对应行，取前 `cache_seqlens[i]` 个位置；
- **因果掩码怎么对齐**：本轮的 n 个 query 是序列的**最后** n 个位置。KV 长 7、query 3 个时，第一个 query 的绝对位置是 4，能看到位置 0～4 的 key。这叫"右下角对齐"：把 `[n, kv_len]` 的掩码矩阵画出来，它的对角线落在右下角。

参考后端就是把这段描述直接翻译成代码：

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.forward@@

对每个请求：按 page table 从池中取出它的全部 K、V，构造右下角对齐的因果掩码，调用 `scaled_dot_product_attention`（`enable_gqa=True` 让 16 个 q 头共享 8 个 KV 头）。这是一个 Python 循环，速度不快，但完全没有歧义，CPU 上也能跑。

元数据的准备：

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_metadata@@

page table 取每个请求所在行的前 `max_seqlen_k` 列，拼成 `[bs, max_seqlen_k]`，这与官方 FlashAttention 后端的做法相同。用的是 `batch.padded_reqs` 而不是 `batch.reqs`：CUDA Graph 补齐进来的 dummy 请求也要有元数据。

把上面的例子跑一遍，和"不分页、不变长的完整因果注意力"对比：

@@code examples/ch05_attention.py@@

@@output ch05_attention@@

几点观察：

- 请求 B 的 KV 在池中不连续：前缀缓存命中的 4 个 token 在 40～43，本轮新算的 3 个在 20～22。`out_loc` 只包含本轮新 token 的位置。
- page table 按 `max_seqlen_k = 9` 截取，短的请求后面是无关的 0，只要用 `cache_seqlens` 截断就不会读到。
- `get_last_indices` 给出 `[5, 8, 9]`：三个请求最后一个 query 在展平后 q 中的下标。
- 与逐个请求的完整因果注意力相比，误差在 float32 舍入的量级。

## 创建后端：单个或组合

@@code python/minisgl/attention/__init__.py:create_attention_backend@@

后端通过注册表按名字创建。`"fa,fi"` 表示 prefill 用 FlashAttention、decode 用 FlashInfer，由 `HybridBackend` 按 `batch.is_prefill` 转发：

@@code python/minisgl/attention/base.py:HybridBackend@@

CUDA Graph 只捕获 decode，所以三个 graph 相关的方法只转发给 decode 后端。`--attn auto` 时，引擎在 CPU 上选 `torch`，在 Hopper（SM90）上选 `fa,fi`，其他 GPU 上选 `fi`。

!!! diff "与官方的差异：参考后端"
    官方没有 PyTorch 后端（SGLang 正式版有一个 `torch_native` 后端，作用类似）。我们的 `torch` 后端元数据格式与官方的 FlashAttention 后端相同，CUDA Graph 的三个钩子也按 FlashAttention 后端的方式实现，所以第 18 章可以在 CPU 上验证 CUDA Graph 的数据流。官方在 Blackwell（SM100）上默认选 `trtllm` 后端，我们没有实现它。

!!! upstream "官方实现"
    - 接口与组合后端：@@upstream attention/base.py:BaseAttnBackend@@、@@upstream attention/base.py:HybridBackend@@
    - 元数据格式参照：@@upstream attention/fa.py:FlashAttentionBackend.prepare_metadata@@
    - 自动选择后端：@@upstream engine/engine.py:_adjust_config@@

## 测试

@@code tests/test_ch05_attention.py:test_torch_backend_mixed_batch_with_prefix_hits@@

## 练习

1. 参考后端对每个请求都构造一个 `[n, kv_len]` 的掩码。decode 时 n = 1，掩码全为真。给 decode 写一条快路径：把所有请求的 KV 按最大长度补齐成 `[bs, max_len, H, D]`，用一次 batched SDPA 算完。
2. 如果忘了在 `forward` 里先调用 `store_kv`，prefill 的结果会怎样？decode 呢？
3. `cu_seqlens_q` 在"没有前缀命中的 prefill"时等于什么？在"纯 decode"时等于什么？（提示：看第 17 章 FlashInfer 后端里的两个快路径。）

??? success "参考答案"
    1. 用 `cache_seqlens` 构造 `[bs, 1, max_len]` 的长度掩码（`arange(max_len) < len`），KV 用 `page_table[:, :max_len]` 一次 gather 出来；注意补齐部分要被掩掉。
    2. prefill 时本轮的 key 读到的是池中的旧数据（未初始化或别的请求的），结果错误；decode 时最后一个位置（当前 token）的 KV 是旧的，同样错误。只有"先写后读"，本轮 token 才能看到自己。
    3. 没有前缀命中时每个请求的 query 数等于 KV 长度，`cu_seqlens_q == cu_seqlens_k`；纯 decode 时每个请求 1 个 query，`cu_seqlens_q = [0, 1, 2, ..., bs]`。

## 小结

- [x] 注意力后端的两件事：每个 batch 一次 `prepare_metadata`，每层一次 `forward`（先写 KV、再算注意力）。
- [x] 元数据描述变长和分页：`cu_seqlens_q` 划分 query，`cache_seqlens` 给出 KV 长度，page table 给出 KV 位置；因果掩码右下角对齐。
- [x] 参考后端用 Python 循环 + SDPA 实现，语义清楚、CPU 可跑，是验证 GPU 后端的基准。
- [x] 后端按名字创建，可以 prefill 和 decode 用不同的后端。
