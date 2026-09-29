# MoE 与 fused MoE

<p class="lead">Qwen3-30B-A3B 这样的 MoE 模型，每层有上百个专家，每个 token 只激活其中几个。难点在于：一个 batch 里的 token 各自去往不同的专家，按专家逐个计算会启动上百个小矩阵乘。fused MoE 把所有"(token, 专家) 对"按专家排好序，每个专家的那一段补齐到块大小的整数倍，于是整层 MoE 只需两次 kernel 启动。这一章实现 MoE 层、参考后端和一个 Triton 写的 fused MoE，并在 CPU 上用 Triton 解释器验证。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. MoE 的路由是怎么算的？`norm_topk_prob` 是什么意思？
    2. 为什么把 (token, 专家) 对按专家排序之后，每个专家的段要补齐到 `BLOCK_M` 的整数倍？
    3. 张量并行时，MoE 的专家权重怎么切？需要什么通信？
    4. 在没有 GPU 的机器上怎么运行 Triton kernel？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 路由层算出每个专家的分数，softmax 之后选 top-k 个专家和权重；`norm_topk_prob` 表示把选出的 k 个权重重新归一化（除以它们的和），让它们加起来是 1。
    2. fused MoE kernel 里每个输出块（`BLOCK_M` 行）只能属于一个专家、用这个专家的权重做矩阵乘；把每个专家的段补齐到 `BLOCK_M` 的整数倍，块才不会跨两个专家。
    3. 每个专家的权重都按中间维切：gate / up 按列切、down 按行切，每个 rank 持有所有专家的一部分；路由层不切（每个 rank 都算完整的路由）。每层最后一次 all-reduce 把部分和加起来。
    4. 用 Triton 的解释器模式（`TRITON_INTERPRET=1`）：kernel 在 CPU 上用 numpy 以相同的语义逐块运行，用来验证正确性（性能没有意义）。

**本章要写的文件**：`layers/moe.py`、`moe/base.py`、`moe/torch_backend.py`、`moe/fused.py`、`moe/__init__.py`，以及 `models/utils.py` 中的 `MoEMLP`。

## MoE 层

@@code python/minisgl/models/utils.py:MoEMLP@@

`gate` 是一个不切分的线性层（每个 rank 一份完整的），算出每个 token 对每个专家的打分。`MoELayer` 持有所有专家打包后的权重，计算交给 Context 里的 MoE 后端：

@@code python/minisgl/layers/moe.py:MoELayer@@

`gate_up_proj` 形状 `[专家数, 2 × 中间维, hidden]`，每个专家的 gate 和 up 拼在一起（与稠密 MLP 相同）。张量并行时，每个专家的中间维都被切开，和稠密 MLP 一样：先列并行、再行并行，最后一次 all-reduce。权重加载器把 checkpoint 里分开存的 `experts.0.gate_proj`、`experts.1.gate_proj`……读齐后打包成三维张量（第 3 章）。

## 路由

@@code python/minisgl/moe/base.py:select_experts@@

softmax 后取前 k 个专家；`norm_topk_prob=True`（Qwen3-MoE）时把这 k 个权重重新归一化到和为 1。

## 参考后端

@@code python/minisgl/moe/torch_backend.py:TorchMoeBackend@@

按专家分组：对每个被选中的专家，挑出路由到它的 token，做一次 SwiGLU MLP，乘上路由权重，用 `index_add_` 累加回各自的 token。直白，但专家有多少个，就有多少轮小矩阵乘。

## fused MoE

@@diagram fused-moe 按专家排序、补齐，再做两次 fused GEMM@@

@@code python/minisgl/moe/fused.py:moe_align_block_size@@

第一步把所有 (token, 专家) 对按专家排序，每个专家的段补齐到 `BLOCK_M` 的倍数，补齐的位置填一个占位值（等于对数）。于是每 `BLOCK_M` 个连续的对属于同一个专家，一个线程块可以处理这 `BLOCK_M` 个 token 与**一个专家**权重的矩阵乘。

@@code python/minisgl/moe/fused.py:fused_moe_kernel@@

kernel 的每个程序实例负责输出的一个 `[BLOCK_M, BLOCK_N]` 块：读这个块对应的专家号，按排序后的下标取出 `BLOCK_M` 行输入（第一次 GEMM 时一个 token 对应 `TOP_K` 个对，所以行号是 `对编号 // TOP_K`），与该专家的权重做分块矩阵乘；占位符的行被掩掉。第二次 GEMM 时顺便乘上路由权重。

@@code python/minisgl/moe/fused.py:FusedMoeBackend.forward@@

整层 MoE：排序（PyTorch）→ 第一次 fused GEMM（`w1`）→ SiLU × up → 第二次 fused GEMM（`w2`，乘路由权重）→ 按 token 把 k 个结果相加。

## 在 CPU 上运行 Triton

Triton 提供解释器模式（`TRITON_INTERPRET=1`）：kernel 用 NumPy 在 CPU 上逐个程序实例地解释执行，语义与 GPU 上相同，只是很慢。它必须在导入 `triton` 之前设置。

@@code examples/ch20_moe.py@@

@@output ch20_moe@@

5 个 token、每个选 2 个专家，共 10 个对。排序后专家 1 的两个对（第 3、4 个）组成第一个块，补两个占位符；专家 3 只有一个对，也占满一个块……共 6 个块。块越大，补齐浪费越多；块越小，矩阵乘效率越低。37 个 token 的 fused MoE 与参考实现的误差在 float32 舍入的量级。

## 端到端

用随机权重构造一个两层、8 个专家的小 Qwen3-MoE 模型，与 Hugging Face 的实现对比（CPU 上默认用参考后端）：

@@code tests/test_ch20_moe.py:test_tiny_qwen3_moe_matches_hf@@

!!! upstream "官方实现"
    - @@upstream layers/moe.py:MoELayer@@、@@upstream models/qwen3_moe.py@@
    - fused MoE：@@upstream moe/fused.py@@ 与 `kernel/triton/fused_moe.py`。路由（`topk_softmax`）和排序补齐（`moe_align_block_size`）用 sgl_kernel 里的 CUDA kernel；块大小按形状二选一：token 数不超过专家数时用小块（16 × 32 × 64），否则用大块（64 × 64 × 32）——正是练习 1 讨论的问题

!!! diff "与官方的差异"
    我们的 Triton kernel 是简化版：块大小固定（`BLOCK_M=16`、`BLOCK_N=32`、`BLOCK_K=32`），没有按形状选择，没有官方 kernel 里的 `GROUP_SIZE_M` 分组（提高 L2 命中率），排序补齐用 PyTorch 而不是 CUDA kernel。参考后端 `torch` 是我们加的，CPU 上默认使用。

## 测试

@@code tests/test_ch20_moe.py:test_fused_triton_moe_in_interpreter@@

`tests/test_ch20_moe.py` 还用最直白的"逐 token、逐专家"循环验证了参考后端，以及第 2 章提到的 Llama 3 长上下文 RoPE（同样用随机权重的小模型与 Hugging Face 对比）。

## 练习

1. `BLOCK_M=64`、128 个专家、每步 decode 只有 32 个 token（每个选 8 个专家）时，补齐之后有多少个对？浪费的比例是多少？这对 decode 意味着什么？
2. 专家并行（EP）把不同的专家放在不同的 rank 上，每个 token 要被送到它的专家所在的 rank。它与本章的张量并行切法相比，通信模式有什么不同？（参考[推理系统手册的专家并行一章](serving://distributed/expert-parallel/)。）
3. 把 SiLU × up 融合进第一次 GEMM 的输出：kernel 需要同时算出 gate 和 up 两个块。怎样安排 `offs_n` 才能让一个程序实例拿到配对的 gate 和 up 列？

??? success "参考答案"
    1. 256 个对最多分布在 128 个专家上，大多数专家只有 1～3 个对，每个都要补到 64：最坏约 128 × 64 = 8192 行，有效只有 256 行，浪费约 97%。decode 时 MoE 的 fused kernel 效率很低，所以 decode 常用更小的块、或专门为小批量设计的 kernel（例如按专家做 GEMV）。
    2. 张量并行每个 rank 都有全部专家的一部分，通信是一次 all-reduce（数据量与 token 数 × hidden 成正比）；专家并行每个 rank 有一部分专家的全部，需要 all-to-all 把 token 分发到专家所在的 rank、算完再收回来，数据量取决于路由分布，还可能出现负载不均。
    3. 让程序实例负责中间维上的一段 `[n, n + BLOCK_N)`，同时加载 gate 列 `[n, n+BLOCK_N)` 和 up 列 `[I + n, I + n + BLOCK_N)`，做两次 `tl.dot`，再 `silu(gate) * up` 后写出 `[BLOCK_M, BLOCK_N]`。

## 小结

- [x] MoE 层：不切分的路由层 + 打包成三维张量的专家权重；张量并行时每个专家按中间维切，一次 all-reduce。
- [x] 路由：softmax → top-k → 可选重新归一化。
- [x] fused MoE：按专家排序并补齐到块大小，每个输出块只属于一个专家，整层两次 kernel 启动。
- [x] Triton 解释器让 kernel 在 CPU 上以相同的语义运行，用来验证正确性。
