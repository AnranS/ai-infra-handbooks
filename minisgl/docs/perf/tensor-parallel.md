# 张量并行

<p class="lead">模型放不进一张卡，或者一张卡的带宽撑不起想要的延迟，就要把每一层的权重切开分到多张卡上：这就是张量并行（TP）。前面各章的线性层、嵌入层、KV 池里那些 <code>get_tp_info().size</code> 在这一章终于派上用场。我们在 CPU 上用 gloo 做通信，TP=2、TP=4 的输出与单卡的 Hugging Face 完全一致。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 decoder 层做张量并行，需要几次 all-reduce？分别在哪里？
    2. 列并行和行并行分别切权重的哪一维？为什么 MLP 要"先列后行"？
    3. Qwen3-0.6B 有 8 个 KV 头，TP=16 时每个 rank 怎么分 KV 头？
    4. 词表并行的嵌入层用 all-reduce，输出层用 all-gather，为什么不同？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 两次：注意力的 `o_proj`（行并行）之后一次，MLP 的 `down_proj`（行并行）之后一次。
    2. 列并行按输出维切（每个 rank 算一部分输出），行并行按输入维切（每个 rank 得到部分和，需要 all-reduce）。先列后行：列并行输出的正好是行并行需要的那部分输入，中间的激活逐元素计算，不需要通信，整个 MLP 只要最后一次 all-reduce。
    3. 8 个 KV 头不够分给 16 个 rank：每个 KV 头复制到 2 个 rank 上，每个 rank 放 1 个 KV 头（和它对应的 query 头）。
    4. 嵌入层按词表切分：每个 rank 只查自己那段词表里的 token，查不到的输出 0，all-reduce 求和就得到完整的嵌入；输出层按词表切分时每个 rank 算出一部分词表的 logits，要 all-gather 拼成完整的词表分布才能采样。

**本章要回顾的文件**：`layers/linear.py`、`layers/embedding.py`、`models/weight.py` 中的 `shard_tensor`、`distributed/`、`engine/engine.py` 中的 `_init_communication`。

@@tree@@

这一步的 main：`examples/ch16_tp.py`——它只用到上面这些文件；`python tools/steps.py check` 会逐章搭出这棵树、跑这个 main。

@@video tp 动画：张量并行怎样切一层、通信几次（约 1.5 分钟）@@

## Megatron 式切分

一个 decoder 层有两个"矩阵乘夹着逐元素运算"的结构：注意力（`qkv_proj` → 注意力 → `o_proj`）和 MLP（`gate_up_proj` → SiLU × up → `down_proj`）。Megatron-LM 的切法（原理见[推理系统手册的张量并行一章](serving://distributed/tensor-parallel/)）：

- **第一个矩阵按输出维切**（列并行）：每个 rank 算出一部分输出通道。注意力按**头**切：每个 rank 负责若干个头，头与头之间的注意力计算互不相关，所以注意力本身不需要任何通信；MLP 按中间维切，SiLU × up 是逐元素的，也不需要通信。
- **第二个矩阵按输入维切**（行并行）：每个 rank 用自己那部分中间结果乘自己那部分权重，得到完整形状的**部分和**，最后一次 **all-reduce** 相加。

所以每层两次 all-reduce（`o_proj` 和 `down_proj` 之后），其余全是本地计算。

@@diagram tp-sharding 张量并行下一个 decoder 层的数据流@@

@@code python/minisgl/layers/linear.py:LinearQKVMerged@@

@@code python/minisgl/layers/linear.py:LinearRowParallel@@

合并的 `qkv_proj` 在切分时要小心：不能把合并后的大矩阵直接切成两半（那会把 q 的全部分给 rank 0、k 和 v 分给 rank 1），而要先分别切 q、k、v，再在每个 rank 上拼接。流式加载器正是这么做的：先 `shard_tensor`，再合并（第 3 章）。

## KV 头不够分时复制

@@code python/minisgl/models/weight.py:shard_tensor@@

GQA 模型的 KV 头比 q 头少。Qwen3-0.6B 有 16 个 q 头、8 个 KV 头：TP=2 时每个 rank 8 个 q 头、4 个 KV 头；TP=16 时每个 rank 1 个 q 头，而 KV 头只有 8 个——每两个 rank 共用同一个 KV 头（`div_even(..., allow_replicate=True)` 返回 1），各自保存一份副本。这会浪费一些显存，但保证了注意力仍然不需要通信。

## 词表并行

嵌入层的词表按行切，每个 rank 保存一段；查表时不在本段的 token 填 0，然后 **all-reduce** 相加——每个 token 恰好在一个 rank 上查到真实的向量。输出层同样按词表切，每个 rank 算出自己那段词表的 logits；采样需要完整的分布，所以用 **all-gather** 把各段拼起来（第 2 章的 `ParallelLMHead.forward`）。前者是"各 rank 的结果相加"，后者是"各 rank 的结果拼接"。

## 运行

@@code examples/ch16_tp.py@@

@@output ch16_tp@@

- 权重形状：TP=2 时 q_proj、k_proj、gate_proj 按输出维减半，o_proj、down_proj 按输入维减半，词表减半，norm 不切；TP=16 时 k_proj 仍有 128 行——一个完整的 KV 头（复制）。
- 两个 rank 的 `qkv_proj` 都是 (8 + 2 × 4) × 128 = 2048 行；每个 rank 的 KV 池只存 4 个 KV 头。
- 一次前向 57 次 all-reduce（28 层 × 2 + 嵌入层 1）、1 次 all-gather（输出层），与分析一致。
- logits 与单卡的最大误差 1e-5 量级：all-reduce 改变了浮点加法的顺序，结果不再逐位相同，但贪心解码的 token 完全一致（本章的测试在服务层面验证了 TP=2、TP=4）。

## 通信的实现

@@code python/minisgl/distributed/impl.py:TorchDistributedImpl@@

`DistributedCommunicator` 通过插件列表选择实现，默认用 `torch.distributed`：GPU 上走 NCCL，CPU 上走 gloo。gloo 不支持 `all_gather_into_tensor`，我们在 CPU 上改用 `all_gather` 写进输出张量的各个切片。

引擎初始化时建立进程组：

@@code python/minisgl/engine/engine.py:Engine._init_communication@@

GPU 上另建一个 gloo 组，专门用于 CPU 上的控制信息（第 14 章的"广播条数"、启动时的显存同步）。

!!! diff "与官方的差异：PyNCCL"
    官方默认（`use_pynccl=True`）不用 `torch.distributed` 做 GPU 通信，而是通过 tvm-ffi 直接调用 NCCL（`kernel/csrc/src/pynccl.cu`），并预先分配一块通信缓冲区。好处是开销更小，且与 CUDA Graph 的配合更可控。我们只实现了 `torch.distributed` 这一种，PyNCCL 留作练习（需要 GPU）。

!!! upstream "官方实现"
    - 线性层：@@upstream layers/linear.py:LinearQKVMerged@@、@@upstream layers/linear.py:LinearRowParallel@@
    - 切分：@@upstream models/weight.py:_shard_tensor@@
    - 通信：@@upstream distributed/impl.py:DistributedCommunicator@@
    - 进程组：@@upstream engine/engine.py:Engine._init_communication@@

## 测试

@@code tests/test_ch16_tp.py:test_tensor_parallel_matches_hf@@

每个 TP 配置起一个完整的服务（多个调度器进程、tokenizer、API Server），3 个提示词的贪心输出与单卡 Hugging Face 逐字相同。这同时验证了第 14 章的多 rank 消息同步。

!!! interview "怎么讲清楚"
    讲张量并行：Megatron 切法——注意力按头切（qkv 列并行、o_proj 行并行），MLP 先列并行（gate/up 按中间维切）再行并行（down），每个 decoder 层只需要两次 all-reduce（注意力后一次、MLP 后一次）。合并的 qkv 要先分别切 q、k、v 再拼接，否则各 rank 拿到的头不对；KV 头数少于 TP 数时（Qwen3-0.6B 有 8 个 KV 头、TP=16）每个 KV 头复制到两个 rank。词表并行：嵌入层每个 rank 只查自己那段词表、查不到的置零，all-reduce 求和；输出层各 rank 算自己那段词表的 logits，all-gather 拼接。各 rank 与单卡只差浮点求和顺序带来的微小误差。

## 练习

1. 计算 Llama-3.1-70B（80 层、hidden 8192、bf16）在 TP=8、每轮 decode 批大小 64 时，每层两次 all-reduce 的数据量。NVLink 带宽 450 GB/s（单向）时，通信大约占多少时间？
2. 如果 TP 数不能整除 q 头数（例如 Qwen2.5-0.5B 有 14 个 q 头、TP=4），会在哪里报错？怎样支持？
3. 实现 PyNCCL 插件：用 `ctypes` 加载 `libnccl.so`，实现 `all_reduce`；在 GPU 上与 `torch.distributed` 的结果比较。

??? success "参考答案"
    1. 每次 all-reduce 的数据是 `[64, 8192]` 个 bf16，1 MiB；ring all-reduce 每个 rank 发送约 2 × (8−1)/8 × 1 MiB ≈ 1.75 MiB，约 4 µs 的带宽时间，再加上每次 all-reduce 几微秒的固定延迟。80 层 × 2 次 ≈ 160 次，合计约 1～2 ms——小批量时固定延迟占主导，这就是 decode 阶段 TP 规模不宜太大的原因。
    2. `LinearQKVMerged` 里的 `div_even(num_qo_heads, tp_size)` 断言失败。可以给头数补齐（padding 出空头），或者只支持能整除的 TP 数；正式的引擎通常直接要求整除。
    3. 参考官方 `kernel/pynccl.py` 与 SGLang 的 `pynccl_wrapper.py`：rank 0 用 `ncclGetUniqueId` 生成 ID，通过 CPU 进程组广播；各 rank `ncclCommInitRank`；`all_reduce` 调用 `ncclAllReduce`，stream 用当前的 CUDA stream。

## 小结

- [x] Megatron 切法：先列并行（按头 / 按中间维）、再行并行，每层两次 all-reduce。
- [x] 合并的 qkv 要先分别切再拼接；KV 头不够分时在多个 rank 上复制。
- [x] 词表并行：嵌入层 all-reduce 求和，输出层 all-gather 拼接。
- [x] 各 rank 的输出与单卡只差浮点求和顺序带来的微小误差，贪心解码结果一致。
