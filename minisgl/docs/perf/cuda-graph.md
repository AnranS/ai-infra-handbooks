# CUDA Graph

<p class="lead">一个 28 层的模型 decode 一步，要启动几百个 kernel。小批量时每个 kernel 只算几微秒，而 CPU 发射一个 kernel 也要几微秒——GPU 大部分时间在等 CPU。CUDA Graph 把一整次前向录制下来，之后一次 replay 就能发射全部 kernel。代价是录下来的 kernel 读写的地址全都固定了，所有输入都必须先拷进固定的缓冲区。这一章实现 mini-sglang 的 <code>GraphRunner</code>，并用一个 CPU 仿真验证"所有输入都走固定缓冲区"这个约定。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. CUDA Graph 为什么只用于 decode，不用于 prefill？
    2. 录制时批大小是 4，实际 batch 只有 3 个请求，怎么办？补上的那个请求算出的东西写到哪里？
    3. 注意力元数据（每个请求的 KV 长度、页表）每一轮都不同，graph 怎么读到新的值？
    4. 为什么先录制最大的批大小？

??? success "自测参考答案（先自己答，再展开对照）"
    1. decode 每一步只有很少的 token、却有几百个小 kernel，CPU 发射开销占了大头，graph 能消除它；prefill 的形状（token 数）每次都不同、变化范围大，录不过来，而且 prefill 计算量大，发射开销本来就不重要。
    2. 用 dummy 请求把 batch 补到 4：它的输入、元数据都放在固定缓冲区里，算出的 KV 写到 KV 池里专门多分配的那一页，结果直接丢弃。
    3. 所有元数据都放在录制时就固定的缓冲区里，graph 只认这些地址；每次 replay 之前，把这一轮的新值拷进这些缓冲区。
    4. 先录最大的，让它按最大的需求从内存池里分配临时内存，之后更小的 graph 可以复用同一个内存池，不再增加显存。

**本章要写的文件**：`engine/graph.py`；注意力后端里的 `init_capture_graph`、`prepare_for_capture`、`prepare_for_replay`。

@@video cudagraph 动画：CUDA Graph 的录制、补齐与 replay（约 1.5 分钟）@@

## 录制与 replay

CUDA Graph 录制的是"kernel + 参数"，参数里的指针在录制时就固定了。所以：

1. **输入放在固定缓冲区里**。录制前让 batch 的 `input_ids`、`positions`、`out_loc` 直接指向缓冲区的切片；replay 前把当前 batch 的值拷进去；
2. **注意力元数据也放在固定缓冲区里**，由注意力后端负责：录制时元数据指向后端自己的缓冲区，replay 前把新的元数据拷进去；
3. **每个批大小录一个 graph**，实际 batch 向上补齐到最近的已录制大小。

@@diagram cuda-graph 录制与 replay：所有输入都走固定缓冲区@@

@@code python/minisgl/engine/graph.py:GraphCaptureBuffer@@

@@code python/minisgl/engine/graph.py:GraphRunner._capture_graphs@@

录制时用的 batch 全是 dummy 请求（第 6 章：它在 page table 里有专门的一行，指向 KV 池里专门的一页）。每个批大小先正常跑一遍（预热：触发各种库的懒初始化，避免它们被录进 graph），再在 `torch.cuda.graph` 上下文里跑一遍录制。从最大的批大小开始录，之后的小 graph 复用同一个内存池（`pool`），显存占用由最大的那个决定。

默认录制的批大小是 1、2、4，以及 8 到上限之间所有 8 的倍数，上限按显存决定（80 GB 以上的卡 256，否则 160）。

## 补齐与 replay

@@code python/minisgl/engine/graph.py:GraphRunner.pad_batch@@

`pad_batch` 在调度器准备 batch 的第一步调用（第 7 章）：能用 graph 时，用 dummy 请求把 batch 补到最近的已录制大小。补上的 dummy 请求也会得到位置、输入、元数据，它们的 KV 写进 dummy 页，logits 被丢弃（`logits[:batch.size]`）。

@@code python/minisgl/engine/graph.py:GraphRunner.replay@@

注意力后端的三个钩子，以参考后端为例（与官方 FlashAttention 后端的做法相同）：

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_for_capture@@

@@code python/minisgl/attention/torch_backend.py:TorchAttnBackend.prepare_for_replay@@

FlashInfer 后端的做法不同：它为 CUDA Graph 提供了专门的 wrapper，构造时传入固定地址的缓冲区，`plan` 会把新元数据拷进这些缓冲区，所以 replay 前重新 plan 一次即可（第 17 章 `FlashInferBackend.prepare_for_replay`）。

## CPU 上的仿真

CPU 上没有 CUDA Graph，但我们仍然想验证"所有输入都走固定缓冲区"这个约定有没有被遵守——这是 CUDA Graph 相关 bug 最常见的来源：某个输入忘了拷进缓冲区，graph 读到的就是录制时（或上一轮）的旧值，而且不会报任何错。

@@code python/minisgl/engine/graph.py:EmulatedGraph@@

`EmulatedGraph` 在"录制"时记下要执行的函数和录制时的 batch。录制时的 batch 里，输入字段都指向缓冲区，注意力元数据都指向后端的缓冲区；replay 时用这个 batch（而不是当前的 batch）重新执行一遍前向。于是它的行为与真正的 CUDA Graph 一样：**只能看到缓冲区里的东西**。

@@code examples/ch18_cuda_graph.py@@

@@output ch18_cuda_graph@@

3 个请求的 decode 每轮都补齐到 4、走 replay，输出与不用 graph 时完全相同。然后故意制造一个 bug：replay 前漏拷 `positions`。graph 里的 RoPE 读到的是录制时缓冲区里的位置（全是 0），从第二个 token 开始输出就错了，而整个过程没有任何报错——在真正的 GPU 上也会是这个样子。

!!! diff "与官方的差异"
    `EmulatedGraph` 是我们加的：设备是 CPU 且显式设置了 `cuda_graph_max_bs` 时，用它代替 `torch.cuda.CUDAGraph`（CPU 上默认关闭 graph）。录制循环、补齐、replay 的逻辑与官方相同。

!!! upstream "官方实现"
    - @@upstream engine/graph.py:GraphRunner@@
    - 批大小的选择：@@upstream engine/graph.py:_determine_cuda_graph_bs@@
    - 注意力后端的钩子：@@upstream attention/fa.py:FlashAttentionBackend.prepare_for_replay@@、@@upstream attention/fi.py:FlashInferBackend.prepare_for_capture@@
    - 注意：关闭 CUDA Graph 用 `--cuda-graph-max-bs 0`；销毁 graph 必须在释放 NCCL 资源之前，否则可能卡住（官方 `destroy_cuda_graphs` 的注释）。

## 测试

@@code tests/test_ch18_cuda_graph.py:test_forgetting_to_copy_an_input_breaks_replay@@

同一个文件里还有：参考后端、FlashInfer、FlashAttention 三种后端在仿真 graph 下的端到端测试（每轮 decode 都走 replay，输出与 Hugging Face 一致），以及批大小列表的计算。

!!! interview "面试怎么答"
    CUDA Graph 题：decode 每步几百个小 kernel，CPU 发射开销比 GPU 计算还长，CUDA Graph 把整次 decode 前向录下来、一次 replay；prefill 的 token 数变化大、本身计算重，不需要也不适合录。录下的是固定的地址，所以所有输入和注意力元数据都要放在固定的缓冲区里，replay 前把新值拷进去（漏拷一个就会静默出错）；每个批大小录一个图，实际 batch 用 dummy 请求补齐到最近的档位，dummy 的 KV 写进专门留的那一页。先录最大的批大小，让它分配的内存池被后面的图复用。收益在小模型、小 batch 时最大。

## 练习

1. 如果 batch 有 5 个请求、已录制 `[1, 2, 4, 8]`，补齐到几？补齐带来的额外计算有多少？怎样选择批大小列表才能在"graph 数量（显存）"和"补齐浪费"之间取得平衡？
2. 采样也能录进 graph 吗？mini-sglang 为什么没有这样做？
3. 张量并行时，graph 里的 all-reduce 能被录制吗？需要注意什么？

??? success "参考答案"
    1. 补到 8，多算 3 个 dummy 请求（约 60%）。小批量时各种批大小都很常见，间隔要小（1、2、4、8）；大批量时 kernel 已经比较"饱"，补齐浪费的比例也小，间隔可以大（每 8 或 16 一档）——官方的默认列表就是这个思路。
    2. 可以，但采样参数（温度、top-k、top-p）因请求而异，还需要随机数状态，这些都要放进固定缓冲区；而且贪心时只需要一个 argmax，本身开销很小。
    3. 可以，NCCL 支持在 CUDA Graph 中录制集合通信。要保证所有 rank 以相同的顺序录制和 replay 同样的 graph（第 14 章的"各 rank 完全一致"再次派上用场），并且在销毁 NCCL 通信器之前先销毁 graph。

## 小结

- [x] CUDA Graph 录下整次 decode 前向，replay 一次发射全部 kernel，消除小批量时的 CPU 发射开销。
- [x] 所有输入和注意力元数据都必须放在固定缓冲区里，replay 前拷入；每个批大小一个 graph，实际 batch 用 dummy 请求补齐。
- [x] `EmulatedGraph` 在 CPU 上仿真"只能看到缓冲区"的行为：漏拷任何一个输入，输出就会出错，且不会报错。
