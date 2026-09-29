---
title: CUDA Graph：补齐、固定缓冲区与 replay
chapter: perf/cuda-graph.md
difficulty: 中等
tags: [CUDA Graph, 固定缓冲区, 补齐]
---
CUDA Graph 录下来的 kernel 读写的地址是固定的，所以每一轮的输入都要**拷进录制时用的那块缓冲区**，而不是换一块新内存。书中用 `EmulatedGraph` 在 CPU 上模拟这一点，模板里给出了同样思路的简化版：

- `Buffers(max_bs)`：三个输入数组 `input_ids`、`positions`、`seq_lens`（长度 `max_bs`）和输出 `logits`；
- `EmulatedGraph(model, buffers, bs)`：录制时**记住这几个数组对象本身**；`replay()` 只从记住的数组的前 `bs` 个元素读输入、把结果写进记住的 `logits`，完全不看其他东西；
- `model.forward(input_ids, positions, seq_lens)` 是普通（eager）的前向，返回 logits。

实现 `GraphRunner`：

1. `determine_graph_bs(max_bs)`（模块级函数）：`[1, 2, 4]` 加上 8 到 `max_bs` 之间所有 8 的倍数，只保留 `<= max_bs` 的，升序去重；`max_bs < 1` 时返回空列表；
2. `GraphRunner(model, max_bs)`：分配一份 `Buffers(max_bs)`，为每个批大小录一个 `EmulatedGraph`，**从大到小**录制（把录制顺序记在 `self.capture_order` 里）；
3. `pad(reqs)`：`reqs` 是 `[(input_id, position, seq_len), ...]`；补上 dummy 请求 `(0, 0, 1)` 到最近的已录制批大小；超过最大值时不补齐、返回原列表；
4. `run(reqs)`：能用 graph 时：补齐 → 把补齐后的三列**原地写进缓冲区** → replay → 返回前 `len(reqs)` 行 logits；不能用 graph 时直接调用 `model.forward`。

测试会比较 `run` 与 eager 前向的结果，并检查每一轮都真的走了 replay。

<!-- 题解 -->
```python
bs = next((b for b in self.bs_list if b >= n), None)
padded = reqs + [DUMMY] * (bs - n)
ids, pos, lens = zip(*padded)
self.buf.input_ids[:bs] = ids          # 原地写：graph 记住的是这个数组对象
self.buf.positions[:bs] = pos
self.buf.seq_lens[:bs] = lens
self.graphs[bs].replay()
return self.buf.logits[:n].copy()
```

最常见的错误：`self.buf.input_ids = np.array(ids)`——这创建了一个新数组，graph 读的还是旧数组，结果悄悄地错了，没有任何报错。漏拷某一列（比如 `positions`）也是一样，书中第 18 章专门演示了这个 bug。
从大到小录制是为了让后面的小 graph 复用最大那个的内存池。
