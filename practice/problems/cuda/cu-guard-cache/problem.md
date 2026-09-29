---
title: torch.compile 的 guard 缓存：什么时候会重新编译
chapter: framework/compile.md
difficulty: 中等
tags: [torch.compile, Dynamo, guard, 动态形状]
---
模拟 Dynamo 对一个函数的编译缓存（只看输入的形状）。`GuardCache(cache_limit=8, mark_dynamic=())`，`mark_dynamic` 是一组 `(第几个输入, 第几维)`，表示事先标成动态的维度。`call(shapes)` 处理一次调用，`shapes` 是每个输入的形状（元组的元组），返回 `"hit"`、`"compile"` 或 `"eager"`：

1. **检查 guard**：每个已编译的版本记着一组 guard——每个输入的维数，以及每一维要么是一个具体的大小（静态），要么是 `"dyn"`（动态，要求大小 ≥ 2）。按**从新到旧**的顺序检查，第一个全部满足的就命中，返回 `"hit"`；
2. **缓存满了**：没有命中、而且已经编译了 `cache_limit` 个版本时，不再编译，返回 `"eager"`（退回 eager 执行）；
3. **重新编译**：先更新"自动动态"的集合——和**第一次编译时**的形状比，大小变了的维度（维数相同的输入才比）从此都算动态；然后生成新版本的 guard：在 `mark_dynamic` 或自动动态集合里的维度是 `"dyn"`，其他维度是当前的具体大小。例外：当前大小是 0 或 1 的维度总是按具体大小编译（Dynamo 对 0 和 1 做特化）。返回 `"compile"`。

另外提供 `num_compiles()` 和 `guards()`（按编译顺序返回每个版本的 guard，格式与上面一致，例如 `(("dyn", 16), (16, 16))`）。

```python
c = GuardCache()
[c.call(((b, 16), (16, 16))) for b in (4, 4, 8, 16, 32)]
# ['compile', 'hit', 'compile', 'hit', 'hit']：第二次编译把 batch 维标成了动态
c.guards()      # [((4, 16), (16, 16)), (('dyn', 16), (16, 16))]
```

<!-- 题解 -->
这就是本章 `recompile.py` 的输出背后的规则："第一次按具体形状编译，某一维第二次出现不同的值时标成动态"。几个推论：

- batch 从 1 开始的服务会多编译一次：动态版本要求大小 ≥ 2，batch 为 1 时总要单独的静态版本；
- 事先 `mark_dynamic` 就能省掉"先按具体形状编一次"，第一次编译出来就是动态的；
- 维数变化（比如有时传 2 维、有时传 3 维）没法用动态维度表示，每种都要单独编译；编译的版本数到了上限（`torch._dynamo.config.recompile_limit`，默认 8），之后就退回 eager 执行——性能突然掉下来，日志里会有 "hit recompile_limit" 的警告。

推理引擎因此都在启动时固定好动态维度并预热（vLLM 编译时只把 token 数这一维标成动态，再为 CUDA Graph 的几种 batch 大小分别捕获），避免线上请求触发编译。
