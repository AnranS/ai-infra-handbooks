---
title: 原子写文件与计时器
chapter: core/errors-context.md
difficulty: 中等
tags: [上下文管理器, contextlib, 异常处理]
---
实现两个上下文管理器。

**`atomic_write(path, mode="w")`**：像 `open(path, mode)` 一样返回一个可写的文件对象，但要保证**要么完整写入、要么完全不改动原文件**：

- 先写到同一目录下的临时文件，`with` 块正常结束后再用 `os.replace` 把临时文件换成目标文件；
- `with` 块里抛出异常时，删除临时文件、保留原文件不变，并让异常继续向外抛出；
- `mode` 只允许 `"w"` 和 `"wb"`，其他值抛出 `ValueError`。

**`Timer()`**：`with Timer() as t:` 结束后 `t.elapsed` 是经过的秒数（`float`）；块内也可以随时读取 `t.elapsed`，得到"到现在为止"的时间。`Timer` 不吞掉异常。

```python
with atomic_write("config.json") as f:
    f.write(json.dumps(cfg))        # 中途崩溃也不会留下写了一半的 config.json

with Timer() as t:
    work()
print(f"{t.elapsed:.3f}s")
```

<!-- 题解 -->
用 `contextlib.contextmanager` 写 `atomic_write` 最简单：`yield` 前打开临时文件，`yield` 放在 `try` 里；
正常结束走 `os.replace(tmp, path)`（同一文件系统上是原子操作），`except BaseException` 分支删除临时文件后 `raise`。

`Timer` 写成类：`__enter__` 记录 `time.perf_counter()` 并返回 `self`；`__exit__` 记录结束时间并返回 `None`（假值），异常就会继续传播。
`elapsed` 用 property，结束前返回"现在 - 开始"，结束后返回固定值。
