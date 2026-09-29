---
title: 进程间消息的序列化
chapter: serve/message.md
difficulty: 中等
tags: [序列化, dataclass, 递归]
---
mini-sglang 的各个进程（API server、tokenizer、调度器、detokenizer）之间用 ZMQ 传消息。消息是 dataclass，里面可能嵌套别的 dataclass、列表、字典，以及一维张量（这里用 numpy 数组代替）。
先把消息转换成只由基本类型（`int`、`float`、`str`、`bool`、`None`、`bytes`、`list`、`tuple`、`dict`）组成的结构，再交给 msgpack 编码。实现：

- `serialize(obj)`：
  - 基本类型原样返回；`list`、`tuple` 递归处理每个元素（保持类型）；`dict` 递归处理每个值（键不变）；
  - 一维 numpy 数组：`{"__type__": "Tensor", "buffer": arr.tobytes(), "dtype": str(arr.dtype)}`（不是一维时抛出 `ValueError`）；
  - 其他对象（dataclass）：`{"__type__": 类名, 字段名: 递归序列化的值, ...}`（用 `obj.__dict__`）；
- `deserialize(cls_map, data)`：`cls_map` 是"类名 → 类"的字典。遇到带 `"__type__"` 的字典就还原成对象（`Tensor` 还原成 numpy 数组，**要复制一份**，不能和 buffer 共享只读内存）；普通字典、列表、元组递归还原。

```python
@dataclass
class UserMsg:
    uid: int
    input_ids: np.ndarray
    sampling_params: SamplingParams

data = serialize(msg)                       # 只含基本类型，可以直接 msgpack.packb
deserialize({"UserMsg": UserMsg, "SamplingParams": SamplingParams}, data)   # 还原
```

<!-- 题解 -->
与书中 `message/utils.py` 一致，两对互相递归的函数：`_serialize_any` 处理容器和基本类型、遇到对象交给 `serialize_type`；反序列化同理。

`np.frombuffer` 返回的数组和 buffer 共享内存，而 msgpack 解出来的 `bytes` 是只读的，所以要 `.copy()`（书中是 `torch.from_numpy(np.frombuffer(...).copy())`）。
为什么不用 pickle：pickle 能执行任意代码，跨进程/跨机器时不安全；而且对大量小消息，这种"只有基本类型"的结构配合 msgpack 更快。
