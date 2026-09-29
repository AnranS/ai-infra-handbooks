---
title: 按字段顺序编码的消息：省略默认值与向后兼容
chapter: source/rust-frontend.md
difficulty: 中等
tags: [协议, 序列化, 跨语言]
---
vLLM 的前端（Python 或 Rust）和引擎核心之间传的 `EngineCoreRequest` 按**字段顺序**编码成一个数组（msgspec 的 `array_like=True`），末尾等于默认值的字段省略不发（`omit_defaults=True`）。
这里用 Python 列表代替 msgpack 的数组，实现这套规则。

`schema` 是 `[(字段名, 默认值), ...]`，默认值为 `REQUIRED` 的字段是必填的（必填字段总在前面）。

1. `encode(obj, schema)`：`obj` 是字典（缺少的非必填字段取默认值；缺少必填字段抛 `ValueError`；出现 schema 里没有的字段也抛 `ValueError`）。按 schema 的顺序取值，**去掉末尾所有等于默认值的非必填字段**，返回列表；
2. `decode(values, schema)`：按位置对应字段，缺少的末尾字段补默认值，返回字典；值比字段多（对方有你不认识的新字段）或少于必填字段数时抛 `ValueError`；
3. `compatible(old, new)`：新 schema 能否和旧 schema 互通——新 schema 必须是旧 schema 原样（字段名、顺序、默认值都不变）再在**末尾追加**若干**非必填**字段。

```python
S = [("request_id", REQUIRED), ("prompt", REQUIRED), ("priority", 0), ("client_index", 0)]
encode({"request_id": "r1", "prompt": [1, 2]}, S)                  # ["r1", [1, 2]]
encode({"request_id": "r1", "prompt": [1, 2], "client_index": 3}, S)  # ["r1", [1, 2], 0, 3]
decode(["r1", [1, 2]], S)["priority"]                             # 0
```

<!-- 题解 -->
- 只能去掉**末尾**的默认值：中间的字段即使等于默认值也必须占位，否则后面的字段会错位；
- 解码时多出来的值说明对方的 schema 比你新：这里选择报错（Rust 端按固定的元组解码时也会失败），更宽松的做法是忽略多余的尾部字段，但那样会悄悄丢信息；
- 向后兼容的唯一安全做法是"只在末尾追加、并且带默认值"：老版本发来的短数组会被新版本补上默认值，新版本在新字段等于默认值时发出的数组老版本也能读。在中间插字段、改顺序、改默认值都会让两边按位置解码时错位——而类型碰巧一致时甚至不会报错。
