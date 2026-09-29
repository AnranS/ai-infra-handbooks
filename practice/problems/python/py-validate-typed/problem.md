---
title: 按类型标注做运行时校验
chapter: types/typing.md
difficulty: 困难
tags: [typing, get_type_hints, get_origin]
---
类型标注默认不会在运行时检查。写一个 `validate(value, tp)`：检查 `value` 是否符合类型 `tp`，符合返回 `True`，否则抛出 `TypeError`，信息里写出**出错的位置**。

需要支持的类型：

- 普通类：`int`、`str`、`float`……用 `isinstance` 判断，但 `bool` 不算 `int`；
- `list[T]`、`set[T]`、`tuple[T, ...]`（变长）、`tuple[A, B]`（定长）、`dict[K, V]`；
- `Optional[T]`、`T | None`、`Union[A, B]`、`A | B`；
- `Literal["a", "b"]`；
- `typing.Any`；
- `TypedDict` 子类：字典必须包含所有键（`total=True`），每个值符合标注；不允许多余的键。

出错位置的格式：从根开始，列表/元组下标写成 `[0]`，字典的值写成 `["key"]`，例如：

```python
class Req(TypedDict):
    prompt: str
    stop: list[str]

validate({"prompt": "hi", "stop": ["a", 1]}, Req)
# TypeError: $["stop"][1]: 期望 str，实际是 int
```

根的位置写成 `$`。只要求信息里包含这个位置字符串（例如 `$["stop"][1]`）。

<!-- 题解 -->
`typing.get_origin(tp)` 取出泛型的"外壳"（`list[int]` → `list`，`int | None` → `types.UnionType`，`Optional[int]` → `typing.Union`，`Literal[...]` → `typing.Literal`），
`typing.get_args(tp)` 取出参数。TypedDict 用 `typing.is_typeddict(tp)` 判断，`typing.get_type_hints(tp)` 取字段标注。
递归时把路径作为参数往下传；Union 的做法是"逐个尝试，有一个通过就算通过"。
