REQUIRED = object()


def encode(obj, schema):
    names = {name for name, _ in schema}
    unknown = set(obj) - names
    if unknown:
        raise ValueError(f"schema 里没有这些字段：{sorted(unknown)}")
    values = []
    for name, default in schema:
        if name in obj:
            values.append(obj[name])
        elif default is REQUIRED:
            raise ValueError(f"缺少必填字段 {name}")
        else:
            values.append(default)
    n = len(values)
    while n > 0 and schema[n - 1][1] is not REQUIRED and values[n - 1] == schema[n - 1][1]:
        n -= 1
    return values[:n]


def decode(values, schema):
    required = sum(1 for _, d in schema if d is REQUIRED)
    if len(values) > len(schema):
        raise ValueError(f"值有 {len(values)} 个，比字段多：对方的 schema 更新")
    if len(values) < required:
        raise ValueError(f"值只有 {len(values)} 个，少于 {required} 个必填字段")
    out = {}
    for i, (name, default) in enumerate(schema):
        out[name] = values[i] if i < len(values) else default
    return out


def compatible(old, new):
    if len(new) < len(old):
        return False
    for (n1, d1), (n2, d2) in zip(old, new):
        if n1 != n2 or (d1 is REQUIRED) != (d2 is REQUIRED) or (d1 is not REQUIRED and d1 != d2):
            return False
    return all(d is not REQUIRED for _, d in new[len(old):])
