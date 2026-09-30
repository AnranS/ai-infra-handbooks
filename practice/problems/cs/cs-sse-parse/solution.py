def dechunk(stream):
    buf, out = b"", b""
    it = iter(stream)
    while True:
        while b"\r\n" not in buf:                       # 还没收到完整的长度行
            try:
                buf += next(it)
            except StopIteration:
                return out
        line, rest = buf.split(b"\r\n", 1)
        size = int(line.split(b";")[0], 16)             # 忽略块扩展
        if size == 0:
            return out
        while len(rest) < size + 2:                     # 数据 + 结尾的 \r\n
            try:
                rest += next(it)
            except StopIteration:
                return out + rest[:size]
        out += rest[:size]
        buf = rest[size + 2:]


def sse_events(body):
    events = []
    for block in body.replace(b"\r\n", b"\n").split(b"\n\n"):
        event, data = None, []
        for line in block.split(b"\n"):
            if not line or line.startswith(b":"):       # 空行与注释（心跳）
                continue
            field, _, value = line.partition(b":")
            if value.startswith(b" "):                  # 冒号后的一个空格是格式的一部分
                value = value[1:]
            if field == b"data":
                data.append(value.decode())
            elif field == b"event":
                event = value.decode()
        if data:
            events.append((event, "\n".join(data)))
    return events
