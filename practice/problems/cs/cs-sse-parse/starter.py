def dechunk(stream):
    out = b""
    for part in stream:                                 # 假设每个片段正好是一个完整的块
        line, rest = part.split(b"\r\n", 1)
        size = int(line, 16)
        if size == 0:
            break
        out += rest[:size]
    return out


def sse_events(body):
    events = []
    for block in body.split(b"\n\n"):
        event, data = None, []
        for line in block.split(b"\n"):
            field, _, value = line.partition(b":")
            if field == b"data":
                data.append(value.decode().strip())     # strip 会把有意义的空格也去掉
            elif field == b"event":
                event = value.decode().strip()
        if data:
            events.append((event, "\n".join(data)))
    return events
