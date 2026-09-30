from checker import check
from solution import dechunk, sse_events


def test_example():
    check(dechunk([b"5\r\nhel", b"lo\r\n3\r\nabc\r\n0\r\n\r\n"]), b"helloabc", "跨片段的块")
    check(sse_events(b"data: a\ndata: b\n\nevent: done\ndata: [DONE]\n\n"),
          [(None, "a\nb"), ("done", "[DONE]")], "多行 data 与 event 字段")


def test_byte_at_a_time():
    raw = b"5\r\nhello\r\n1\r\n!\r\n0\r\n\r\n"
    check(dechunk([raw[i:i + 1] for i in range(len(raw))]), b"hello!", "一次只收到一个字节")


def test_one_big_chunk():
    check(dechunk([b"c\r\nhello world!\r\n0\r\n\r\n"]), b"hello world!", "十六进制长度 c = 12")


def test_chunk_extension():
    check(dechunk([b"5;a=b\r\nhello\r\n0\r\n\r\n"]), b"hello", "块扩展要忽略")


def test_sse_comment_and_crlf():
    body = b": keep-alive\r\n\r\ndata: x\r\n\r\ndata:  y\r\n\r\n"
    check(sse_events(body), [(None, "x"), (None, " y")], "心跳注释忽略；只去掉一个空格")


def test_sse_empty_and_tail():
    check(sse_events(b"\n\ndata: only\n\n\n"), [(None, "only")], "空块跳过")
    check(sse_events(b"event: ping\n\n"), [], "没有 data 的事件跳过")


def test_stream_of_events():
    parts = [b'%x\r\ndata: {"t": %d}\n\n\r\n' % (len(b'data: {"t": %d}\n\n' % i), i) for i in range(5)]
    body = dechunk(parts + [b"0\r\n\r\n"])
    events = sse_events(body)
    check(len(events), 5, "五个事件")
    check(events[2][1], '{"t": 2}', "第三个事件的载荷")
