from checker import check
from solution import simulate


def op(s, kind, dur=None, event=None):
    d = {"stream": s, "op": kind}
    if dur is not None:
        d["dur"] = dur
    if event is not None:
        d["event"] = event
    return d


def pipeline(streams, order="depth"):
    stages = [("h2d", 4), ("kernel", 6), ("d2h", 4)]
    if order == "depth":
        return [op(s, k, d) for s in streams for k, d in stages]
    return [op(s, k, d) for k, d in stages for s in streams]


def test_example():
    timeline, total = simulate(pipeline([1, 2, 3]))
    check(total, 26, "3 个 stream 的总时间")
    check(timeline[:3], [(0, 4), (4, 10), (10, 14)], "stream 1 的三个操作")


def test_breadth_first_order():
    timeline, total = simulate(pipeline([1, 2, 3], order="breadth"))
    check(total, 26, "广度优先发出的总时间")
    check(timeline[6:], [(10, 14), (16, 20), (22, 26)], "三个 D2H")


def test_in_order_engine_blocks():
    """引擎按发出顺序服务：先发出的长 kernel 挡住了后面本可以开始的 kernel"""
    ops = [op(1, "h2d", 10), op(1, "kernel", 5), op(2, "kernel", 1)]
    timeline, total = simulate(ops)
    check(timeline, [(0, 10), (10, 15), (15, 16)], "时间线")


def test_events():
    ops = [op(1, "kernel", 5), op(1, "record", event="e"), op(2, "wait", event="e"), op(2, "d2h", 3),
           op(3, "wait", event="never"), op(3, "h2d", 2)]
    timeline, total = simulate(ops)
    check(timeline, [(0, 5), (5, 8), (0, 2)], "时间线")
    check(total, 8, "总时间")


def test_same_stream_serializes_across_engines():
    ops = [op(1, "h2d", 3), op(1, "d2h", 3), op(2, "h2d", 1)]
    timeline, _ = simulate(ops)
    check(timeline, [(0, 3), (3, 6), (3, 4)], "同一个 stream 的操作串行；H2D 引擎按发出顺序")


def test_empty():
    check(simulate([]), ([], 0), "没有操作")
