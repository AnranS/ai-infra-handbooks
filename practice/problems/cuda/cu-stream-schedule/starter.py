def simulate(ops):
    t, timeline = 0, []
    for o in ops:                          # 完全串行：没有利用多个引擎和 stream 并发
        if o["op"] in ("h2d", "kernel", "d2h"):
            timeline.append((t, t + o["dur"]))
            t += o["dur"]
    return timeline, t
