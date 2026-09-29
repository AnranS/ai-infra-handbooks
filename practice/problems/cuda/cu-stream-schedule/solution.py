def simulate(ops):
    engine = {"h2d": "h2d", "kernel": "compute", "d2h": "d2h"}
    stream_ready, engine_free, events = {}, {}, {}
    timeline = []
    for o in ops:
        s, kind = o["stream"], o["op"]
        if kind in engine:
            e = engine[kind]
            start = max(stream_ready.get(s, 0), engine_free.get(e, 0))
            end = start + o["dur"]
            stream_ready[s] = engine_free[e] = end
            timeline.append((start, end))
        elif kind == "record":
            events[o["event"]] = stream_ready.get(s, 0)
        elif kind == "wait":
            stream_ready[s] = max(stream_ready.get(s, 0), events.get(o["event"], 0))
        else:
            raise ValueError(f"未知操作：{kind}")
    return timeline, max((e for _, e in timeline), default=0)
