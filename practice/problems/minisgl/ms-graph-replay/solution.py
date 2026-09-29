import numpy as np

DUMMY = (0, 0, 1)


class Buffers:
    def __init__(self, max_bs, vocab=8):
        self.input_ids = np.zeros(max_bs, dtype=np.int64)
        self.positions = np.zeros(max_bs, dtype=np.int64)
        self.seq_lens = np.ones(max_bs, dtype=np.int64)
        self.logits = np.zeros((max_bs, vocab))


class EmulatedGraph:
    """录制时记住缓冲区里的数组对象；replay 只读这些数组（模拟 CUDA Graph 的固定地址）。"""

    def __init__(self, model, buffers, bs):
        self.model, self.bs = model, bs
        self.arrays = (buffers.input_ids, buffers.positions, buffers.seq_lens, buffers.logits)
        self.replays = 0

    def replay(self):
        ids, pos, lens, out = self.arrays
        out[:self.bs] = self.model.forward(ids[:self.bs], pos[:self.bs], lens[:self.bs])
        self.replays += 1


def determine_graph_bs(max_bs):
    if max_bs < 1:
        return []
    return sorted({b for b in [1, 2, 4] + list(range(8, max_bs + 1, 8)) if b <= max_bs})


class GraphRunner:
    def __init__(self, model, max_bs):
        self.model = model
        self.bs_list = determine_graph_bs(max_bs)
        self.buf = Buffers(max_bs)
        self.graphs = {}
        self.capture_order = []
        for bs in sorted(self.bs_list, reverse=True):       # 从大到小录制
            self.graphs[bs] = EmulatedGraph(model, self.buf, bs)
            self.capture_order.append(bs)

    def pad(self, reqs):
        bs = next((b for b in self.bs_list if b >= len(reqs)), None)
        if bs is None:
            return list(reqs)
        return list(reqs) + [DUMMY] * (bs - len(reqs))

    def run(self, reqs):
        n = len(reqs)
        padded = self.pad(reqs)
        if len(padded) not in self.graphs:
            ids, pos, lens = (np.array(c) for c in zip(*reqs))
            return self.model.forward(ids, pos, lens)
        bs = len(padded)
        ids, pos, lens = zip(*padded)
        self.buf.input_ids[:bs] = ids                       # 原地写进 graph 记住的缓冲区
        self.buf.positions[:bs] = pos
        self.buf.seq_lens[:bs] = lens
        self.graphs[bs].replay()
        return self.buf.logits[:n].copy()
