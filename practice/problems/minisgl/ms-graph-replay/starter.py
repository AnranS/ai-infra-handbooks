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
    pass


class GraphRunner:
    def __init__(self, model, max_bs):
        self.model = model
        self.bs_list = determine_graph_bs(max_bs)
        self.buf = Buffers(max_bs)
        self.graphs = {}
        self.capture_order = []

    def pad(self, reqs):
        pass

    def run(self, reqs):
        ids, pos, lens = (np.array(c) for c in zip(*reqs))
        return self.model.forward(ids, pos, lens)          # 没有用 graph
