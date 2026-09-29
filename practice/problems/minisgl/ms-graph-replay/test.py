import numpy as np

from checker import check, check_close
from solution import DUMMY, GraphRunner, determine_graph_bs


class ToyModel:
    """logits 同时依赖 token、位置和序列长度：漏拷任何一列结果都会变"""

    def forward(self, ids, pos, lens):
        v = np.arange(8)
        return np.sin(np.asarray(ids)[:, None] * 0.7 + v) + np.cos(np.asarray(pos)[:, None] * 0.3 - v) \
            + np.asarray(lens)[:, None] * 0.01 * v


def reqs(n, seed):
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(0, 100)), int(rng.integers(0, 500)), int(rng.integers(1, 600))) for _ in range(n)]


def test_example():
    check(determine_graph_bs(20), [1, 2, 4, 8, 16], "determine_graph_bs(20)")
    r = GraphRunner(ToyModel(), 16)
    check(r.capture_order, [16, 8, 4, 2, 1], "从大到小录制")
    check(len(r.pad(reqs(3, 0))), 4, "3 个请求补齐到 4")
    check(r.pad(reqs(3, 0))[-1], DUMMY, "补上的是 dummy 请求")


def test_bs_list():
    check(determine_graph_bs(0), [], "max_bs=0")
    check(determine_graph_bs(3), [1, 2], "max_bs=3")
    check(determine_graph_bs(33)[-3:], [16, 24, 32], "max_bs=33")


def test_replay_matches_eager_many_rounds():
    m = ToyModel()
    r = GraphRunner(m, 16)
    for rnd in range(12):
        batch = reqs(1 + rnd % 7 * 2, seed=rnd)
        ids, pos, lens = (np.array(c) for c in zip(*batch))
        check_close(r.run(batch), m.forward(ids, pos, lens), rtol=1e-12, atol=1e-12,
                    what=f"第 {rnd} 轮（{len(batch)} 个请求）")
    assert sum(g.replays for g in r.graphs.values()) == 12, "每一轮都应该走 replay"


def test_too_large_falls_back():
    m = ToyModel()
    r = GraphRunner(m, 8)
    batch = reqs(11, seed=99)
    ids, pos, lens = (np.array(c) for c in zip(*batch))
    check_close(r.run(batch), m.forward(ids, pos, lens), what="超过最大批大小：走 eager")
    check(sum(g.replays for g in r.graphs.values()), 0, "没有 replay")
