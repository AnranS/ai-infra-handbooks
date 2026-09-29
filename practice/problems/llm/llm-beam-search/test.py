import hashlib

import numpy as np

from checker import check, check_close
from solution import beam_search


def table_model(table, V):
    def step(prefix):
        row = table.get(prefix[-1], None)
        p = np.full(V, 1e-9) if row is None else np.array(row, dtype=np.float64)
        return np.log(p / p.sum())

    return step


def hashed_model(V, seed):
    def step(prefix):
        h = hashlib.sha256((str(seed) + ",".join(map(str, prefix))).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        p = rng.dirichlet(np.ones(V) * 0.5)
        return np.log(p)

    return step


def ref(step_fn, bos, eos, k, max_len, lp=1.0):
    alive, fin = [((bos,), 0.0)], []
    for _ in range(max_len):
        c = []
        for tk, sc in alive:
            lg = step_fn(tk)
            c += [(tk + (t,), sc + float(lg[t])) for t in range(len(lg))]
        c.sort(key=lambda z: (-z[1], z[0]))
        alive = []
        for tk, sc in c:
            if tk[-1] == eos:
                if len(fin) < k:
                    fin.append((tk, sc))
            elif len(alive) < k:
                alive.append((tk, sc))
            if len(alive) == k:
                break
        if len(fin) >= k or not alive:
            break
    fin += alive
    out = sorted(((tk[1:], sc / (len(tk) - 1) ** lp) for tk, sc in fin), key=lambda z: (-z[1], z[0]))
    return out


def same(got, want, what):
    check([g[0] for g in got], [w[0] for w in want], what + "：序列")
    check_close([g[1] for g in got], [w[1] for w in want], rtol=1e-10, what=what + "：打分")


def test_example_beats_greedy():
    """贪心先选 0.6 的 A，但 B 后面接 EOS 的概率更高"""
    BOS, A, B, EOS = 0, 1, 2, 3
    table = {BOS: [0, 0.6, 0.4, 0], A: [0, 0.3, 0.3, 0.4], B: [0, 0, 0, 1.0]}
    got = beam_search(table_model(table, 4), BOS, EOS, beam_size=2, max_len=3, length_penalty=0.0)
    check(got[0][0], (B, EOS), "最优序列")
    check_close(got[0][1], np.log(0.4), rtol=1e-6, what="最优序列的打分")


def test_random_models_against_reference():
    for seed in range(12):
        V, eos = 6, 5
        step = hashed_model(V, seed)
        for k, max_len, lp in [(1, 5, 1.0), (3, 6, 1.0), (4, 5, 0.0), (2, 7, 2.0)]:
            same(beam_search(step, 0, eos, k, max_len, lp), ref(step, 0, eos, k, max_len, lp),
                 f"seed={seed}, beam={k}, max_len={max_len}, lp={lp}")


def test_max_len_cuts_off():
    step = table_model({0: [0, 1, 0], 1: [0, 1, 0]}, 3)   # 永远不生成 EOS=2
    got = beam_search(step, 0, 2, beam_size=2, max_len=4)
    check(got[0][0], (1, 1, 1, 1), "达到 max_len 时返回未结束的假设")


def test_finished_count_limit():
    step = table_model({0: [0, 0.5, 0.3, 0.2]}, 4)        # 下一步：1 或 2 然后再看
    got = beam_search(step, 0, 3, beam_size=1, max_len=5, length_penalty=0.0)
    check(len(got), 1, "beam_size=1 时最多一个已完成的假设")
