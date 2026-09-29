import random

from checker import check
from solution import Req, overlap_loop


def normal_loop(specs, sample, eos):
    msgs = {}
    for uid, plen, mt in specs:
        out = []
        for n in range(mt):
            tok = sample(uid, n)
            fin = tok == eos or n + 1 >= mt
            out.append((uid, tok, fin))
            if fin:
                break
        msgs[uid] = out
    return msgs


def run(specs, sample, eos):
    reqs = [Req(u, p, m) for u, p, m in specs]
    msgs = overlap_loop(reqs, sample, eos)
    by = {}
    for m in msgs:
        by.setdefault(m[0], []).append(m)
    return by


def test_example_max_tokens():
    """max_tokens=3：结束标记应该在第 3 个 token 上"""
    got = run([(1, 5, 3)], lambda uid, n: 100 + n, eos=0)
    check(got[1], [(1, 100, False), (1, 101, False), (1, 102, True)], "请求 1 的消息")


def test_example_eos():
    """第 2 个 token 是 EOS：之后不能再有消息"""
    got = run([(7, 3, 10)], lambda uid, n: 0 if n == 1 else 50 + n, eos=0)
    check(got[7], [(7, 50, False), (7, 0, True)], "请求 7 的消息")


def test_random_against_normal_loop():
    rng = random.Random(0)
    for trial in range(100):
        specs = [(u, rng.randint(1, 20), rng.randint(1, 8)) for u in range(rng.randint(1, 6))]
        table = {(u, n): rng.choice([0] + list(range(10, 20))) for u, _, _ in specs for n in range(20)}
        sample = lambda uid, n: table[uid, n]  # noqa: E731
        want = normal_loop(specs, sample, eos=0)
        got = run(specs, sample, eos=0)
        for u, _, _ in specs:
            check(got.get(u, []), want[u], f"随机用例 {trial}：请求 {u} 的消息序列")
