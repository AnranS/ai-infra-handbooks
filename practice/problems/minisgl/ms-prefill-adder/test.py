import random

from checker import check
from solution import PendingReq, schedule_prefill


def test_example():
    reqs = [PendingReq(1, input_len=10, output_len=5), PendingReq(2, 30, 5, cached_len=4), PendingReq(3, 8, 5)]
    batch, rest = schedule_prefill(reqs, token_budget=20, reserved_size=0, available_kv=100, free_rows=4)
    check(batch, [(1, 0, 10), (2, 4, 14)], "第一轮：请求 1 全部，请求 2 分块做到 14")
    check([(r.uid, r.chunked) for r in rest], [(2, 14), (3, None)], "请求 2 排在最前面，其次是请求 3")
    batch, rest = schedule_prefill(rest, 20, reserved_size=15 + 31, available_kv=100, free_rows=2)
    check(batch, [(2, 14, 30), (3, 0, 4)], "第二轮：请求 2 接着做完，请求 3 开始分块")
    check([(r.uid, r.chunked) for r in rest], [(3, 4)], "剩下请求 3")


def test_admission_blocks_queue():
    reqs = [PendingReq(1, 10, 100), PendingReq(2, 5, 1)]
    batch, rest = schedule_prefill(reqs, 64, reserved_size=0, available_kv=50, free_rows=4)
    check(batch, [], "第一个请求最坏需要 110 > 50：先来先服务，后面的也不接收")
    check([r.uid for r in rest], [1, 2], "队列不变")


def test_reserved_and_rows():
    reqs = [PendingReq(i, 10, 10) for i in range(5)]
    batch, rest = schedule_prefill(reqs, 1000, reserved_size=30, available_kv=100, free_rows=10)
    check([b[0] for b in batch], [0, 1, 2], "30 + 3×20 = 90 ≤ 100，第 4 个放不下")
    batch, _ = schedule_prefill([PendingReq(i, 10, 10) for i in range(5)], 1000, 0, 1000, free_rows=2)
    check(len(batch), 2, "只有 2 个空闲行")


def test_cached_prefix_reduces_need():
    reqs = [PendingReq(1, 100, 10, cached_len=95)]
    batch, rest = schedule_prefill(reqs, 64, reserved_size=0, available_kv=20, free_rows=1)
    check(batch, [(1, 95, 100)], "命中 95 个：只需要 5 + 10 = 15 的 KV")


def ref(pending, budget, reserved, avail, rows):
    out, chunk, n_taken = [], [], 0
    for r in pending:
        if budget <= 0:
            break
        if r.chunked is not None:
            s = r.chunked
        else:
            need = r.input_len - r.cached_len + r.output_len
            if rows == 0 or need + reserved > avail:
                break
            rows -= 1
            reserved += need
            s = r.cached_len
        k = min(budget, r.input_len - s)
        budget -= k
        out.append((r.uid, s, s + k))
        n_taken += 1
        if s + k < r.input_len:
            chunk.append((r.uid, s + k))
    rest = chunk + [(r.uid, r.chunked) for r in pending[n_taken:]]
    return out, rest


def test_random_rounds():
    rng = random.Random(0)
    for trial in range(30):
        mk = lambda: [PendingReq(i, rng.randint(1, 60), rng.randint(0, 30), 0) for i in range(rng.randint(1, 8))]  # noqa: E731
        a, b = mk(), None
        for r in a:
            r.cached_len = rng.randint(0, r.input_len - 1)
        b = [PendingReq(r.uid, r.input_len, r.output_len, r.cached_len) for r in a]
        for rnd in range(5):
            kw = dict(token_budget=rng.choice([8, 16, 64]), reserved_size=rng.randint(0, 40),
                      available_kv=rng.randint(40, 200), free_rows=rng.randint(0, 6))
            want_batch, want_rest = ref(b, kw["token_budget"], kw["reserved_size"], kw["available_kv"], kw["free_rows"])
            batch, rest = schedule_prefill(a, **kw)
            check(batch, want_batch, f"随机用例 {trial} 第 {rnd} 轮的 batch")
            check([(r.uid, r.chunked) for r in rest], want_rest, f"随机用例 {trial} 第 {rnd} 轮的队列")
            done = {u for u, s, e in want_batch}
            chunked = dict((u, c) for u, c in want_rest if c is not None)
            b = [PendingReq(r.uid, r.input_len, r.output_len, r.cached_len, chunked.get(r.uid)) for r in b
                 if r.uid in chunked or r.uid not in done]
            b.sort(key=lambda r: [x[0] for x in want_rest].index(r.uid))
            a = rest
