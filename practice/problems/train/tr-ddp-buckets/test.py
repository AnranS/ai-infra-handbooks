from checker import check, check_close
from solution import comm_timeline, exposed_comm, make_buckets


def close_all(actual, expected, what, **tol):
    """逐个元素比较（浏览器里没有 numpy，不能直接对列表用 check_close）"""
    flat = lambda xs: [v for x in xs for v in (x if isinstance(x, (list, tuple)) else [x])]
    check(len(actual), len(expected), f"{what}：个数")
    for i, (a, e) in enumerate(zip(flat(actual), flat(expected))):
        check_close(a, e, what=f"{what}（第 {i} 个数）", **tol)


def test_example():
    check(make_buckets([4, 4, 4, 4], cap=8), [[3, 2], [1, 0]], "题目中的例子")


def test_buckets():
    check(make_buckets([4, 4, 4, 4], cap=8), [[3, 2], [1, 0]], "每个桶放两个参数")
    check(make_buckets([1, 2, 3, 4], cap=5), [[3], [2, 1], [0]], "从最后一个参数往前放")
    check(make_buckets([2, 20, 2], cap=5), [[2], [1], [0]], "超过上限的参数独占一个桶")
    check(make_buckets([1, 1, 1], cap=100), [[2, 1, 0]], "上限很大时只有一个桶")


def test_timeline():
    sizes = [4, 4, 4, 4]
    ready = [4.0, 3.0, 2.0, 1.0]                     # 反向从最后一个参数开始
    buckets = make_buckets(sizes, 8)
    close_all(comm_timeline(ready, buckets, sizes, bw=4), [(2.0, 4.0), (4.0, 6.0)], what="两个桶")
    check_close(exposed_comm(ready, buckets, sizes, bw=4), 2.0, what="露出来的是最后一个桶的通信")


def test_one_bucket_no_overlap():
    sizes = [4, 4, 4, 4]
    ready = [4.0, 3.0, 2.0, 1.0]
    one = make_buckets(sizes, 1000)
    check_close(exposed_comm(ready, one, sizes, bw=4), 4.0, what="只有一个桶时通信完全不重叠")
    per_param = make_buckets(sizes, 4)
    close_all(comm_timeline(ready, per_param, sizes, bw=4), [(1, 2), (2, 3), (3, 4), (4, 5)], what="每个参数一个桶")
    check_close(exposed_comm(ready, per_param, sizes, bw=4), 1.0, what="每个参数一个桶")


def test_comm_bound():
    sizes = [8, 8, 8]
    ready = [0.3, 0.2, 0.1]                          # 计算很快，通信成为瓶颈：桶要排队
    buckets = make_buckets(sizes, 8)
    close_all(comm_timeline(ready, buckets, sizes, bw=8), [(0.1, 1.1), (1.1, 2.1), (2.1, 3.1)], what="通信排队")
    check_close(exposed_comm(ready, buckets, sizes, bw=8), 2.8, what="通信排队")
