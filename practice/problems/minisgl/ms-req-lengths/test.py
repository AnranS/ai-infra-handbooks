from checker import check, raises
from solution import Req


def snap(r):
    return (r.cached_len, r.device_len, r.extend_len, r.remain_len, r.can_decode)


def test_example():
    r = Req([1, 2, 3, 4, 5], cached_len=2, output_len=3)
    check(snap(r), (2, 5, 3, 3, True), "创建时 (cached, device, extend, remain, can_decode)")
    r.complete_one()
    check(snap(r), (5, 6, 1, 2, True), "prefill 之后")
    r.append_host(9)
    r.complete_one()
    r.append_host(8)
    r.complete_one()
    check(snap(r), (7, 8, 1, 0, False), "两轮 decode 之后")
    check(r.input_ids, [1, 2, 3, 4, 5, 9, 8], "input_ids")
    check(r.max_device_len, 8, "max_device_len 不变")


def test_validation():
    with raises(ValueError, "cached_len == len(input_ids)"):
        Req([1, 2], cached_len=2, output_len=1)
    with raises(ValueError, "cached_len < 0"):
        Req([1, 2], cached_len=-1, output_len=1)
    with raises(ValueError, "output_len < 0"):
        Req([1, 2], cached_len=0, output_len=-1)
    check(snap(Req([7], 0, 0)), (0, 1, 1, 0, False), "output_len = 0：只做 prefill")


def test_overlap_timing():
    """complete_one 先于 append_host：两者之间 len(input_ids) == device_len - 1"""
    r = Req([1, 2, 3], 0, 5)
    r.complete_one()
    check(len(r.input_ids), r.device_len - 1, "发射之后、拿到结果之前")
    r.append_host(4)
    check(len(r.input_ids), r.device_len, "拿到结果之后")
