import math

from checker import check, check_close, raises
from solution import clip_loss, group_advantages, informative_groups


def close_all(actual, expected, what, **tol):
    """逐个元素比较（浏览器里没有 numpy，不能直接对列表用 check_close）"""
    flat = lambda xs: [v for x in xs for v in (x if isinstance(x, (list, tuple)) else [x])]
    check(len(actual), len(expected), f"{what}：个数")
    for i, (a, e) in enumerate(zip(flat(actual), flat(expected))):
        check_close(a, e, what=f"{what}（第 {i} 个数）", **tol)


def test_example():
    s = 1 / math.sqrt(2)
    close_all(group_advantages([1, 0, 1, 0], 2), [s, -s, s, -s], rtol=1e-4, what="两组，每组一对一错")
    check(informative_groups([1, 1, 0, 1, 0, 0], 2), [1], "只有第 1 组有信号")


def test_advantages():
    adv = group_advantages([1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5], 4)
    close_all(adv[:4], [1.5, -0.5, -0.5, -0.5], rtol=1e-4, what="一组里只有一条做对（样本标准差 0.5）")
    close_all(adv[4:], [0, 0, 0, 0], atol=1e-9, what="组内奖励相同时优势全为 0")
    a = group_advantages([3.0, 1.0, 2.0], 3)
    check_close(sum(a), 0.0, atol=1e-9, what="组内优势之和为 0")
    with raises(ValueError):
        group_advantages([1, 0, 1], 2)
    with raises(ValueError):
        group_advantages([1, 0], 1)


def test_clip_loss():
    check_close(clip_loss([1.0, 1.0], [2.0, -1.0]), -0.5, what="比率为 1 时就是优势的平均值的相反数")
    check_close(clip_loss([1.5], [1.0]), -1.2, what="正优势：比率超过 1.2 的部分被裁掉")
    check_close(clip_loss([0.5], [1.0]), -0.5, what="正优势、比率变小：不裁剪（取较小的那个）")
    check_close(clip_loss([0.5], [-1.0]), 0.8, what="负优势：比率低于 0.8 的部分被裁掉")
    check_close(clip_loss([1.5], [-1.0]), 1.5, what="负优势、比率变大：不裁剪")


def test_informative():
    check(informative_groups([0, 0, 1, 1, 0, 1], 2), [2], "全错、全对的组都被过滤")
    check(informative_groups([0.2, 0.3, 0.2, 0.2], 4), [0], "有一条不同就算有信号")
    check(informative_groups([1, 1, 1, 1], 2), [], "全部没有信号")
