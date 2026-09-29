import numpy as np

from checker import check, check_close, raises
from solution import TiedLM


def setup(V=10, d=4, seed=0):
    return np.random.default_rng(seed).standard_normal((V, d))


def test_example():
    E = setup()
    lm = TiedLM(E)
    ids, mask = lm.pad_batch([[1, 2, 3], [4]], pad_id=0)
    check(ids, np.array([[1, 2, 3], [4, 0, 0]]), "ids")
    check(mask, np.array([[True, True, True], [True, False, False]]), "mask")
    out = lm.last_token_logits([[1, 2, 3], [4]], pad_id=0)
    check_close(out, np.stack([E[3] @ E.T, E[4] @ E.T]), what="最后一个真实 token 的 logits")


def test_embed_shapes_and_bounds():
    E = setup()
    lm = TiedLM(E)
    check(lm.embed(np.array([[1, 2], [3, 4]])).shape, (2, 2, 4), "二维 ids 的嵌入形状")
    check_close(lm.embed(np.array(7)), E[7], what="标量 id")
    with raises(IndexError, "id = 10"):
        lm.embed(np.array([10]))
    with raises(IndexError, "id = -1"):
        lm.embed(np.array([-1]))


def test_logits_tied():
    E = setup(V=6, d=3, seed=1)
    lm = TiedLM(E)
    h = np.random.default_rng(2).standard_normal((2, 5, 3))
    check_close(lm.logits(h), h @ E.T, what="logits = h @ E.T")
    check(lm.logits(h).shape, (2, 5, 6), "logits 形状")


def test_pad_batch_details():
    lm = TiedLM(setup())
    ids, mask = lm.pad_batch([[5], [1, 2, 3, 4], [], [9, 9]], pad_id=-1)
    check(ids.dtype, np.dtype(np.int64), "ids 的类型")
    check(ids.tolist(), [[5, -1, -1, -1], [1, 2, 3, 4], [-1] * 4, [9, 9, -1, -1]], "ids")
    check(mask.sum(axis=1).tolist(), [1, 4, 0, 2], "每行真实 token 数")


def test_last_token_with_pad_equal_to_real_id():
    """pad_id 恰好也是一个合法 token 时，也要取对位置"""
    E = setup(seed=3)
    lm = TiedLM(E)
    out = lm.last_token_logits([[2, 7, 7], [7, 1], [3]], pad_id=7)
    check_close(out, np.stack([E[7] @ E.T, E[1] @ E.T, E[3] @ E.T]), what="pad_id=7 时的 logits")
