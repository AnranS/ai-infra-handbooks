import numpy as np

from checker import check, raises
from solution import BaseOP, OPList


class Linear(BaseOP):
    def __init__(self, i, o):
        self.weight = np.zeros((o, i), np.float32)


class Block(BaseOP):
    def __init__(self):
        self.qkv_proj = Linear(4, 12)
        self.norm_weight = np.ones(4, np.float32)
        self._cache = np.zeros(3)


class Model(BaseOP):
    def __init__(self, n=2):
        self.embed = Linear(4, 10)
        self.layers = OPList([Block() for _ in range(n)])
        self.hidden = 4                                      # 普通属性：忽略


def test_example():
    check(list(Model().state_dict()), ["embed.weight", "layers.0.qkv_proj.weight", "layers.0.norm_weight",
                                       "layers.1.qkv_proj.weight", "layers.1.norm_weight"], "state_dict 的键（按定义顺序）")


def test_load_replaces_reference():
    m = Model()
    sd = {k: np.full(v.shape, 3, dtype=v.dtype) for k, v in m.state_dict().items()}
    arr = sd["layers.1.qkv_proj.weight"]
    backup = dict(sd)
    m.load_state_dict(sd)
    assert m.layers.op_list[1].qkv_proj.weight is arr, "应该直接替换引用，而不是拷贝"
    check(sd.keys() == backup.keys(), True, "不能修改调用者的字典")


def test_errors():
    m = Model(1)
    good = {k: np.zeros(v.shape, v.dtype) for k, v in m.state_dict().items()}
    bad_shape = dict(good, **{"layers.0.norm_weight": np.zeros(5, np.float32)})
    with raises(ValueError, "形状不对"):
        m.load_state_dict(bad_shape)
    bad_dtype = dict(good, **{"embed.weight": np.zeros((10, 4), np.float16)})
    with raises(ValueError, "dtype 不对"):
        Model(1).load_state_dict(bad_dtype)
    missing = dict(good)
    del missing["embed.weight"]
    with raises(KeyError, "缺少键"):
        Model(1).load_state_dict(missing)
    extra = dict(good, **{"lm_head.weight": np.zeros(3)})
    try:
        Model(1).load_state_dict(extra)
    except KeyError as e:
        assert "lm_head.weight" in str(e), f"错误信息应该列出多余的键：{e}"
    else:
        raise AssertionError("有多余的键时应该抛出 KeyError")


def test_nested_lists():
    class Expert(BaseOP):
        def __init__(self):
            self.w = np.zeros(2, np.float32)

    class MoE(BaseOP):
        def __init__(self):
            self.experts = OPList([Expert(), Expert(), Expert()])

    check(list(MoE().state_dict(prefix="mlp")), ["mlp.experts.0.w", "mlp.experts.1.w", "mlp.experts.2.w"], "带前缀")
