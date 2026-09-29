from checker import check, raises
from solution import Bounded, Field, OneOf, Positive, Typed


class ModelConfig:
    hidden_size = Positive(int)
    num_heads = Positive(int)
    dropout = Bounded(float, 0.0, 1.0)
    dtype = OneOf("float16", "bfloat16", "float32")
    name = Typed(str)

    def __init__(self, hidden_size, num_heads, dropout=0.0, dtype="bfloat16"):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.dtype = dtype


def expect(exc, fn, field):
    try:
        fn()
    except exc as e:
        assert field in str(e), f"错误信息应该包含字段名 {field!r}，实际是：{e}"
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"应该抛出 {exc.__name__}，实际抛出了 {type(e).__name__}: {e}") from None
    raise AssertionError(f"应该抛出 {exc.__name__}")


def test_example():
    cfg = ModelConfig(1024, 16)
    check((cfg.hidden_size, cfg.num_heads, cfg.dropout, cfg.dtype), (1024, 16, 0.0, "bfloat16"), "字段值")
    expect(ValueError, lambda: setattr(cfg, "num_heads", 0), "num_heads")
    expect(ValueError, lambda: setattr(cfg, "dtype", "int4"), "dtype")
    assert isinstance(ModelConfig.hidden_size, Field), "通过类访问应该返回描述符本身"


def test_types():
    cfg = ModelConfig(8, 2)
    expect(TypeError, lambda: setattr(cfg, "hidden_size", 8.0), "hidden_size")
    expect(TypeError, lambda: setattr(cfg, "hidden_size", True), "hidden_size")
    expect(TypeError, lambda: setattr(cfg, "dropout", "0.1"), "dropout")
    cfg.dropout = 1
    check((cfg.dropout, type(cfg.dropout)), (1.0, float), "float 字段接受 int 并转成 float")
    expect(ValueError, lambda: setattr(cfg, "dropout", 1.5), "dropout")
    expect(ValueError, lambda: ModelConfig(-1, 2), "hidden_size")


def test_per_instance_storage():
    a, b = ModelConfig(8, 2), ModelConfig(16, 4)
    a.hidden_size = 32
    check((a.hidden_size, b.hidden_size), (32, 16), "两个实例互不影响")
    check(a.__dict__["hidden_size"], 32, "值存在实例的 __dict__ 里")


def test_unset_field():
    cfg = ModelConfig(8, 2)
    expect(AttributeError, lambda: cfg.name, "name")
    cfg.name = "qwen"
    check(cfg.name, "qwen", "cfg.name")


def test_reuse_in_another_class():
    class Other:
        k = Positive(int)
        p = Bounded(float, 0, 1)

    o = Other()
    o.k = 3
    expect(ValueError, lambda: setattr(o, "p", -0.5), "p")
    check(o.k, 3, "o.k")
