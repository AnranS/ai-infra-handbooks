import random

import numpy as np

from checker import check, raises
from solution import load_weights


def dense_ckpt(layers=2, seed=0):
    rng = np.random.default_rng(seed)
    w = {}
    for l in range(layers):
        p = f"model.layers.{l}"
        w[f"{p}.self_attn.q_proj.weight"] = rng.standard_normal((8, 4))
        w[f"{p}.self_attn.k_proj.weight"] = rng.standard_normal((2, 4))
        w[f"{p}.self_attn.v_proj.weight"] = rng.standard_normal((2, 4))
        w[f"{p}.self_attn.o_proj.weight"] = rng.standard_normal((4, 8))
        w[f"{p}.mlp.gate_proj.weight"] = rng.standard_normal((6, 4))
        w[f"{p}.mlp.up_proj.weight"] = rng.standard_normal((6, 4))
        w[f"{p}.mlp.down_proj.weight"] = rng.standard_normal((4, 6))
        w[f"{p}.input_layernorm.weight"] = rng.standard_normal(4)
    w["model.embed_tokens.weight"] = rng.standard_normal((10, 4))
    return w


def test_example():
    w = dense_ckpt(1)
    out = dict(load_weights(w.items()))
    p = "model.layers.0"
    check(sorted(out), sorted([f"{p}.self_attn.qkv_proj.weight", f"{p}.self_attn.o_proj.weight",
                               f"{p}.mlp.gate_up_proj.weight", f"{p}.mlp.down_proj.weight",
                               f"{p}.input_layernorm.weight", "model.embed_tokens.weight"]), "输出的名字")
    qkv = np.concatenate([w[f"{p}.self_attn.{x}_proj.weight"] for x in "qkv"])
    check(out[f"{p}.self_attn.qkv_proj.weight"], qkv, "qkv_proj = [q; k; v]")
    check(out[f"{p}.mlp.gate_up_proj.weight"], np.concatenate([w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.up_proj.weight"]]),
          "gate_up_proj = [gate; up]")


def test_any_order_and_streaming():
    w = dense_ckpt(4, seed=1)
    items = list(w.items())
    random.Random(0).shuffle(items)
    pulled = []

    def source():
        for it in items:
            pulled.append(it[0])
            yield it

    gen = load_weights(source())
    first = next(gen)
    assert len(pulled) < len(items), "拿到第一个输出之前就读完了所有权重：不是流式的"
    rest = dict([first] + list(gen))
    check(len(rest), 4 * 5 + 1, "输出的张量数")


def test_experts():
    rng = np.random.default_rng(2)
    E = 4
    w = {}
    for e in range(E):
        w[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"] = rng.standard_normal((3, 2))
        w[f"model.layers.0.mlp.experts.{e}.up_proj.weight"] = rng.standard_normal((3, 2))
        w[f"model.layers.0.mlp.experts.{e}.down_proj.weight"] = rng.standard_normal((2, 3))
    w["model.layers.0.mlp.gate.weight"] = rng.standard_normal((E, 2))
    items = list(w.items())
    random.Random(1).shuffle(items)
    out = dict(load_weights(items, num_experts=E))
    check(sorted(out), ["model.layers.0.mlp.experts.down_proj", "model.layers.0.mlp.experts.gate_up_proj",
                        "model.layers.0.mlp.gate.weight"], "打包后的名字")
    gu = out["model.layers.0.mlp.experts.gate_up_proj"]
    check(gu.shape, (E, 6, 2), "gate_up 打包后的形状")
    for e in range(E):
        check(gu[e], np.concatenate([w[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"],
                                     w[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]]), f"专家 {e}")


def test_incomplete_group():
    w = dense_ckpt(1)
    del w["model.layers.0.self_attn.v_proj.weight"]
    with raises(ValueError, "缺少 v_proj"):
        list(load_weights(w.items()))
