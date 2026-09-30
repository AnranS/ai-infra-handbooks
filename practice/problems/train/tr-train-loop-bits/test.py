import math

import numpy as np

from checker import check, check_close
from solution import clip_by_global_norm, decay_groups, lr_at


def test_example():
    check_close(lr_at(0, 3e-3, 3e-4, 60, 600), 5e-5, what="warmup 的第一步")
    check_close(lr_at(600, 3e-3, 3e-4, 60, 600), 3e-4, what="衰减到 min_lr")


def test_schedule():
    check_close(lr_at(59, 3e-3, 3e-4, 60, 600), 3e-3, what="warmup 的最后一步到达峰值")
    check_close(lr_at(60, 3e-3, 3e-4, 60, 600), 3e-3, what="衰减从峰值开始")
    check_close(lr_at(330, 3e-3, 3e-4, 60, 600), 3e-4 + 0.5 * 2.7e-3, what="衰减到一半")
    lrs = [lr_at(s, 3e-3, 3e-4, 60, 600) for s in range(60, 601)]
    check(all(a >= b for a, b in zip(lrs, lrs[1:])), True, "衰减阶段单调不增")
    check_close(lr_at(10, 1.0, 0.0, 0, 20), 0.5 * (1 + math.cos(math.pi * 0.5)), what="没有 warmup")
    check_close(lr_at(5, 1.0, 0.1, 5, 5), 1.0, what="max_steps == warmup 时不除以 0")


def test_clip():
    g = [np.array([3.0, 0.0]), np.array([[0.0, 4.0]])]
    new, norm = clip_by_global_norm(g, 1.0)
    check_close(norm, 5.0, what="全局范数是所有梯度拼在一起的范数")
    check_close(new[0], np.array([0.6, 0.0]) * (5.0 / (5.0 + 1e-6)), rtol=1e-6, what="第一个梯度按同一个比例缩放")
    check_close(new[1], np.array([[0.0, 0.8]]) * (5.0 / (5.0 + 1e-6)), rtol=1e-6, what="第二个梯度")
    check(new[1].shape, (1, 2), "形状不变")
    check_close(g[0], np.array([3.0, 0.0]), what="不修改传入的数组")
    small, n2 = clip_by_global_norm(g, 10.0)
    check_close((n2, float(small[1][0, 1])), (5.0, 4.0), what="没超过上限时不动")
    check(small[0] is g[0], False, "返回新的数组")
    rng = np.random.default_rng(0)
    many = [rng.standard_normal(s) for s in ((64, 32), (32,), (10, 10, 3))]
    clipped, total = clip_by_global_norm(many, 1.0)
    after = math.sqrt(sum(float(np.sum(c * c)) for c in clipped))
    check_close(after, 1.0, rtol=1e-5, what="裁剪后全局范数等于上限")
    check_close(total, math.sqrt(sum(float(np.sum(m * m)) for m in many)), what="返回裁剪前的范数")


def test_decay_groups():
    shapes = {"tok_emb.weight": (8192, 128), "blocks.0.norm1.weight": (128,), "blocks.0.attn.qkv.weight": (384, 128),
              "blocks.0.mlp.bias": (512,), "norm.weight": (128,), "conv.weight": (16, 3, 3, 3)}
    decay, no_decay = decay_groups(shapes)
    check(decay, ["tok_emb.weight", "blocks.0.attn.qkv.weight", "conv.weight"], "矩阵和词嵌入做衰减")
    check(no_decay, ["blocks.0.norm1.weight", "blocks.0.mlp.bias", "norm.weight"], "增益和偏置不做")
