import math
import struct

import numpy as np

from checker import check
from solution import bf16_ulp_at, from_bf16_bits, to_bf16_bits


def ref_bits(f):
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    if math.isnan(f):
        return 0x7FC0
    upper, lower = u >> 16, u & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return upper & 0xFFFF


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_example():
    check(int(to_bf16_bits(np.float32(1.0))), 0x3F80, "1.0 的 bf16 位模式")
    check(float(from_bf16_bits(np.uint16(0x3F80))), 1.0, "0x3F80 -> 1.0")
    check(bf16_ulp_at(1.0), 2 ** -7, "bf16_ulp_at(1.0)")


def test_round_to_nearest_even():
    cases = {
        0x3F808000: 0x3F80,   # 正好一半，保留部分最低位是 0 -> 不进位
        0x3F818000: 0x3F82,   # 正好一半，最低位是 1 -> 进位到偶数
        0x3F808001: 0x3F81,   # 略大于一半 -> 进位
        0x3F807FFF: 0x3F80,   # 略小于一半 -> 舍去
        0xBF818000: 0xBF82,   # 负数同理
    }
    for bits, want in cases.items():
        got = int(to_bf16_bits(np.array([f32(bits)], dtype=np.float32))[0])
        check(got, want, f"float32 位模式 {bits:#010x} 舍入后")


def test_specials():
    x = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, np.finfo(np.float32).max], dtype=np.float32)
    got = to_bf16_bits(x)
    check(got.dtype, np.dtype(np.uint16), "结果类型")
    check([int(v) for v in got[:4]], [0x0000, 0x8000, 0x7F80, 0xFF80], "±0、±inf")
    back = from_bf16_bits(got)
    assert np.isnan(back[4]), "NaN 要保持是 NaN"
    assert np.isinf(back[5]), "接近 float32 最大值的数舍入后变成 inf"


def test_random_against_reference():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.standard_normal(2000).astype(np.float32) * 10 ** rng.uniform(-30, 30, 2000).astype(np.float32),
                        rng.integers(0, 2 ** 32, 2000, dtype=np.uint64).astype(np.uint32).view(np.float32)])
    got = to_bf16_bits(x)
    want = np.array([ref_bits(float(v)) for v in x], dtype=np.uint16)
    check(got, want, "4000 个随机 float32 的舍入结果")


def test_roundtrip_and_ulp():
    b = np.arange(0, 0x7F80, 37, dtype=np.uint16)
    check(to_bf16_bits(from_bf16_bits(b)), b, "bf16 -> float32 -> bf16 往返不变")
    check(bf16_ulp_at(256.0), 2.0, "256 附近的间隔")
    check(float(from_bf16_bits(to_bf16_bits(np.float32(257.0)))), 256.0, "257 在 bf16 里舍入到 256")
    check(bf16_ulp_at(3.0), 2 ** -6, "3.0 附近的间隔")
