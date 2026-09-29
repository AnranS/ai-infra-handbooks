from checker import check, check_close
from solution import bytes_per_param, decode_floor, min_gpus, speedup_needed


def test_example():
    check_close(bytes_per_param(4, 32, 2), 0.5625, what="INT4 + bf16 缩放")
    check(min_gpus(1e12, 1.0, 141e9), 11, "FP8 万亿模型")


def test_formats():
    check_close(bytes_per_param(4, 32, 1), 0.53125, what="MXFP4")
    check_close(bytes_per_param(4, 16, 1), 0.5625, what="NVFP4（不计张量级缩放）")
    check_close(bytes_per_param(16, None, 0), 2.0, what="BF16")
    check(min_gpus(1e12, 0.5625, 141e9), 6, "4 比特一台机器就够")
    check(min_gpus(1e12, 2.0, 141e9), 21, "BF16")


def test_decode():
    check_close(decode_floor(32e9, 1.0, 8, 4.8e12), 32e9 / (8 * 4.8e12), what="FP8")
    check_close(speedup_needed(1.0, 0.5625), 1 / 0.5625, what="FP8 → INT4")
    assert speedup_needed(1.0, 0.5625) < 2, "缩放因子也要读，达不到 2 倍"
