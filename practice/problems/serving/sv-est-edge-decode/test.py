from checker import check, check_close, raises
from solution import bits_per_weight, decode_tps, kv_bytes, pick_format, prefill_seconds


def test_example():
    check_close(decode_tps(8e9 * bits_per_weight("Q4_K") / 8, 0, 60), 0.7 * 60 / 4.5, what="8B、Q4_K、60 GB/s")
    check(pick_format(3e9, 20, 60), "Q4_K", "3B 模型想要每秒 20 个 token")


def test_bits():
    for fmt, bits in (("F16", 16), ("Q8_0", 8.5), ("Q6_K", 6.5625), ("Q4_K", 4.5), ("Q4_1", 5.0), ("Q4_0", 4.5)):
        check_close(bits_per_weight(fmt), bits, what=f"{fmt} 的比特数")
    with raises(ValueError, "不认识的格式"):
        bits_per_weight("Q3_X")


def test_kv_and_context():
    check_close(kv_bytes(36, 8, 128, 8192), 2 * 36 * 8 * 128 * 8192 * 2, what="Qwen3-8B 的 8K 上下文，bf16")
    check_close(kv_bytes(28, 8, 128, 4096, kv_bits=8), 2 * 28 * 8 * 128 * 4096, what="8 比特的 KV")
    w = 8e9 * 4.5 / 8
    slow = decode_tps(w, kv_bytes(36, 8, 128, 8192), 60)
    check_close(slow, 0.7 * 60e9 / (w + 2 * 36 * 8 * 128 * 8192 * 2), what="上下文 8K 时要多读 KV")
    check(slow < decode_tps(w, 0, 60), True, "上下文越长越慢")
    check_close(decode_tps(1e9, 0, 100, efficiency=1.0), 100.0, what="效率 100% 时")


def test_pick_format():
    check(pick_format(1e9, 20, 60), "F16", "1B 模型用 F16 也够快")
    check(pick_format(3e9, 10, 60), "Q8_0", "要求降低到每秒 10 个 token")
    check(pick_format(8e9, 9, 60), "Q4_K", "8B 模型每秒 9 个 token")
    check(pick_format(70e9, 20, 60), None, "70B 模型在手机上怎么量化都达不到")
    check(pick_format(8e9, 20, 400), "Q8_0", "笔记本 400 GB/s")


def test_prefill():
    check_close(prefill_seconds(8e9, 1000, 20), 2.0, what="8B、1000 个 token、20 TFLOPS 的 NPU、四成利用率")
    check_close(prefill_seconds(3e9, 4000, 50, utilization=0.5), 2 * 3e9 * 4000 / 25e12, what="换一组参数")
