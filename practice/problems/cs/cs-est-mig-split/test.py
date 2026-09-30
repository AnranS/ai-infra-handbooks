from checker import check, check_close
from solution import best_split, decode_ms, mig_profile

H100 = [("1g.10gb", 1, 10), ("2g.20gb", 2, 20), ("3g.40gb", 3, 40), ("7g.80gb", 7, 80)]


def test_example():
    p = mig_profile(3, 40)
    check(p["sms"], 57, "3/7 的 SM")
    check_close(p["bw_gbs"], 3350 * 3 / 7, rtol=1e-9, what="带宽也按份切")
    check_close(decode_ms(14, p), 9.8, rtol=1e-2, what="7B 的 BF16 权重在 3g.40gb 上")
    check(best_split(3, 30, H100[:3]), ("1g.10gb", 7), "小模型切 7 份")


def test_full_card():
    p = mig_profile(7, 80)
    check(p["sms"], 132, "整卡")
    check_close(decode_ms(14, p), 4.2, rtol=1e-2, what="整卡上的 decode 下限")


def test_does_not_fit():
    check(decode_ms(14, mig_profile(1, 10)), None, "14 GB 权重放不进 10 GB")
    check(best_split(14, 100, [("1g.10gb", 1, 10)]), None, "没有档位放得下")


def test_small_model_many_instances():
    check(best_split(3, 10, H100), ("1g.10gb", 7), "1.5B 模型：6.3 ms < 10 ms，切 7 份")
    check(best_split(3, 3, H100), ("3g.40gb", 2), "目标 3 ms 时只有 3g 和整卡够快，3g 能切两份")


def test_big_model_needs_full_card():
    check(best_split(70, 60, H100), ("7g.80gb", 1), "35B 的 BF16 权重只有整卡放得下")
    check(best_split(140, 60, H100), None, "70B 一张卡都放不下")
