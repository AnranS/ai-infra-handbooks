from checker import check, check_close
from solution import choose, decode_ms, fits, mig_share


def test_example():
    count, mem, share = mig_share("2g.20gb")
    check((count, mem), (3, 20), "一张卡切 3 份，每份 20 GB")
    check_close(share, 2 / 7, rtol=1e-9, what="算力份额")
    check(fits("1g.10gb", 3, 4), True, "7 GB 放得进 10 GB")
    check_close(decode_ms(3, 3350, 1 / 7), 6.27, rtol=0.01, what="按份额打折后的 decode")
    check(choose(3, 4, 30, tenants=5), "1g.10gb", "小模型切到最细")
    check(choose(14, 6, 30, tenants=3), "2g.20gb", "7B 模型最小只能用 2g")


def test_profiles():
    check(mig_share("1g.10gb")[0], 7, "最多切 7 份")
    check(mig_share("7g.80gb")[0], 1, "整卡")
    check_close(mig_share("7g.80gb")[2], 1.0, rtol=1e-9, what="整卡份额为 1")


def test_fits():
    check(fits("1g.10gb", 8, 3), False, "权重加 KV 超了")
    check(fits("3g.40gb", 14, 20), True, "40 GB 够用")
    check(fits("2g.20gb", 20, 0), True, "正好装满")
    check(fits("2g.20gb", 20, 1), False, "多一点就放不下")


def test_decode_share():
    full = decode_ms(14, 3350)
    third = decode_ms(14, 3350, 3 / 7)
    check(third > full, True, "份额小则更慢")
    check_close(third / full, 7 / 3, rtol=1e-9, what="正好是份额的倒数")


def test_choose_constraints():
    check(choose(3, 4, 1, tenants=5), None, "延迟预算太紧，哪个档位都不满足")
    check(choose(60, 10, 100, tenants=1), "7g.80gb", "大模型只能整卡")
    check(choose(60, 30, 100, tenants=1), None, "整卡也放不下")
    check(choose(3, 2, 100, tenants=8), None, "一张卡最多 7 份")


def test_choose_prefers_smallest():
    check(choose(3, 1, 100, tenants=1), "1g.10gb", "能切细就切细，省卡")
    check(choose(14, 4, 100, tenants=1), "2g.20gb", "装得下的最小档位")
