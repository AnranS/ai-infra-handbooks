from checker import check
from solution import block_keys, longest_hit, plan_load


def test_example():
    keys = block_keys(list(range(40)), 16)
    check(len(keys), 2, "40 个 token、块大小 16：两个完整的块")
    check(longest_hit(keys, [set(), {keys[0]}, {keys[1]}]), [1, 2], "题目中的例子")


def test_keys():
    a = list(range(100, 164))
    b = a[:20] + [0] + a[21:]
    ka, kb = block_keys(a, 16), block_keys(b, 16)
    check(ka[0] == kb[0], True, "第 0 块相同")
    check([x == y for x, y in zip(ka[1:], kb[1:])], [False, False, False], "第 20 个 token 不同：第 1 块及之后全部不同")
    check(block_keys(a, 16, seed=1)[0] == ka[0], False, "不同的种子（模型版本、LoRA）得到不同的键")
    check(block_keys(a[:15], 16), [], "不满一块没有键")
    check(block_keys(a, 16)[:2], block_keys(a[:40], 16), "键只取决于前缀")


def test_longest_hit():
    keys = block_keys(list(range(64)), 16)
    tiers = [{keys[0]}, {keys[0], keys[1], keys[3]}, set()]
    check(longest_hit(keys, tiers), [0, 1], "第 2 块不命中，第 3 块即使在也用不了")
    check(longest_hit(keys, [set()]), [], "全部不命中")
    check(longest_hit(keys, [set(keys), set(keys)]), [0, 0, 0, 0], "取最快的那一层")


def test_plan_load():
    keys = block_keys(list(range(64)), 16)
    tiers = [{keys[0]}, {keys[1]}, {keys[2]}]
    check(plan_load(keys, tiers, 16, [0.0, 0.1, 5.0], 1.0), (32, 32), "第 2 块在 SSD 上，读回比重算还贵")
    check(plan_load(keys, tiers, 16, [0.0, 0.1, 0.5], 1.0), (48, 16), "第 3 块未命中")
    check(plan_load(keys, [{keys[1]}, {keys[0]}], 16, [0.0, 2.0], 1.0), (0, 64), "第 0 块选择重算，后面的块也只能重算")
