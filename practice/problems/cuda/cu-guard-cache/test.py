from checker import check
from solution import GuardCache


def run(cache, batches, tail=((16, 16),)):
    return [cache.call(((b, 16),) + tuple(tail)) for b in batches]


def test_example():
    c = GuardCache()
    check(run(c, (4, 4, 8, 16, 32)), ["compile", "hit", "compile", "hit", "hit"], "本章 recompile.py 的规律")
    check(c.guards(), [((4, 16), (16, 16)), (("dyn", 16), (16, 16))], "两个版本的 guard")
    check(c.num_compiles(), 2, "编译次数")


def test_size_one_specialization():
    c = GuardCache()
    check(run(c, (4, 8, 1, 1, 2, 64)), ["compile", "compile", "compile", "hit", "hit", "hit"], "batch=1 要单独编译一个静态版本")
    check(c.guards()[2], ((1, 16), (16, 16)), "大小为 1 的维度按具体大小编译")
    c2 = GuardCache()
    check(run(c2, (1, 1, 5, 9)), ["compile", "hit", "compile", "hit"], "从 1 开始：第二次编译标成动态")
    check(c2.guards()[1], (("dyn", 16), (16, 16)), "5 与第一次的 1 不同：变成动态")


def test_mark_dynamic():
    c = GuardCache(mark_dynamic={(0, 0)})
    check(run(c, (4, 8, 16, 100)), ["compile", "hit", "hit", "hit"], "事先标成动态：只编译一次")
    check(c.guards(), [(("dyn", 16), (16, 16))], "第一次编译就是动态的")
    check(c.call(((1, 16), (16, 16))), "compile", "标成动态也不包括大小 1")


def test_multiple_dims_and_newest_first():
    c = GuardCache()
    seq = [((2, 128, 64),), ((2, 256, 64),), ((4, 256, 64),), ((8, 512, 64),)]
    check([c.call(s) for s in seq], ["compile", "compile", "compile", "hit"], "序列长度先变、batch 后变")
    check(c.guards(), [((2, 128, 64),), ((2, "dyn", 64),), (("dyn", "dyn", 64),)], "每次重新编译都把新变化的维度标成动态")
    check(c.call(((2, 300, 64),)), "hit", "最新的版本最先检查：命中全动态的版本")
    check(c.call(((2, 300, 32),)), "compile", "最后一维变了")
    check(c.guards()[-1], (("dyn", "dyn", "dyn"),), "三维都动态了")


def test_cache_limit():
    c = GuardCache(cache_limit=3)
    shapes = [((4,),), ((4, 4),), ((4, 4, 4),), ((4, 4, 4, 4),), ((4,),)]
    check([c.call(s) for s in shapes], ["compile", "compile", "compile", "eager", "hit"], "维数变化没法动态化；到上限之后退回 eager")
    check(c.num_compiles(), 3, "不超过上限")
