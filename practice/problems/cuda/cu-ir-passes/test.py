from checker import check
from solution import cse, const_fold, dce, optimize


def run(instrs, env=None):
    """解释执行这段 IR，返回所有 store 的结果"""
    env = dict(env or {})
    out = {}
    for dst, op, args in instrs:
        if op == "const":
            env[dst] = args[0]
        elif op == "store":
            out[dst] = env[args[0]]
        else:
            x, y = env[args[0]], env[args[1]]
            env[dst] = x + y if op == "+" else x - y if op == "-" else x * y
    return out


IR = [("a", "const", (2.0,)), ("t1", "+", ("x", "a")), ("t2", "+", ("x", "a")),
      ("t3", "*", ("t1", "t2")), ("t4", "*", ("x", "x")), ("out", "store", ("t3",))]
ENV = {"x": 5.0}


def test_example():
    check([i[0] for i in optimize(IR)], ["a", "t1", "t3", "out"], "t2 被合并、t4 没人用")
    check(run(optimize(IR), ENV), run(IR, ENV), "结果不变")


def test_all_constant():
    ir = [("a", "const", (2.0,)), ("b", "const", (3.0,)), ("c", "+", ("a", "b")),
          ("out", "store", ("c",))]
    got = optimize(ir)
    check(run(got), {"out": 5.0}, "整段都能在编译期算出来")
    check(len(got) <= 2, True, "只剩常量和 store")


def test_const_fold():
    got = const_fold([("a", "const", (2.0,)), ("b", "const", (3.0,)), ("c", "*", ("a", "b"))])
    check(got[-1], ("c", "const", (6.0,)), "两个常量相乘直接算出来")
    check(len(got), 3, "折叠后仍然要留下 const 指令")


def test_const_fold_with_variable():
    ir = [("a", "const", (2.0,)), ("c", "+", ("a", "x"))]
    check(const_fold(ir)[-1], ("c", "+", ("a", "x")), "有变量参与时不能折叠")


def test_cse_renames():
    ir = [("t1", "+", ("x", "y")), ("t2", "+", ("x", "y")), ("t3", "*", ("t2", "t2")),
          ("out", "store", ("t3",))]
    got = cse(ir)
    check(len(got), 3, "重复的表达式只算一次")
    check(got[1], ("t3", "*", ("t1", "t1")), "后续指令的参数要改名")


def test_dce_transitive():
    ir = [("t1", "const", (1.0,)), ("t2", "+", ("t1", "t1")), ("t3", "*", ("t2", "t2")),
          ("t4", "const", (9.0,)), ("out", "store", ("t3",))]
    got = dce(ir)
    check([i[0] for i in got], ["t1", "t2", "t3", "out"], "间接用到的不能删，没用到的要删")


def test_semantics_preserved():
    import random
    rng = random.Random(4)
    for _ in range(200):
        ir, names = [], ["x", "y"]
        for k in range(12):
            if rng.random() < 0.3:
                dst = f"c{k}"
                ir.append((dst, "const", (float(rng.randrange(1, 5)),)))
            else:
                dst = f"t{k}"
                op = rng.choice(["+", "-", "*"])
                ir.append((dst, op, (rng.choice(names), rng.choice(names))))
            names.append(dst)
        ir.append(("out", "store", (names[-1],)))
        env = {"x": 2.0, "y": 3.0}
        check(run(optimize(ir), env), run(ir, env), "优化不能改变语义")


def test_no_growth():
    check(len(optimize(IR)) <= len(IR), True, "优化后指令不会变多")
    already = optimize(IR)
    check(optimize(already), already, "再优化一次不变（到了不动点）")
