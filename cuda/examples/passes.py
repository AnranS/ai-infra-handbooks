# 中间表示（SSA 风格的三地址码）和几个最基本的优化 pass
from dataclasses import dataclass


@dataclass
class Instr:
    dst: str
    op: str                                    # "const" / "+" / "*" / "load" / "store"
    args: tuple

    def __str__(self):
        if self.op == "const":
            return f"{self.dst} = {self.args[0]:g}"
        if self.op in ("load", "store"):
            return f"{self.dst} = {self.op} {self.args[0]}" if self.op == "load" else f"store {self.args[0]} -> {self.dst}"
        return f"{self.dst} = {self.args[0]} {self.op} {self.args[1]}"


# 一段手写的 IR：把 (a+b)*(a+b) + (a+b)*2 存到 out，中间故意重复算了几次 a+b，还有一条没人用的指令
program = [
    Instr("t1", "load", ("a",)),
    Instr("t2", "load", ("b",)),
    Instr("t3", "+", ("t1", "t2")),
    Instr("t4", "+", ("t1", "t2")),            # 和 t3 完全一样
    Instr("t5", "*", ("t3", "t4")),
    Instr("t6", "const", (2.0,)),
    Instr("t7", "+", ("t1", "t2")),            # 又一次
    Instr("t8", "*", ("t7", "t6")),
    Instr("t9", "+", ("t5", "t8")),
    Instr("t10", "*", ("t1", "t2")),           # 算了但没人用
    Instr("out", "store", ("t9",)),
]


def cse(instrs):
    """公共子表达式消除：同样的 (op, 参数) 只算一次，后面的引用改成第一次的结果"""
    seen, rename, out = {}, {}, []
    for ins in instrs:
        args = tuple(rename.get(a, a) for a in ins.args)
        key = (ins.op, args)
        if ins.op in ("+", "*") and key in seen:
            rename[ins.dst] = seen[key]        # 重复的表达式：直接复用
            continue
        seen[key] = ins.dst
        out.append(Instr(ins.dst, ins.op, args))
    return out


def dce(instrs):
    """死代码消除：从最后的输出反向标记用到的值，没被用到的指令删掉"""
    live = {ins.dst for ins in instrs if ins.op == "store"} | {
        a for ins in instrs if ins.op == "store" for a in ins.args}
    for ins in reversed(instrs):
        if ins.dst in live or ins.op == "store":
            live |= set(ins.args)
    return [ins for ins in instrs if ins.dst in live or ins.op == "store"]


def run(instrs, a, b):
    """解释执行这段 IR，用来验证优化前后结果一致"""
    env = {"a": a, "b": b}
    for ins in instrs:
        if ins.op == "const":
            env[ins.dst] = ins.args[0]
        elif ins.op == "load":
            env[ins.dst] = env[ins.args[0]]
        elif ins.op == "store":
            env[ins.dst] = env[ins.args[0]]
        else:
            x, y = env[ins.args[0]], env[ins.args[1]]
            env[ins.dst] = x + y if ins.op == "+" else x * y
    return env["out"]


print("优化前：")
for ins in program:
    print("   ", ins)
after_cse = cse(program)
after_dce = dce(after_cse)
print("\n经过公共子表达式消除和死代码消除：")
for ins in after_dce:
    print("   ", ins)
print(f"\n指令数 {len(program)} -> {len(after_dce)}；结果一致："
      f"{all(run(program, a, b) == run(after_dce, a, b) for a, b in [(1, 2), (3, 5), (-2, 7)])}")
print("\n这三步（建 IR、在 IR 上跑一串 pass、验证语义不变）就是所有编译器的骨架。")
print("SSA（每个变量只赋值一次）让'这两个表达式是不是同一个值'变成简单的比较，上面的 CSE 正是靠它。")
