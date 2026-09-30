def const_fold(instrs):
    consts = {}
    out = []
    for dst, op, args in instrs:
        if op == "const":
            consts[dst] = args[0]
        elif op in ("+", "-", "*") and all(a in consts for a in args):
            x, y = consts[args[0]], consts[args[1]]
            consts[dst] = x + y if op == "+" else x - y if op == "-" else x * y
            continue                           # 折叠后没有生成 const 指令，后面就找不到这个值了
        out.append((dst, op, args))
    return out


def cse(instrs):
    seen, out = {}, []
    for dst, op, args in instrs:
        key = (op, args)                       # 没有对参数做改名替换
        if key in seen:
            continue
        seen[key] = dst
        out.append((dst, op, args))
    return out


def dce(instrs):
    live = {a for dst, op, args in instrs if op == "store" for a in args}
    return [i for i in instrs if i[0] in live or i[1] == "store"]   # 只看一层，间接用到的会被误删


def optimize(instrs):
    return dce(cse(const_fold(instrs)))
