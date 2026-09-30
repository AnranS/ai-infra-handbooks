def const_fold(instrs):
    consts = {}
    out = []
    for dst, op, args in instrs:
        if op == "const":
            consts[dst] = args[0]
            out.append((dst, op, args))
        elif op in ("+", "-", "*") and all(a in consts for a in args):
            x, y = consts[args[0]], consts[args[1]]
            value = x + y if op == "+" else x - y if op == "-" else x * y
            consts[dst] = value
            out.append((dst, "const", (value,)))
        else:
            out.append((dst, op, args))
    return out


def cse(instrs):
    seen, rename, out = {}, {}, []
    for dst, op, args in instrs:
        args = tuple(rename.get(a, a) for a in args)
        key = (op, args)
        if op in ("+", "-", "*", "const") and key in seen:
            rename[dst] = seen[key]            # 复用第一次的结果
            continue
        if op != "store":
            seen[key] = dst
        out.append((dst, op, args))
    return out


def dce(instrs):
    live = set()
    for dst, op, args in instrs:
        if op == "store":
            live |= set(args)
    keep = []
    for dst, op, args in reversed(instrs):     # 倒着走：先看到用它的人
        if op == "store" or dst in live:
            live |= set(args)
            keep.append((dst, op, args))
    return keep[::-1]


def optimize(instrs):
    return dce(cse(const_fold(instrs)))
