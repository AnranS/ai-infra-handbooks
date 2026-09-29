import operator

_OPS = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "div": operator.truediv}


def evaluate(expr, env=None):
    env = env or {}
    match expr:
        case bool():
            raise ValueError(f"无法识别的表达式：{expr!r}")
        case int() | float():
            return expr
        case str(name):
            if name not in env:
                raise NameError(f"未定义的变量：{name}")
            return env[name]
        case ("neg", e):
            return -evaluate(e, env)
        case ("add" | "sub" | "mul" | "div" as op, a, b):
            return _OPS[op](evaluate(a, env), evaluate(b, env))
        case ("max", first, *rest):
            return max(evaluate(e, env) for e in (first, *rest))
        case ("let", str(name), value, body):
            return evaluate(body, {**env, name: evaluate(value, env)})
        case ("if", cond, then, else_):
            return evaluate(then, env) if evaluate(cond, env) != 0 else evaluate(else_, env)
        case _:
            raise ValueError(f"无法识别的表达式：{expr!r}")
