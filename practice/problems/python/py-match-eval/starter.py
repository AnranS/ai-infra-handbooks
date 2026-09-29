def evaluate(expr, env=None):
    env = env or {}
    match expr:
        case int() | float():
            return expr
        case _:
            raise ValueError(f"无法识别的表达式：{expr!r}")
