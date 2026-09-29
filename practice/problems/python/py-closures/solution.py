import functools


def make_counter(start=0, step=1):
    value = start - step

    def counter():
        nonlocal value
        value += step
        return value

    return counter


def compose(*fns):
    def composed(x):
        for f in reversed(fns):
            x = f(x)
        return x

    return composed


def make_multipliers(n):
    return [lambda x, i=i: x * i for i in range(n)]
