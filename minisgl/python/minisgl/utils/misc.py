from __future__ import annotations


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """a // b，要求整除。allow_replicate=True 时允许 b > a（此时每份得到 1，即复制）。"""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """向上取整的除法。"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """把 a 向上对齐到 b 的倍数。"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """把 a 向下对齐到 b 的倍数。"""
    return (a // b) * b
