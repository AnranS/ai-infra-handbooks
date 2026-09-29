import types
import typing
from typing import Any, Literal, Union, get_args, get_origin


def validate(value, tp, path="$"):
    if isinstance(value, tp):
        return True
    raise TypeError(f"{path}: 期望 {tp}，实际是 {type(value).__name__}")
