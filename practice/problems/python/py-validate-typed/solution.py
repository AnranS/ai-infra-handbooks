import types
import typing
from typing import Any, Literal, Union, get_args, get_origin


def _name(tp):
    return getattr(tp, "__name__", None) or str(tp)


def _fail(path, tp, value):
    raise TypeError(f"{path}: 期望 {_name(tp)}，实际是 {type(value).__name__}")


def validate(value, tp, path="$"):
    if tp is Any:
        return True
    if tp is None or tp is type(None):
        if value is not None:
            _fail(path, "None", value)
        return True
    if typing.is_typeddict(tp):
        if not isinstance(value, dict):
            _fail(path, tp, value)
        hints = typing.get_type_hints(tp)
        for key in value:
            if key not in hints:
                raise TypeError(f"{path}: 多余的键 {key!r}")
        for key, sub in hints.items():
            if key not in value:
                raise TypeError(f"{path}: 缺少键 {key!r}")
            validate(value[key], sub, f'{path}["{key}"]')
        return True
    origin, args = get_origin(tp), get_args(tp)
    if origin in (Union, types.UnionType):
        errors = []
        for option in args:
            try:
                return validate(value, option, path)
            except TypeError as e:
                errors.append(str(e))
        raise TypeError(f"{path}: 不符合 {tp} 的任何一种：" + "；".join(errors))
    if origin is Literal:
        if not any(value == a and type(value) is type(a) for a in args):
            raise TypeError(f"{path}: 期望 {tp}，实际是 {value!r}")
        return True
    if origin in (list, set, frozenset):
        if not isinstance(value, origin):
            _fail(path, origin, value)
        (item,) = args or (Any,)
        for i, v in enumerate(value):
            validate(v, item, f"{path}[{i}]")
        return True
    if origin is tuple:
        if not isinstance(value, tuple):
            _fail(path, tuple, value)
        if len(args) == 2 and args[1] is Ellipsis:
            for i, v in enumerate(value):
                validate(v, args[0], f"{path}[{i}]")
        else:
            if len(value) != len(args):
                raise TypeError(f"{path}: 期望长度 {len(args)} 的元组，实际长度 {len(value)}")
            for i, (v, sub) in enumerate(zip(value, args)):
                validate(v, sub, f"{path}[{i}]")
        return True
    if origin is dict:
        if not isinstance(value, dict):
            _fail(path, dict, value)
        kt, vt = args or (Any, Any)
        for k, v in value.items():
            validate(k, kt, f"{path}.key({k!r})")
            validate(v, vt, f'{path}["{k}"]')
        return True
    if isinstance(tp, type):
        if isinstance(value, bool) and tp is not bool:
            _fail(path, tp, value)
        if tp is float and isinstance(value, int) and not isinstance(value, bool):
            return True
        if not isinstance(value, tp):
            _fail(path, tp, value)
        return True
    raise TypeError(f"{path}: 不支持的类型标注 {tp!r}")
