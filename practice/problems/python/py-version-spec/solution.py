import operator
import re

_VERSION = re.compile(r"^(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?$")
_PRE = {"a": 0, "b": 1, "rc": 2}
_OPS = {"==": operator.eq, "!=": operator.ne, ">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}
_SPEC = re.compile(r"^(==|!=|>=|<=|~=|>|<)\s*(\S+)$")


def parse_version(s: str):
    m = _VERSION.match(s.strip())
    if not m:
        raise ValueError(f"无法解析的版本号：{s!r}")
    nums = [int(x) for x in m.group(1).split(".")]
    while len(nums) > 1 and nums[-1] == 0:
        nums.pop()
    pre = (_PRE[m.group(2)], int(m.group(3))) if m.group(2) else (3, 0)
    return (tuple(nums), pre)


def satisfies(version: str, spec: str) -> bool:
    v = parse_version(version)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = _SPEC.match(part)
        if not m:
            raise ValueError(f"无法解析的约束：{part!r}")
        op, target = m.groups()
        if op == "~=":
            segs = [int(x) for x in target.split(".") if x.isdigit()]
            if len(segs) < 2 or not re.fullmatch(r"\d+(\.\d+)+", target):
                raise ValueError(f"~= 至少需要两段版本号：{target!r}")
            upper = ".".join(map(str, segs[:-2] + [segs[-2] + 1]))
            if not (v >= parse_version(target) and v < parse_version(upper)):
                return False
        elif not _OPS[op](v, parse_version(target)):
            return False
    return True
