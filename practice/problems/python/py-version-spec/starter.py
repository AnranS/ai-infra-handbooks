import re


def parse_version(s: str):
    return tuple(int(x) for x in s.split("."))


def satisfies(version: str, spec: str) -> bool:
    pass
