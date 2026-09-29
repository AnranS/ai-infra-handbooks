from checker import check, raises
from solution import parse_version, satisfies


def test_example():
    check(satisfies("2.4.1", ">=2.4, <3"), True, 'satisfies("2.4.1", ">=2.4, <3")')
    check(satisfies("1.27.0", "~=1.26.2"), False, 'satisfies("1.27.0", "~=1.26.2")')
    check(satisfies("2.5.0rc1", ">=2.5"), False, 'satisfies("2.5.0rc1", ">=2.5")')


def test_parse_and_compare():
    assert parse_version("2.4") == parse_version("2.4.0"), "2.4 应该等于 2.4.0"
    order = ["1.0a1", "1.0b2", "1.0rc1", "1.0", "1.0.1", "1.1", "2.0.0rc3", "2", "10.0"]
    keys = [parse_version(v) for v in order]
    for a, b, ka, kb in zip(order, order[1:], keys, keys[1:]):
        assert ka < kb, f"{a} 应该小于 {b}"
    for bad in ["", "1.", ".1", "1..2", "v1.0", "1.0-rc1", "1.0rc", "abc"]:
        with raises(ValueError, f"parse_version({bad!r})"):
            parse_version(bad)


def test_operators():
    cases = [("2.4.0", "==2.4", True), ("2.4.1", "==2.4", False), ("2.4.1", "!=2.4", True), ("3.0", "<3", False),
             ("3.0", "<=3", True), ("3.0.1", ">3", True), ("2.9.9", ">3", False), ("1.0", "", True),
             ("1.0", " >= 0.9 , != 1.0 ", False)]
    for v, spec, want in cases:
        check(satisfies(v, spec), want, f"satisfies({v!r}, {spec!r})")


def test_compatible_release():
    cases = [("1.26.0", "~=1.26", True), ("1.99", "~=1.26", True), ("2.0", "~=1.26", False), ("1.25.9", "~=1.26", False),
             ("1.26.2", "~=1.26.2", True), ("1.26.9", "~=1.26.2", True), ("1.27", "~=1.26.2", False),
             ("1.26.1", "~=1.26.2", False), ("2.0.0rc1", "~=1.26", True)]
    for v, spec, want in cases:
        check(satisfies(v, spec), want, f"satisfies({v!r}, {spec!r})")
    with raises(ValueError, "~=1 只有一段"):
        satisfies("1.0", "~=1")


def test_bad_spec():
    with raises(ValueError, "未知运算符"):
        satisfies("1.0", "=>1.0")
