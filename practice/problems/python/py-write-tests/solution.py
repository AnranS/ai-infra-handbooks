from checker import raises


def test_plain_and_bytes(parse_size):
    assert parse_size("512") == 512
    assert parse_size("0B") == 0
    assert parse_size("7 b") == 7


def test_decimal_units(parse_size):
    assert parse_size("10KB") == 10_000
    assert parse_size("3MB") == 3_000_000
    assert parse_size("2GB") == 2_000_000_000


def test_binary_units(parse_size):
    assert parse_size("2KiB") == 2048
    assert parse_size("1MiB") == 1024 ** 2
    assert parse_size("1GiB") == 1024 ** 3


def test_case_and_spaces(parse_size):
    assert parse_size("10kb") == 10_000
    assert parse_size("4 gib") == 4 * 1024 ** 3
    assert parse_size("  8   MB ") == 8_000_000


def test_fractions(parse_size):
    assert parse_size("1.5 KiB") == 1536
    assert parse_size("2.5MB") == 2_500_000


def test_invalid(parse_size):
    for bad in ["", "abc", "-1", "-2KB", "0.5B", "1.3B", "10XB", "KB"]:
        with raises(ValueError, f"parse_size({bad!r})"):
            parse_size(bad)
