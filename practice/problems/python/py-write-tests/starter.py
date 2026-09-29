from checker import raises


def test_plain_bytes(parse_size):
    assert parse_size("512") == 512
