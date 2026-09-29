def validate_batch(requests, validators):
    for r in requests:
        for v in validators:
            v(r)          # 遇到第一个错误就停了


def count_errors(fn):
    counts = {"value": 0, "type": 0}
    try:
        fn()
    except ValueError:
        counts["value"] += 1
    except TypeError:
        counts["type"] += 1
    return counts
