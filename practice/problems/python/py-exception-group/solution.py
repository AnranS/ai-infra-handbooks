def validate_batch(requests, validators):
    groups = []
    for i, r in enumerate(requests):
        errors = []
        for v in validators:
            try:
                v(r)
            except Exception as e:  # noqa: BLE001
                errors.append(e)
        if errors:
            groups.append(ExceptionGroup(f"request {i}", errors))
    if groups:
        raise ExceptionGroup(f"{len(groups)} 个请求校验失败", groups)


def _leaves(eg):
    if isinstance(eg, BaseExceptionGroup):
        return [leaf for e in eg.exceptions for leaf in _leaves(e)]
    return [eg]


def count_errors(fn):
    counts = {"value": 0, "type": 0}
    try:
        fn()
    except* ValueError as eg:
        counts["value"] += len(_leaves(eg))
    except* TypeError as eg:
        counts["type"] += len(_leaves(eg))
    return counts
