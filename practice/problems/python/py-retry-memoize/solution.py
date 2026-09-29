import functools
from collections import OrderedDict


def retry(times, exceptions=(Exception,), on_retry=None):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == times:
                        raise
                    if on_retry is not None:
                        on_retry(attempt, e)

        return wrapper

    return decorator


def memoize(maxsize=None):
    def decorator(fn):
        cache = OrderedDict()
        stats = {"hits": 0, "misses": 0}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                stats["hits"] += 1
                cache.move_to_end(key)
                return cache[key]
            stats["misses"] += 1
            value = fn(*args, **kwargs)
            cache[key] = value
            if maxsize is not None and len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        def cache_info():
            return (stats["hits"], stats["misses"], len(cache))

        def cache_clear():
            cache.clear()
            stats["hits"] = stats["misses"] = 0

        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        return wrapper

    return decorator
