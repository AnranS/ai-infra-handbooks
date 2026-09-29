def prefix_hits(requests: list[list[int]]) -> list[int]:
    result = []
    for i, req in enumerate(requests):
        best = 0
        for prev in requests[:i]:
            n = 0
            while n < len(req) and n < len(prev) and req[n] == prev[n]:
                n += 1
            best = max(best, n)
        result.append(best)
    return result
