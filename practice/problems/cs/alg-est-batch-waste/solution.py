def batch_waste(lengths):
    if not lengths:
        return 0
    return max(lengths) * len(lengths) - sum(lengths)


def _batches(requests, batch_size):
    return [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]


def total_waste(requests, batch_size):
    return sum(batch_waste(b) for b in _batches(requests, batch_size))


def sorted_waste(requests, batch_size):
    return total_waste(sorted(requests), batch_size)


def waste_ratio(requests, batch_size, sort_first=False):
    waste = sorted_waste(requests, batch_size) if sort_first else total_waste(requests, batch_size)
    useful = sum(requests)
    return waste / (waste + useful) if waste + useful else 0.0
