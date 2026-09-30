def batch_waste(lengths):
    return max(lengths) - min(lengths)         # 只看首尾之差，没乘个数


def _batches(requests, batch_size):
    return [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]


def total_waste(requests, batch_size):
    return sum(batch_waste(b) for b in _batches(requests, batch_size))


def sorted_waste(requests, batch_size):
    return total_waste(sorted(requests, reverse=True), batch_size)


def waste_ratio(requests, batch_size, sort_first=False):
    waste = sorted_waste(requests, batch_size) if sort_first else total_waste(requests, batch_size)
    return waste / sum(requests)               # 分母用的是有效 token，不是总槽位
