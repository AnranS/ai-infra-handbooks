def add_request(req, queue=None):
    if queue is None:
        queue = []
    queue.append(req)
    return queue


def remove_finished(reqs):
    reqs[:] = [r for r in reqs if not r["finished"]]


def make_grid(rows, cols):
    return [[0] * cols for _ in range(rows)]


def same_model(a, b):
    return a == b


def total_cost(prices):
    return round(sum(prices), 2)
