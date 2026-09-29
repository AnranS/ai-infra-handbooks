def add_request(req, queue=[]):
    queue.append(req)
    return queue


def remove_finished(reqs):
    for r in reqs:
        if r["finished"]:
            reqs.remove(r)


def make_grid(rows, cols):
    return [[0] * cols] * rows


def same_model(a, b):
    return a is b


def total_cost(prices):
    return sum(prices)
