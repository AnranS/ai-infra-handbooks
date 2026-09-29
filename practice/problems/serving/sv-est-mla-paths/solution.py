def _costs(s):
    expand = 2 * s["dc"] * s["H"] * (s["nope"] + s["dv"])
    pair_expand = 2 * s["H"] * (s["nope"] + s["rope"]) + 2 * s["H"] * s["dv"]
    absorb = 2 * s["H"] * s["nope"] * s["dc"] + 2 * s["H"] * s["dc"] * s["dv"]
    pair_absorb = 2 * s["H"] * (s["dc"] + s["rope"]) + 2 * s["H"] * s["dc"]
    return expand, pair_expand, absorb, pair_absorb


def path_flops(q, k, shape):
    expand, pair_expand, absorb, pair_absorb = _costs(shape)
    pairs = q * k - q * (q - 1) / 2
    return k * expand + pairs * pair_expand, q * absorb + pairs * pair_absorb


def choose(q, k, shape):
    e, a = path_flops(q, k, shape)
    return "expand" if e < a else "absorb"


def crossover_q(shape):
    expand, pair_expand, _, pair_absorb = _costs(shape)
    return expand / (pair_absorb - pair_expand)
