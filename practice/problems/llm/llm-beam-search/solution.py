import numpy as np


def beam_search(step_fn, bos, eos, beam_size, max_len, length_penalty=1.0):
    alive = [((bos,), 0.0)]
    finished = []
    for _ in range(max_len):
        cands = []
        for tokens, score in alive:
            logp = step_fn(tokens)
            for t, lp in enumerate(logp):
                cands.append((tokens + (t,), score + float(lp)))
        cands.sort(key=lambda c: (-c[1], c[0]))
        alive = []
        for tokens, score in cands:
            if tokens[-1] == eos:
                if len(finished) < beam_size:
                    finished.append((tokens, score))
            else:
                alive.append((tokens, score))
                if len(alive) == beam_size:
                    break
        if len(finished) >= beam_size or not alive:
            break
    finished += alive
    ranked = [(t[1:], s / (len(t) - 1) ** length_penalty) for t, s in finished]
    ranked.sort(key=lambda c: (-c[1], c[0]))
    return ranked
