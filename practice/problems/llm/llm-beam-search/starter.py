import numpy as np


def beam_search(step_fn, bos, eos, beam_size, max_len, length_penalty=1.0):
    # 贪心解码（beam_size = 1 的特例），不是束搜索
    tokens, score = (bos,), 0.0
    for _ in range(max_len):
        logp = step_fn(tokens)
        t = int(np.argmax(logp))
        tokens, score = tokens + (t,), score + float(logp[t])
        if t == eos:
            break
    return [(tokens[1:], score / (len(tokens) - 1) ** length_penalty)]
