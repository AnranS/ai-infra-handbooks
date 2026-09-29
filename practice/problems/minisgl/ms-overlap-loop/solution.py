class Req:
    def __init__(self, uid, prompt_len, max_tokens):
        self.uid = uid
        self.device_len = prompt_len
        self.max_device_len = prompt_len + max_tokens
        self.input_len = prompt_len
        self.n_sampled = 0

    @property
    def can_decode(self):
        return self.max_device_len - self.device_len > 0

    def complete_one(self):
        self.device_len += 1


def overlap_loop(reqs, sample, eos):
    running = list(reqs)
    messages = []
    last = None                                  # 上一轮的 [(req, token), ...]
    finished_prev = set()                        # 上一次处理时结束的请求：它们在本轮的结果是过期的
    while running or last:
        cur = []
        for r in running:                        # 发射本轮
            r.complete_one()
            cur.append((r, sample(r.uid, r.n_sampled)))
            r.n_sampled += 1
        if last:                                 # 处理上一轮
            newly = set()
            for r, tok in last:
                if r.uid in finished_prev:           # 过期结果：请求已经结束了
                    continue
                r.input_len += 1
                finished = tok == eos or r.input_len >= r.max_device_len
                messages.append((r.uid, tok, finished))
                if finished:
                    newly.add(r.uid)
                    if r in running:
                        running.remove(r)
            finished_prev = newly
        last = cur if cur else None
    return messages
