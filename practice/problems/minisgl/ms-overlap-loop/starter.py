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
    while running or last:
        cur = []
        for r in running:                        # 发射本轮
            r.complete_one()
            cur.append((r, sample(r.uid, r.n_sampled)))
            r.n_sampled += 1
        if last:                                 # 处理上一轮
            for r, tok in last:
                r.input_len += 1
                finished = tok == eos or not r.can_decode       # 官方写法：结束标记提前一个 token
                messages.append((r.uid, tok, finished))
                if finished and r in running:
                    running.remove(r)
        last = cur if cur else None
    return messages
