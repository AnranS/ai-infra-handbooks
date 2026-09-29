class IntArrayFSM:
    def start(self):
        return "start"

    def step(self, state, ch):
        digit = ch.isdigit() and ch.isascii()
        if state == "start":
            return "open" if ch == "[" else None
        if state == "open":
            if ch == "]":
                return "done"
            if ch == "-":
                return "neg"
            if ch == "0":
                return "zero"
            return "digits" if digit else None
        if state in ("neg", "space"):
            if ch == "-" and state == "space":
                return "neg"
            if ch == "0":
                return "zero"
            return "digits" if digit and ch != "0" else None
        if state in ("zero", "digits"):
            if ch == ",":
                return "comma"
            if ch == "]":
                return "done"
            return "digits" if digit and state == "digits" else None
        if state == "comma":
            return "space" if ch == " " else None
        return None

    def is_final(self, state):
        return state == "done"


def accepts(fsm, text):
    s = fsm.start()
    for ch in text:
        s = fsm.step(s, ch)
        if s is None:
            return False
    return fsm.is_final(s)


def allowed_tokens(fsm, state, vocab, eos_id):
    allowed = set()
    for i, tok in enumerate(vocab):
        if i == eos_id:
            if fsm.is_final(state):
                allowed.add(i)
            continue
        s = state
        for ch in tok:
            s = fsm.step(s, ch)
            if s is None:
                break
        if s is not None and tok:
            allowed.add(i)
    return allowed


def constrained_greedy(fsm, vocab, eos_id, score_fn, max_steps):
    text, state = "", fsm.start()
    for _ in range(max_steps):
        ok = allowed_tokens(fsm, state, vocab, eos_id)
        if not ok:
            break
        scores = score_fn(text)
        best = max(ok, key=lambda i: (scores[i], -i))
        if best == eos_id:
            break
        for ch in vocab[best]:
            state = fsm.step(state, ch)
        text += vocab[best]
    return text
