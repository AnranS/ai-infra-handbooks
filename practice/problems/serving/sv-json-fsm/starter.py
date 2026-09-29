class IntArrayFSM:
    def start(self):
        return "start"

    def step(self, state, ch):
        return None

    def is_final(self, state):
        return state == "done"


def accepts(fsm, text):
    pass


def allowed_tokens(fsm, state, vocab, eos_id):
    pass


def constrained_greedy(fsm, vocab, eos_id, score_fn, max_steps):
    pass
