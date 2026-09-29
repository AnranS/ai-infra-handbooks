class Detokenizer:
    def __init__(self, decode, eos_id):
        self.decode, self.eos_id = decode, eos_id
        self.ids = []
        self.surr = 0
        self.read = 0

    def step(self, token, finished=False):
        if not (finished and token == self.eos_id):
            self.ids.append(token)
        read_str = self.decode(self.ids[self.surr:])
        surr_str = self.decode(self.ids[self.surr:self.read])
        new = read_str[len(surr_str):]
        if new and not new.endswith("�"):
            self.surr, self.read = self.read, len(self.ids)
            return new
        return ""
