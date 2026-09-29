class Detokenizer:
    def __init__(self, decode, eos_id):
        self.decode, self.eos_id = decode, eos_id
        self.ids = []

    def step(self, token, finished=False):
        if finished and token == self.eos_id:
            return ""
        self.ids.append(token)
        return self.decode([token])       # 单独解码新 token：会丢空格、会输出 �
