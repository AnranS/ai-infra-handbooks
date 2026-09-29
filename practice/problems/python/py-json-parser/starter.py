class Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def error(self, msg):
        raise ValueError(f"第 {self.i} 个字符：{msg}")


def loads(text: str):
    pass
