"""detokenizer.py —— 增量反分词与停止字符串：流式输出时每次只吐出新的、完整的文本。"""


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop: list[str] = ()):
        self.tok, self.stop = tokenizer, list(stop)
        self.ids: list[int] = []
        self.prefix_offset = self.read_offset = 0
        self.text = ""              # 已经确定的完整输出
        self.num_sent = 0           # 已经发给客户端的字符数
        self.stopped = False
        # 可能是某个停止字符串开头的尾巴先扣住不发，确认不是停止字符串后再发
        self.holdback = max((len(s) for s in self.stop), default=1) - 1

    def update(self, token_id: int) -> str:
        """加入一个新 token，返回可以发送给客户端的新文本。"""
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(full) > len(prefix) and not full.endswith("�"):   # 末尾不是半个 UTF-8 字符
            self.text += full[len(prefix):]
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        for s in self.stop:
            pos = self.text.find(s)
            if pos != -1:
                self.text, self.stopped = self.text[:pos], True
        return self._emit(final=self.stopped)

    def flush(self) -> str:
        return self._emit(final=True)

    def _emit(self, final: bool) -> str:
        end = len(self.text) if final else max(self.num_sent, len(self.text) - self.holdback)
        delta, self.num_sent = self.text[self.num_sent:end], end
        return delta
