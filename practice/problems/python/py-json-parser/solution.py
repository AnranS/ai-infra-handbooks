import re

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_WS = " \t\n\r"


class Parser:
    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def error(self, msg):
        raise ValueError(f"第 {self.i} 个字符：{msg}")

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in _WS:
            self.i += 1

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def expect(self, ch):
        if self.peek() != ch:
            self.error(f"期望 {ch!r}")
        self.i += 1

    def value(self):
        self.ws()
        c = self.peek()
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        if c == '"':
            return self.string()
        for word, val in (("true", True), ("false", False), ("null", None)):
            if self.s.startswith(word, self.i):
                self.i += len(word)
                return val
        if c == "-" or c.isdigit():
            return self.number()
        self.error("期望一个 JSON 值" if c else "文本意外结束")

    def number(self):
        m = _NUMBER.match(self.s, self.i)
        if not m:
            self.error("数字格式不对")
        self.i = m.end()
        if self.peek().isdigit():
            self.error("数字不能有前导 0")
        text = m.group(0)
        return float(text) if m.group(1) or m.group(2) else int(text)

    def hex4(self):
        h = self.s[self.i:self.i + 4]
        if len(h) != 4 or not all(ch in "0123456789abcdefABCDEF" for ch in h):
            self.error("\\u 后面需要 4 位十六进制数")
        self.i += 4
        return int(h, 16)

    def string(self):
        self.expect('"')
        out = []
        while True:
            if self.i >= len(self.s):
                self.error("字符串没有结束")
            c = self.s[self.i]
            if c == '"':
                self.i += 1
                return "".join(out)
            if c == "\\":
                self.i += 1
                e = self.peek()
                if e in _ESCAPES:
                    out.append(_ESCAPES[e])
                    self.i += 1
                elif e == "u":
                    self.i += 1
                    code = self.hex4()
                    if 0xD800 <= code <= 0xDBFF and self.s.startswith("\\u", self.i):
                        save = self.i
                        self.i += 2
                        low = self.hex4()
                        if 0xDC00 <= low <= 0xDFFF:
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        else:
                            self.i = save
                    out.append(chr(code))
                else:
                    self.error("无效的转义")
            elif ord(c) < 0x20:
                self.error("字符串里有未转义的控制字符")
            else:
                out.append(c)
                self.i += 1

    def arr(self):
        self.expect("[")
        result = []
        self.ws()
        if self.peek() == "]":
            self.i += 1
            return result
        while True:
            result.append(self.value())
            self.ws()
            if self.peek() == ",":
                self.i += 1
                continue
            self.expect("]")
            return result

    def obj(self):
        self.expect("{")
        result = {}
        self.ws()
        if self.peek() == "}":
            self.i += 1
            return result
        while True:
            self.ws()
            if self.peek() != '"':
                self.error("对象的键必须是字符串")
            key = self.string()
            self.ws()
            self.expect(":")
            result[key] = self.value()
            self.ws()
            if self.peek() == ",":
                self.i += 1
                continue
            self.expect("}")
            return result


def loads(text: str):
    p = Parser(text)
    v = p.value()
    p.ws()
    if p.i != len(text):
        p.error("多余的字符")
    return v
