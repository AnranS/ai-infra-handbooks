# 编译器前端的三步：词法分析（字符 -> 记号）、语法分析（记号 -> 语法树）、求值或生成代码
import re
from dataclasses import dataclass

TOKEN = re.compile(r"\s*(?:(\d+\.?\d*)|([A-Za-z_]\w*)|(\*\*|[-+*/()=,;]))")


def tokenize(src):
    tokens, pos = [], 0
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m:
            raise SyntaxError(f"无法识别的字符：{src[pos]!r}")
        pos = m.end()
        num, name, op = m.groups()
        tokens.append(("num", float(num)) if num else ("name", name) if name else ("op", op))
    tokens.append(("eof", None))
    return tokens


@dataclass
class Num:
    value: float


@dataclass
class Var:
    name: str


@dataclass
class BinOp:
    op: str
    left: object
    right: object


PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}     # 越大结合得越紧


class Parser:
    """Pratt 解析器：用优先级表处理二元运算符，比写一堆递归函数短得多"""

    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def peek(self):
        return self.tokens[self.i]

    def next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse(self, min_prec=0):
        kind, value = self.next()
        if kind == "num":
            node = Num(value)
        elif kind == "name":
            node = Var(value)
        elif (kind, value) == ("op", "("):
            node = self.parse()
            assert self.next() == ("op", ")"), "括号没有闭合"
        elif (kind, value) == ("op", "-"):
            node = BinOp("-", Num(0.0), self.parse(3))     # 一元负号
        else:
            raise SyntaxError(f"意外的记号：{value!r}")
        while True:
            kind, op = self.peek()
            if kind != "op" or op not in PRECEDENCE or PRECEDENCE[op] < min_prec:
                return node
            self.next()
            right = self.parse(PRECEDENCE[op] + (0 if op == "**" else 1))   # ** 右结合
            node = BinOp(op, node, right)


def show(node):
    if isinstance(node, Num):
        return f"{node.value:g}"
    if isinstance(node, Var):
        return node.name
    return f"({show(node.left)} {node.op} {show(node.right)})"


def fold(node):
    """常量折叠：两个操作数都是常量就直接算出来；还顺手做几个代数化简"""
    if not isinstance(node, BinOp):
        return node
    left, right = fold(node.left), fold(node.right)
    if isinstance(left, Num) and isinstance(right, Num):
        value = {"+": left.value + right.value, "-": left.value - right.value,
                 "*": left.value * right.value, "/": left.value / right.value,
                 "**": left.value ** right.value}[node.op]
        return Num(value)
    if node.op == "*" and isinstance(right, Num) and right.value == 1:
        return left                                        # x * 1 -> x
    if node.op == "*" and isinstance(right, Num) and right.value == 0:
        return Num(0.0)                                    # x * 0 -> 0
    if node.op == "+" and isinstance(right, Num) and right.value == 0:
        return left                                        # x + 0 -> x
    return BinOp(node.op, left, right)


for src in ["2 * 3 + x * 1", "a + 2 * 3 * 4", "x * (2 - 2)", "2 ** 3 ** 2", "-x + 1 - 1"]:
    tree = Parser(tokenize(src)).parse()
    print(f"{src:18s} 语法树 {show(tree):28s} 折叠后 {show(fold(tree))}")
