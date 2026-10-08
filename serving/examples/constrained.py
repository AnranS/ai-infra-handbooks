"""constrained.py —— 语法约束解码的最小实现：一个"JSON 模板"匹配器 + 词表前缀树，每步算出允许的 token 集合。

模板由字面量和带类型的字段组成，例如：
    ['{"city": "', STR, '", "temp_c": ', INT, ', "weather": "', ENUM("晴", "多云", "雨"), '"}']
匹配器的状态是 (第几段, 这一段已经读了什么)。一个 token 允许出现，当且仅当它的每个字符都能被匹配器接受。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    kind: str                      # "str" | "int" | "enum"
    options: tuple = ()
    max_len: int = 16


STR, INT = Field("str"), Field("int", max_len=3)


def ENUM(*options):
    return Field("enum", tuple(options))


class TemplateMatcher:
    def __init__(self, template):
        self.template = template

    def step(self, state, ch):
        """在状态 state 下读入字符 ch，返回新状态；不合法返回 None。"""
        seg, buf = state
        while seg < len(self.template):
            part = self.template[seg]
            if isinstance(part, str):                                  # 字面量：必须逐字符一致
                if part[len(buf)] == ch:
                    buf += ch
                    return (seg + 1, "") if len(buf) == len(part) else (seg, buf)
                return None
            if part.kind == "str" and ch not in '"\\\n' and len(buf) < part.max_len:
                return seg, buf + ch
            if part.kind == "int" and ch.isdigit() and len(buf) < part.max_len and not (buf == "0"):
                return seg, buf + ch
            if part.kind == "enum" and any(o.startswith(buf + ch) for o in part.options):
                return seg, buf + ch
            if self._field_complete(part, buf):                        # 字段可以在这里结束：交给下一段
                seg, buf = seg + 1, ""
                continue
            return None
        return None

    @staticmethod
    def _field_complete(field, buf):
        if field.kind == "enum":
            return buf in field.options
        return len(buf) > 0

    def is_final(self, state):
        seg, buf = state
        return seg == len(self.template)

    def forced_text(self, state):
        """当前状态下，接下来"唯一可能"的字面量（可以直接写出，不必让模型生成）。"""
        seg, buf = state
        if seg < len(self.template) and isinstance(self.template[seg], str):
            return self.template[seg][len(buf):]
        return ""


class VocabTrie:
    """把词表中每个 token 解码后的字符串插入一棵字符前缀树，节点上记录"在这里结束的 token"。"""

    def __init__(self, tokenizer, special_ids=()):
        self.root = {}
        for tid in range(len(tokenizer)):
            if tid in special_ids:
                continue
            s = tokenizer.decode([tid])
            if not s or "�" in s:                                 # 不完整的 UTF-8 片段：本实现不允许
                continue
            node = self.root
            for ch in s:
                node = node.setdefault(ch, {})
            node.setdefault(None, []).append(tid)

    def allowed(self, matcher: TemplateMatcher, state):
        """与匹配器同步遍历前缀树：只沿合法的字符往下走，返回 (允许的 token 列表, 访问的节点数)。"""
        result, visited, stack = [], 0, [(self.root, state)]
        while stack:
            node, st = stack.pop()
            visited += 1
            for ch, child in node.items():
                if ch is None:
                    result.extend(child)
                    continue
                nxt = matcher.step(st, ch)
                if nxt is not None:
                    stack.append((child, nxt))
        return result, visited


def advance(matcher, state, text):
    for ch in text:
        state = matcher.step(state, ch)
    return state
