def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    return not stack                           # 结束时必须为空


def longest_valid(s):
    stack = [-1]                               # 基准：最后一个未匹配的右括号的下标
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)                # 这个右括号没人配对，成为新基准
            else:
                best = max(best, i - stack[-1])
    return best
