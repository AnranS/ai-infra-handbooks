def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs:
            if stack and stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    return True                                # 结束时没检查栈是否为空


def longest_valid(s):
    stack = []                                 # 没有初始基准 -1
    best = 0
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif stack:
            stack.pop()
            best = max(best, i - (stack[-1] if stack else 0) + 1)
    return best
