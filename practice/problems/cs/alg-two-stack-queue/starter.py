class MyQueue:
    def __init__(self):
        self.in_stack, self.out_stack = [], []

    def push(self, x):
        while self.out_stack:                  # 每次 push 都倒来倒去：退化成 O(n)
            self.in_stack.append(self.out_stack.pop())
        self.in_stack.append(x)

    def pop(self):
        while self.in_stack:
            self.out_stack.append(self.in_stack.pop())
        return self.out_stack.pop() if self.out_stack else None

    def peek(self):
        return self.out_stack[-1] if self.out_stack else None   # 没有先倒：可能拿到 None

    def empty(self):
        return not self.in_stack and not self.out_stack
