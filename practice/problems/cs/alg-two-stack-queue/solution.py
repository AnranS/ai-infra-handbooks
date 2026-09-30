class MyQueue:
    def __init__(self):
        self.in_stack, self.out_stack = [], []

    def push(self, x):
        self.in_stack.append(x)

    def _move(self):
        if not self.out_stack:                 # 只在空的时候倒，保证摊还 O(1)
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())

    def pop(self):
        self._move()
        return self.out_stack.pop() if self.out_stack else None

    def peek(self):
        self._move()
        return self.out_stack[-1] if self.out_stack else None

    def empty(self):
        return not self.in_stack and not self.out_stack
