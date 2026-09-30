class MinStack:
    def __init__(self):
        self.data = []
        self.min = None                        # 只记一个最小值：弹栈后无法恢复

    def push(self, x):
        self.data.append(x)
        self.min = x if self.min is None else min(x, self.min)

    def pop(self):
        if not self.data:
            return None
        return self.data.pop()

    def top(self):
        return self.data[-1] if self.data else None

    def get_min(self):
        return self.min
