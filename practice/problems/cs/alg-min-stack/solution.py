class MinStack:
    def __init__(self):
        self.data = []
        self.mins = []                         # mins[i] 是 data[:i+1] 的最小值

    def push(self, x):
        self.data.append(x)
        self.mins.append(x if not self.mins else min(x, self.mins[-1]))

    def pop(self):
        if not self.data:
            return None
        self.mins.pop()
        return self.data.pop()

    def top(self):
        return self.data[-1] if self.data else None

    def get_min(self):
        return self.mins[-1] if self.mins else None
