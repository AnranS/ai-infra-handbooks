class StopChecker:
    def __init__(self, stops, include_stop=False):
        self.stops = list(stops)
        self.include_stop = include_stop
        self.done = False

    def feed(self, delta):
        for s in self.stops:                 # 只在当前这一段里找：跨段的停止字符串会漏掉
            i = delta.find(s)
            if i >= 0:
                self.done = True
                return delta[:i], True
        return delta, False

    def flush(self):
        return ""
