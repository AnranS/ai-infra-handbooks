import types


class Upstream:
    """假装是 SGLang 的 Scheduler：get_next_batch_to_run 内部会调 self.get_new_batch_prefill()。"""

    def __init__(self):
        raise RuntimeError("上游的 __init__ 会连 ZMQ、起一堆组件——我们不想执行它")

    def get_next_batch_to_run(self):
        return f"plan({self.get_new_batch_prefill()}, running={self.running})"

    def get_new_batch_prefill(self):
        return "upstream-prefill"

    def log_stats(self):
        return f"stats of {type(self).__name__}"


class Composed:
    """不继承 Upstream，用 __getattr__ 借它的方法。"""

    def __init__(self):
        self.running = 3                    # 状态全在自己身上

    def get_new_batch_prefill(self):        # 想改的方法直接定义在自己身上
        return "omni-prefill"

    def __getattr__(self, name):
        attr = getattr(Upstream, name)
        return types.MethodType(attr, self) if callable(attr) else attr


c = Composed()
print(c.get_next_batch_to_run())   # 上游的方法，内部调到的是我们覆盖的版本
print(c.log_stats())               # 没覆盖的方法原样借用
print("isinstance:", isinstance(c, Upstream))
