class Backend:
    _registry: dict = {}

    def __init_subclass__(cls, name=None, priority=0, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is None:
            return
        if name in Backend._registry:
            raise ValueError(f"后端 {name!r} 已经注册过了")
        cls.name = name
        cls.priority = priority
        Backend._registry[name] = cls

    def forward(self, x):
        raise NotImplementedError

    @classmethod
    def create(cls, name, *args, **kwargs):
        try:
            impl = Backend._registry[name]
        except KeyError:
            raise KeyError(f"没有名为 {name!r} 的后端，可用的有：{', '.join(Backend.available())}") from None
        return impl(*args, **kwargs)

    @staticmethod
    def available():
        return sorted(Backend._registry, key=lambda n: (-Backend._registry[n].priority, n))

    @staticmethod
    def clear_registry():
        Backend._registry.clear()
