class Field:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.name]
        except KeyError:
            raise AttributeError(f"字段 {self.name} 还没有赋值") from None

    def __set__(self, instance, value):
        instance.__dict__[self.name] = self.validate(value)

    def validate(self, value):
        return value


class Typed(Field):
    def __init__(self, tp):
        self.tp = tp

    def validate(self, value):
        if isinstance(value, bool):
            raise TypeError(f"{self.name} 必须是 {self.tp.__name__}，不接受 bool")
        if self.tp is float and isinstance(value, int):
            value = float(value)
        if not isinstance(value, self.tp):
            raise TypeError(f"{self.name} 必须是 {self.tp.__name__}，收到 {type(value).__name__}")
        return super().validate(value)


class Positive(Typed):
    def validate(self, value):
        value = super().validate(value)
        if value <= 0:
            raise ValueError(f"{self.name} 必须是正数，收到 {value!r}")
        return value


class Bounded(Typed):
    def __init__(self, tp, lo, hi):
        super().__init__(tp)
        self.lo, self.hi = lo, hi

    def validate(self, value):
        value = super().validate(value)
        if not self.lo <= value <= self.hi:
            raise ValueError(f"{self.name} 必须在 [{self.lo}, {self.hi}] 内，收到 {value!r}")
        return value


class OneOf(Field):
    def __init__(self, *choices):
        self.choices = choices

    def validate(self, value):
        if value not in self.choices:
            raise ValueError(f"{self.name} 必须是 {list(self.choices)} 之一，收到 {value!r}")
        return value
