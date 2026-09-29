class Field:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        pass

    def __set__(self, instance, value):
        pass


class Typed(Field):
    def __init__(self, tp):
        self.tp = tp


class Positive(Typed):
    pass


class Bounded(Typed):
    def __init__(self, tp, lo, hi):
        super().__init__(tp)
        self.lo, self.hi = lo, hi


class OneOf(Field):
    def __init__(self, *choices):
        self.choices = choices
