import math


class Vector:
    __slots__ = ("_x", "_y")

    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __repr__(self):
        return f"Vector({self._x!r}, {self._y!r})"

    def __eq__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return self._x == other._x and self._y == other._y

    def __hash__(self):
        return hash((self._x, self._y))

    def __abs__(self):
        return math.hypot(self._x, self._y)

    def __bool__(self):
        return bool(self._x or self._y)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._x + other._x, self._y + other._y)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self._x - other._x, self._y - other._y)

    def __mul__(self, k):
        if not isinstance(k, (int, float)):
            return NotImplemented
        return Vector(self._x * k, self._y * k)

    __rmul__ = __mul__

    def __neg__(self):
        return Vector(-self._x, -self._y)

    def __iter__(self):
        yield self._x
        yield self._y
