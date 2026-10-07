# Structural pattern matching

<p class="lead">The <code>match</code> statement <span class="since">3.10+</span> is more than a switch-case. It does "test the structure" and "take the data out" at once, which is particularly clear for nested JSON, commands and syntax trees.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Does `case [x, y]:` match the tuple `(1, 2)`? Does it match the string `"ab"`?
    2. Does `case {"type": "click"}:` match `{"type": "click", "x": 1}`?
    3. Why is `case RED:` a bug?
    4. How does the class pattern `case Point(x=0, y=y):` work?
    5. What is a guard?

??? success "Answers (try it yourself first, then expand)"
    1. It matches the tuple `(1, 2)`: a sequence pattern applies to both lists and tuples; it does not match the string `"ab"`: strings and bytes are deliberately excluded from sequence patterns.
    2. Yes: a mapping pattern only requires the listed keys to be present with matching values, and extra keys do not matter.
    3. A bare name is a capture pattern: `case RED:` matches any value and binds it to a variable called `RED` rather than comparing it against a constant, so the branches after it can never be reached. A constant needs a dotted name, `case Color.RED:` say.
    4. It does an `isinstance(subject, Point)` first and then checks the keywords one by one: the attribute `x` has to equal 0 and the attribute `y`'s value is captured into the variable `y`.
    5. The `if condition` after a `case` pattern: once the pattern matches, the condition is checked, and failing it counts as not matching, so the next branch is tried.

## The basic form {#基本形式}

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 401 | 403:                  # an or pattern
            return "Not allowed"
        case 404:
            return "Not found"
        case _:                          # the wildcard: matches anything, placed last
            return "Something else"

assert http_error(403) == "Not allowed"
assert http_error(418) == "Something else"
```

The branches are tried top to bottom, the first that matches runs and that is the end, with no "fall-through" to the next. When nothing matches, nothing happens (and nothing is raised).

## The kinds of pattern {#模式的种类}

### Capture and wildcard {#捕获与通配}

A bare name is a **capture pattern**: it always matches and binds the value to that name. `_` is the special wildcard, which matches without binding.

### Sequence patterns {#序列模式}

```python
def describe(point):
    match point:
        case (0, 0):
            return "origin"
        case (0, y):
            return f"on y axis at {y}"
        case (x, 0):
            return f"on x axis at {x}"
        case (x, y):
            return f"at ({x}, {y})"
        case [x, y, *rest]:
            return f"{len(rest) + 2}D point"
        case _:
            return "not a point"

assert describe((0, 5)) == "on y axis at 5"
assert describe([3, 4]) == "at (3, 4)"          # both lists and tuples match a sequence pattern
assert describe((1, 2, 3, 4)) == "4D point"
assert describe("ab") == "not a point"          # a string is not a sequence pattern
```

Writing a sequence pattern's brackets as `()` or `[]` makes no difference and both match any sequence (except `str`, `bytes` and `bytearray`). `*rest` collects the remaining elements.

### Mapping patterns {#映射模式}

```python
def handle(event):
    match event:
        case {"type": "click", "pos": (x, y)}:
            return f"click at {x},{y}"
        case {"type": "key", "key": str(k)} if len(k) == 1:
            return f"typed {k!r}"
        case {"type": "key", "key": k}:
            return f"special key {k}"
        case {"type": t, **rest}:
            return f"unknown {t} with {sorted(rest)}"

assert handle({"type": "click", "pos": (3, 4), "button": 1}) == "click at 3,4"
assert handle({"type": "key", "key": "a"}) == "typed 'a'"
assert handle({"type": "key", "key": "Enter"}) == "special key Enter"
assert handle({"type": "scroll", "dy": -1}) == "unknown scroll with ['dy']"
```

A mapping pattern only requires **the listed keys to be present** and ignores the extras (unlike a sequence pattern, which requires the length to match exactly). `**rest` collects the remaining key-value pairs.

### Class patterns and guards {#类模式与守卫}

Something like `str(k)` is a **class pattern**: it checks `isinstance(value, str)` first and then binds the value to `k`. The example above also uses a **guard**, `if len(k) == 1`: once the pattern matches, the condition is checked, and failing it moves on to the next branch.

Class patterns work very well with dataclasses:

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    r: float

@dataclass
class Rect:
    corner: Point
    w: float
    h: float

def area(shape):
    match shape:
        case Circle(r=r):
            return 3.14159 * r * r
        case Rect(w=w, h=h) if w == h:
            return f"square of {w * h}"
        case Rect(Point(0, 0), w, h):          # matched by position, relying on __match_args__
            return f"rect at origin, area {w * h}"
        case Rect(w=w, h=h):
            return w * h

assert area(Circle(Point(0, 0), 1)) == 3.14159
assert area(Rect(Point(1, 1), 2, 2)) == "square of 4"
assert area(Rect(Point(0, 0), 2, 3)) == "rect at origin, area 6"
assert area(Rect(Point(1, 1), 2, 3)) == 6
```

A positional class pattern like `Rect(Point(0, 0), w, h)` relies on the class's `__match_args__` attribute, which a dataclass generates automatically (in field order). A class of your own can define `__match_args__ = ("x", "y")` by hand.

### Binding with `as` {#as-绑定}

Binding the whole to a name while matching a subpattern:

```pycon
>>> match {"cmd": "move", "to": [1, 2]}:
...     case {"cmd": "move", "to": [_, _] as target}:
...         print("moving to", target)
...
moving to [1, 2]
```

## The commonest bug: writing a constant as a bare name {#最常见的-bug把常量写成裸名字}

```pycon
>>> RED = "red"
>>> color = "blue"
>>> match color:
...     case RED:                    # this is not "equals RED" but "captured into the name RED"
...         print("it's red?!")
...
it's red?!
>>> RED
'blue'
```

A bare name is always a capture pattern. Not only does it match any value, it **overwrites the original variable**. Fortunately, when branches follow it, Python raises `SyntaxError: name capture 'RED' makes remaining patterns unreachable` outright.

To compare against a constant, a **dotted name** is required, an enum member say:

```python
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"

def paint(c):
    match c:
        case Color.RED:              # dotted: a value pattern, compared with ==
            return "stop"
        case Color.GREEN:
            return "go"

assert paint(Color.GREEN) == "go"
```

## In practice: parsing a command {#实战命令解析}

`match` suits "do something different by the shape of the input" best:

```python
def run(command: str):
    match command.split():
        case ["quit" | "exit"]:
            return "bye"
        case ["go", ("north" | "south" | "east" | "west") as direction]:
            return f"going {direction}"
        case ["get", item]:
            return f"picked up {item}"
        case ["drop", *items] if items:
            return f"dropped {', '.join(items)}"
        case []:
            return "say something"
        case _:
            return f"unknown command: {command!r}"

assert run("go north") == "going north"
assert run("go up") == "unknown command: 'go up'"
assert run("drop sword shield") == "dropped sword, shield"
assert run("") == "say something"
```

The same logic in `if/elif` needs a great deal of `len(parts) == 2 and parts[0] == "go" and parts[1] in {...}`, nothing like as clear.

## When not to use `match` {#什么时候不用-match}

- Merely comparing one value against a few constants where each branch returns a value: a dict lookup is simpler, `return MESSAGES.get(status, "unknown")`.
- The branches dispatch by type and other modules should be able to extend them: consider `functools.singledispatch` or a polymorphic method.
- The conditions are ranges (`x > 10`): `if/elif` says it more directly.

!!! interview "Answering in an interview"
    On pattern matching: `match` finds the first matching branch top to bottom and does the structural test and the extraction at once; a sequence pattern requires the length to match (with `*rest` available) and does not match strings; a mapping pattern only requires the listed keys and ignores the extras; a class pattern does the `isinstance` first and then matches the attributes, with the positional form relying on `__match_args__`. The commonest bug: the bare name in `case RED:` captures and binds rather than compares, so it matches anything, and a constant has to be a dotted name (`Color.RED`); conditions a pattern cannot express go in an `if` guard. It suits parsing commands, protocol messages and other structured input.

## Exercises {#练习}

**1. An expression evaluator.** Represent arithmetic expressions as nested tuples: a number is itself, and `("+", a, b)`, `("-", a, b)`, `("*", a, b)`, `("/", a, b)` and `("neg", a)`. Write `evaluate(expr)` raising `ZeroDivisionError` on division by zero and `ValueError` on a structure it does not recognize.

??? success "Answer"
    ```python
    def evaluate(expr):
        match expr:
            case int() | float():
                return expr
            case ("neg", a):
                return -evaluate(a)
            case ("+", a, b):
                return evaluate(a) + evaluate(b)
            case ("-", a, b):
                return evaluate(a) - evaluate(b)
            case ("*", a, b):
                return evaluate(a) * evaluate(b)
            case ("/", a, b):
                divisor = evaluate(b)
                if divisor == 0:
                    raise ZeroDivisionError(f"division by zero in {expr!r}")
                return evaluate(a) / divisor
            case _:
                raise ValueError(f"bad expression: {expr!r}")

    assert evaluate(("+", 1, ("*", 2, 3))) == 7
    assert evaluate(("neg", ("-", 10, 4))) == -6
    assert evaluate(("/", 1, 4)) == 0.25
    try:
        evaluate(("^", 2, 3))
    except ValueError as e:
        assert "bad expression" in str(e)
    ```

    A class pattern with no arguments like `int()` only does the `isinstance` check. Note that `bool` is a subclass of `int`, so `True` counts as a number too and a strict implementation excludes it separately.

**2. Routing webhook events.** Handle event dicts of the shapes below and return one line of description:

- `{"event": "push", "repo": {"name": n}, "commits": [...]}` → `"n: 3 commits"` (and `"n: empty push"` with no commits)
- `{"event": "pull_request", "action": "opened" or "reopened", "number": k}` → `"PR #k opened"`
- `{"event": "pull_request", "action": "closed", "merged": True, "number": k}` → `"PR #k merged"`
- anything else → `"ignored"`

??? success "Answer"
    ```python
    def route(evt):
        match evt:
            case {"event": "push", "repo": {"name": name}, "commits": []}:
                return f"{name}: empty push"
            case {"event": "push", "repo": {"name": name}, "commits": [*commits]}:
                return f"{name}: {len(commits)} commits"
            case {"event": "pull_request", "action": "opened" | "reopened", "number": int(n)}:
                return f"PR #{n} opened"
            case {"event": "pull_request", "action": "closed", "merged": True, "number": int(n)}:
                return f"PR #{n} merged"
            case _:
                return "ignored"

    assert route({"event": "push", "repo": {"name": "api"}, "commits": [1, 2, 3]}) == "api: 3 commits"
    assert route({"event": "push", "repo": {"name": "api"}, "commits": []}) == "api: empty push"
    assert route({"event": "pull_request", "action": "reopened", "number": 7}) == "PR #7 opened"
    assert route({"event": "pull_request", "action": "closed", "merged": True, "number": 7}) == "PR #7 merged"
    assert route({"event": "pull_request", "action": "closed", "merged": False, "number": 7}) == "ignored"
    ```

## Summary {#小结}

- [x] `match` does the structural test and the extraction at once, and the first matching branch runs.
- [x] A sequence pattern requires the length to match (with `*rest` available); a mapping pattern only requires the listed keys.
- [x] A class pattern `Cls(attr=pattern)` does the `isinstance` first and then matches the attributes; the positional form relies on `__match_args__`.
- [x] A bare name captures rather than compares; a constant needs a dotted name, an enum member say.
- [x] A guard `if ...` adds the conditions a pattern cannot express.
