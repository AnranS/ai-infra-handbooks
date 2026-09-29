# 结构化模式匹配

<p class="lead"><code>match</code> 语句 <span class="since">3.10+</span> 不只是 switch-case。它能同时做"判断结构"和"取出数据"两件事，处理嵌套的 JSON、命令、语法树时特别清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `case [x, y]:` 能匹配元组 `(1, 2)` 吗？能匹配字符串 `"ab"` 吗？
    2. `case {"type": "click"}:` 能匹配 `{"type": "click", "x": 1}` 吗？
    3. 为什么 `case RED:` 是个 bug？
    4. 类模式 `case Point(x=0, y=y):` 是怎么工作的？
    5. 守卫（guard）是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 能匹配元组 `(1, 2)`：序列模式对列表和元组都适用；不能匹配字符串 `"ab"`：字符串、bytes 被特意排除在序列模式之外。
    2. 能：映射模式只要求列出的键存在且值匹配，多余的键不影响。
    3. 裸名字是捕获模式：`case RED:` 会匹配任何值，并把它赋给一个叫 `RED` 的变量，而不是和常量比较，后面的分支永远到不了。常量要用带点号的名字，比如 `case Color.RED:`。
    4. 先做 `isinstance(subject, Point)`，再逐个检查关键字：属性 `x` 要等于 0，属性 `y` 的值被捕获到变量 `y` 里。
    5. `case` 模式后面的 `if 条件`：模式匹配成功之后再检查这个条件，不满足就当作不匹配，继续尝试下一个分支。

## 基本形式

```python
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 401 | 403:                  # 或模式
            return "Not allowed"
        case 404:
            return "Not found"
        case _:                          # 通配符：什么都匹配，放在最后
            return "Something else"

assert http_error(403) == "Not allowed"
assert http_error(418) == "Something else"
```

从上到下尝试，第一个匹配成功的分支执行后就结束，不会"贯穿"到下一个分支。没有分支匹配时什么也不做（不报错）。

## 模式的种类

### 捕获与通配

裸名字是**捕获模式**：永远匹配成功，并把值绑定到这个名字。`_` 是特殊的通配符，匹配但不绑定。

### 序列模式

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
assert describe([3, 4]) == "at (3, 4)"          # 列表和元组都能匹配序列模式
assert describe((1, 2, 3, 4)) == "4D point"
assert describe("ab") == "not a point"          # 字符串不算序列模式
```

序列模式的括号写成 `()` 还是 `[]` 没有区别，都能匹配任意序列（`str`、`bytes`、`bytearray` 除外）。`*rest` 收集剩余元素。

### 映射模式

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

映射模式只要求**列出的键存在**，多余的键会被忽略（这和序列模式要求长度完全一致不同）。`**rest` 收集剩下的键值对。

### 类模式与守卫

`str(k)` 这种写法是**类模式**：先检查 `isinstance(value, str)`，再把值绑定到 `k`。上面还用了**守卫** `if len(k) == 1`：模式匹配成功后再检查条件，不满足就继续尝试下一个分支。

类模式配合 dataclass 非常好用：

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
        case Rect(Point(0, 0), w, h):          # 按位置匹配，依赖 __match_args__
            return f"rect at origin, area {w * h}"
        case Rect(w=w, h=h):
            return w * h

assert area(Circle(Point(0, 0), 1)) == 3.14159
assert area(Rect(Point(1, 1), 2, 2)) == "square of 4"
assert area(Rect(Point(0, 0), 2, 3)) == "rect at origin, area 6"
assert area(Rect(Point(1, 1), 2, 3)) == 6
```

按位置写的类模式 `Rect(Point(0, 0), w, h)` 依赖类的 `__match_args__` 属性，dataclass 会自动生成它（按字段顺序）。自己写的类可以手动定义 `__match_args__ = ("x", "y")`。

### `as` 绑定

在匹配子模式的同时，把整体绑定到一个名字：

```pycon
>>> match {"cmd": "move", "to": [1, 2]}:
...     case {"cmd": "move", "to": [_, _] as target}:
...         print("moving to", target)
...
moving to [1, 2]
```

## 最常见的 bug：把常量写成裸名字

```pycon
>>> RED = "red"
>>> color = "blue"
>>> match color:
...     case RED:                    # 这不是"等于 RED"，而是"捕获到名字 RED"
...         print("it's red?!")
...
it's red?!
>>> RED
'blue'
```

裸名字永远是捕获模式。它不仅匹配了任何值，还**覆盖了原来的变量**。好在如果它后面还有别的分支，Python 会直接报 `SyntaxError: name capture 'RED' makes remaining patterns unreachable`。

要和常量比较，必须用**带点号的名字**，比如枚举成员：

```python
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"

def paint(c):
    match c:
        case Color.RED:              # 带点号：值模式，用 == 比较
            return "stop"
        case Color.GREEN:
            return "go"

assert paint(Color.GREEN) == "go"
```

## 实战：命令解析

`match` 最适合"根据输入的形状做不同的事"：

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

同样的逻辑用 `if/elif` 写，需要大量的 `len(parts) == 2 and parts[0] == "go" and parts[1] in {...}`，远不如这样清楚。

## 什么时候不用 `match`

- 只是比较一个值等于几个常量之一，而且每个分支只是返回一个值：用字典查表更简单，`return MESSAGES.get(status, "unknown")`。
- 分支逻辑按类型分派，并且希望别的模块能扩展：考虑 `functools.singledispatch` 或多态方法。
- 条件是范围判断（`x > 10`）：`if/elif` 更直接。

## 练习

**1. 表达式求值器。** 用嵌套元组表示算术表达式：数字就是它本身，`("+", a, b)`、`("-", a, b)`、`("*", a, b)`、`("/", a, b)`、`("neg", a)`。写 `evaluate(expr)`，遇到除以零时抛 `ZeroDivisionError`，遇到不认识的结构时抛 `ValueError`。

??? success "参考答案"
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

    `int()` 这种不带参数的类模式只做 `isinstance` 检查。注意 `bool` 是 `int` 的子类，`True` 也会被当成数字，严格的实现要单独排除。

**2. Webhook 事件路由。** 处理如下格式的事件字典，返回一句描述：

- `{"event": "push", "repo": {"name": n}, "commits": [...]}` → `"n: 3 commits"`（没有提交时返回 `"n: empty push"`）
- `{"event": "pull_request", "action": "opened" 或 "reopened", "number": k}` → `"PR #k opened"`
- `{"event": "pull_request", "action": "closed", "merged": True, "number": k}` → `"PR #k merged"`
- 其他 → `"ignored"`

??? success "参考答案"
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

## 小结

- [x] `match` 同时做结构判断和数据提取，从上到下第一个匹配的分支执行。
- [x] 序列模式要求长度一致（可用 `*rest`）；映射模式只要求列出的键存在。
- [x] 类模式 `Cls(attr=pattern)` 先做 `isinstance` 再匹配属性；位置写法依赖 `__match_args__`。
- [x] 裸名字是捕获，不是比较；常量要用带点号的名字，比如枚举成员。
- [x] 守卫 `if ...` 补充模式表达不了的条件。
