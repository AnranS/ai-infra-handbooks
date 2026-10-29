"""MkDocs hook（C++ 手册用）：给能单独运行的 C++ 例子加一个"运行"按钮的占位。

```cpp title="x.cpp"``` 是一个完整程序，cpp/tools/check_code.py 构建时会编译运行它。这个钩子在每个这样的代码块后面
插一个 <div class="aig-cpprun">，带上书里用的编译参数（sanitizer、-Werror、flags=）；页面上的 aig-cpprun.js 看到它，
就在代码块下面放"运行"按钮，把代码交给 Compiler Explorer（godbolt.org）编译运行。

多文件的例子（lib="yes"、with=、project=）跑不了，不加按钮。同一页里的 .hpp / .h 头文件由前端一起发过去。
必须排在 fence_attrs.py 前面：那个钩子会把这些属性从代码块上去掉。
"""

import html
import re
import shlex

# 与 cpp/tools/check_code.py 的 BASE_FLAGS 和 sanitizer 规则保持一致
BASE_FLAGS = ["-std=c++20", "-g", "-O1", "-Wall", "-Wextra", "-pthread"]

_OPEN = re.compile(r"^([ \t]*)(`{3,})cpp\b(.*)$")
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _flags(attrs):
    san = attrs.get("sanitize", "address,undefined")
    flags = list(BASE_FLAGS)
    if san != "none":
        flags += [f"-fsanitize={san}", "-fno-omit-frame-pointer"]
    if attrs.get("expect") not in ("fail", "compile-error"):
        flags.append("-Werror")
    flags += shlex.split(attrs.get("flags", "")) + shlex.split(attrs.get("libs", ""))
    return san, " ".join(flags)


def on_page_markdown(markdown, page, config, files):
    lines = markdown.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, ticks, rest = m.groups()
        attrs = dict(_ATTR.findall(rest))
        end = i + 1                                     # 找到同样缩进、同样长度的收尾
        while end < len(lines) and lines[end].strip() != ticks:
            end += 1
        out.extend(lines[i:end + 1])
        i = end + 1
        title = attrs.get("title", "")
        if not title.endswith(".cpp") or attrs.get("lib") == "yes" or "with" in attrs or "project" in attrs:
            continue
        san, flags = _flags(attrs)
        marker = (f'<div class="aig-cpprun" data-flags="{html.escape(flags)}" data-sanitize="{html.escape(san)}" '
                  f'data-expect="{html.escape(attrs.get("expect", ""))}"></div>')
        out.extend(["", indent + marker, ""])
    return "\n".join(out)
