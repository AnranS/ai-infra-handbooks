"""编辑器的代码补全、函数签名和语法检查。

在判题 worker 里运行（Pyodide 自带 Jedi），只做静态分析，不执行用户代码。每个函数都返回 JSON 字符串，
行号从 1 开始、列号从 0 开始（与 Jedi 相同）。
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings

STUBS = "/tmp/assist-stubs"
MAX_ITEMS = 80
# 题目里的模拟器在判题时才装进 sys.modules，静态分析看不到：给 Jedi 一份只用于补全的存根
_TRITON_STUB = {
    "triton/__init__.py": "from minitl import cdiv, jit\nfrom . import language\n\n\n"
                          "def next_power_of_2(n: int) -> int:\n    \"\"\"不小于 n 的最小 2 的幂\"\"\"\n",
    "triton/language/__init__.py": "from minitl import *  # noqa: F403\nfrom minitl import constexpr, sum, max, min, abs  # noqa: F401\n",
}
_KERNEL_DEF = re.compile(r"^(\s*def\s+\w+\s*\(\s*)(\w+)(\s*[,)])")

jedi = None
_env = None
_project = None
_last: list = []                      # 最近一次补全的结果，detail() 按下标取文档
_last_token = 0


def _setup():
    global _env, _project, jedi
    if _env is None:
        import jedi                   # 按需加载：语法检查不需要 Jedi，worker 在第一次补全前才 loadPackage("jedi")

        for rel, text in _TRITON_STUB.items():
            os.makedirs(os.path.dirname(os.path.join(STUBS, rel)), exist_ok=True)
            with open(os.path.join(STUBS, rel), "w", encoding="utf-8") as f:
                f.write(text)
        os.makedirs("/tmp/assist-project", exist_ok=True)
        _env = jedi.InterpreterEnvironment()
        _project = jedi.Project("/tmp/assist-project", added_sys_path=[STUBS])
    return _env, _project


def _annotate_kernels(code: str, line: int, col: int):
    """gpusim 的 kernel 第一个参数是线程上下文：补上类型标注，Jedi 才知道 t. 后面有什么。"""
    alias = None
    m = re.search(r"^\s*import\s+gpusim(?:\s+as\s+(\w+))?\s*$", code, re.M)
    if m:
        alias = m.group(1) or "gpusim"
    if alias is None:
        return code, col
    lines = code.split("\n")
    for i in range(1, len(lines)):
        prev = lines[i - 1].strip()
        if not (prev.startswith("@") and prev.endswith("kernel")):
            continue
        mm = _KERNEL_DEF.match(lines[i])
        if not mm:
            continue
        ann = f': "{alias}.Thread"'
        pos = mm.end(2)
        lines[i] = lines[i][:pos] + ann + lines[i][pos:]
        if i + 1 == line and col >= pos:
            col += len(ann)
    return "\n".join(lines), col


def _script(code: str):
    env, project = _setup()
    return jedi.Script(code, path="/tmp/assist-project/solution.py", environment=env, project=project)


def _first_paragraph(doc: str, limit: int = 600) -> str:
    doc = doc.strip()
    return doc if len(doc) <= limit else doc[:limit].rsplit("\n", 1)[0] + "\n…"


def complete(code: str, line: int, col: int) -> str:
    """补全候选：[{name, insert, type}]；按下标调用 detail() 取签名和文档"""
    global _last, _last_token
    code, col = _annotate_kernels(code, line, col)
    try:
        comps = _script(code).complete(line, col)
    except Exception as e:  # noqa: BLE001  Jedi 在半截代码上偶尔会失败，补全失败不应该影响做题
        return json.dumps({"items": [], "error": repr(e)})
    prefix = re.search(r"[A-Za-z_]\w*$", code.split("\n")[line - 1][:col])
    prefix = prefix.group(0) if prefix else ""
    if not prefix.startswith("_"):                  # 没打下划线就不列私有名字
        comps = [c for c in comps if not c.name.startswith("_")]
    elif not prefix.startswith("__"):
        comps = [c for c in comps if not c.name.startswith("__")]
    comps = [c for c in comps if c.name != prefix]  # 已经打全了的就不用再提示
    # 大小写完全匹配前缀的排前面，其余保持 Jedi 的顺序
    comps.sort(key=lambda c: not c.name.startswith(prefix))
    _last, _last_token = comps[:MAX_ITEMS], _last_token + 1
    items = [{"name": c.name, "insert": c.name_with_symbols, "type": c.type} for c in _last]
    return json.dumps({"items": items, "token": _last_token, "prefix": prefix}, ensure_ascii=False)


def detail(token: int, index: int) -> str:
    """补全列表里第 index 项的签名和文档（选中时才取，避免一次算几十个）"""
    if token != _last_token or not 0 <= index < len(_last):
        return json.dumps(None)
    c = _last[index]
    try:
        sigs = [s.to_string() for s in c.get_signatures()[:3]]
        doc = c.docstring(raw=True) if sigs else c.docstring()
    except Exception:  # noqa: BLE001
        sigs, doc = [], ""
    return json.dumps({"name": c.name, "type": c.type, "signatures": sigs, "doc": _first_paragraph(doc or "")}, ensure_ascii=False)


def signatures(code: str, line: int, col: int) -> str:
    """光标在函数调用的括号里时：函数签名、每个参数和当前是第几个参数"""
    code, col = _annotate_kernels(code, line, col)
    try:
        sigs = _script(code).get_signatures(line, col)
    except Exception:  # noqa: BLE001
        sigs = []
    out = []
    for s in sigs[:3]:
        try:
            doc = s.docstring(raw=True)
        except Exception:  # noqa: BLE001
            doc = ""
        out.append({"name": s.name, "params": [p.to_string() for p in s.params], "index": s.index,
                    "doc": _first_paragraph(doc or "", 300)})
    return json.dumps(out, ensure_ascii=False)


def check(code: str) -> str:
    """语法检查：用 CPython 自己的解析器，结果和提交时一致。只报第一个错误（解析器遇错即停）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # 无效转义之类的 SyntaxWarning 不算错
        try:
            ast.parse(code, filename="solution.py")
        except SyntaxError as e:
            line = e.lineno or 1
            col = max((e.offset or 1) - 1, 0)
            end_line = e.end_lineno or line
            end_col = (e.end_offset - 1) if e.end_offset else col + 1
            if (end_line, end_col) <= (line, col):   # 有些错误（例如缩进不一致）没有给出有效的结束位置
                end_line, end_col = line, col + 1
            return json.dumps([{"line": line, "col": col, "end_line": end_line, "end_col": end_col,
                                "msg": f"{type(e).__name__}: {e.msg}"}], ensure_ascii=False)
    return "[]"


def warm() -> str:
    """加载 Jedi 后先跑一次，把内置函数和常用模块的存根解析好，第一次补全就不会慢"""
    code = "import collections, heapq, math\nx = []\nx.a\ncollections.d\n"
    complete(code, 3, 3)
    complete(code, 4, 13)
    signatures("print(", 1, 6)
    return "ok"
