"""Playground：在浏览器里（Pyodide）运行一段自由的 Python 代码，收集输出、异常和 matplotlib 图。

支持顶层 await（浏览器里的事件循环已经在运行，不能再用 asyncio.run）。
"""

from __future__ import annotations

import ast
import base64
import contextlib
import io
import linecache
import re
import sys
import time
import traceback

FILENAME = "<playground>"
MAX_OUT = 200_000


class _Capped(io.StringIO):
    """超过上限就不再记录，免得死循环里的 print 把页面撑爆"""

    def __init__(self):
        super().__init__()
        self.dropped = False

    def write(self, s):
        if self.tell() + len(s) > MAX_OUT:
            if not self.dropped:
                super().write("\n…（输出超过 200 KB，后面的不再显示）\n")
                self.dropped = True
            return len(s)
        return super().write(s)


def _traceback(exc: BaseException) -> str:
    """只保留 Playground 代码里的栈帧（外加异常本身），隐藏运行器"""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == FILENAME]
    lines = [f'  第 {f.lineno} 行，{f.name if f.name != "<module>" else "顶层"}\n    {f.line.strip()}' for f in frames if f.line]
    head = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return ("Traceback:\n" + "\n".join(lines) + "\n" + head) if lines else head


def _figures() -> list[str]:
    """代码里画了 matplotlib 图的话，把每张图转成 PNG（base64）"""
    if "matplotlib.pyplot" not in sys.modules:
        return []
    plt = sys.modules["matplotlib.pyplot"]
    images = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        images.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return images


async def run(code: str) -> dict:
    linecache.cache[FILENAME] = (len(code), None, code.splitlines(True), FILENAME)
    if re.search(r"^\s*(import|from)\s+matplotlib", code, re.M):
        import matplotlib

        matplotlib.use("Agg")                         # 浏览器里没有窗口，画到内存里再转成图片
    out = _Capped()
    env = {"__name__": "__main__", "__file__": FILENAME}
    error = None
    start = time.perf_counter()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            compiled = compile(code, FILENAME, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)
            result = eval(compiled, env)
            if result is not None and hasattr(result, "__await__"):
                await result
        except SyntaxError as e:
            error = f"第 {e.lineno} 行：{type(e).__name__}: {e.msg}"
        except BaseException as e:  # noqa: BLE001  用户代码里的任何异常（包括 SystemExit）都要显示出来
            error = _traceback(e)
    ms = (time.perf_counter() - start) * 1000
    images = []
    try:
        images = _figures()
    except Exception as e:  # noqa: BLE001
        error = (error + "\n" if error else "") + f"画图失败：{e}"
    return {"stdout": out.getvalue(), "error": error, "ms": ms, "images": images}
