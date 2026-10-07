"""生成手册里的架构图（SVG）。

每个图是一个函数，用下面这个很小的绘图工具画出来，写到 docs/assets/diagrams/<名字>.svg。
正文里用 @@diagram 名字 标题@@ 引用，hooks/include_code.py 把 SVG 内联进页面。
SVG 里只用 CSS 类（颜色在 docs/assets/extra.css 里定义），所以能跟随网站的亮色 / 暗色主题。

    python tools/diagrams.py            # 生成全部
    python tools/diagrams.py radix      # 只生成名字包含 radix 的

英文版：--lang en 把图里每一段文字按 ../i18n/en/figures.json（中文原文 → 英文）替换，
写到 docs-en/assets/diagrams/；没有译文的原样保留，--missing 只列出缺的那些。

    python tools/diagrams.py --lang en            # 生成全部英文图
    python tools/diagrams.py --lang en --missing  # 只列出还没译文的文字
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path
from typing import Callable, Dict, List

BOOK = Path(__file__).resolve().parent.parent
OUT = BOOK / "docs" / "assets" / "diagrams"
COLORS = ("blue", "green", "orange", "purple", "pink", "teal", "red", "yellow", "gray")

# 英文版的译文表（与其他书共用 i18n/en/figures.json）
_TR: dict = {"map": {}, "missing": set()}


def _t(s: str) -> str:
    s = str(s)
    if not _TR["map"] or not re.search(r"[㐀-鿿]", s):
        return s
    if s in _TR["map"]:
        return _TR["map"][s]
    _TR["missing"].add(s)
    return s


class D:
    """一张图。坐标单位是像素（viewBox），最终按容器宽度缩放。"""

    def __init__(self, name: str, w: int, h: int):
        self.name, self.w, self.h = name, w, h
        self.items: List[str] = []

    # ---------------------------------------------------------------- 基本元素
    def box(self, x, y, w, h, text: str | None = None, c: str = "gray", sub: str | None = None,
            r: int = 8, dash: bool = False, fs: int = 14, mono: bool = False, bold: bool = True,
            solid: bool = False, align: str = "middle", ty: float | None = None):
        cls = f"dg-{'solid' if solid else 'fill'}-{c}" + (" dg-dash" if dash else "")
        self.items.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>')
        if text is not None:
            if align == "middle":
                tx = x + w / 2
            else:
                tx = x + 12
            if sub:
                cy = y + h / 2 - fs * 0.45 if ty is None else ty
                self.text(tx, cy, text, fs=fs, mono=mono, bold=bold, anchor=align,
                          cls="dg-on" if solid else "dg-t")
                self.text(tx, cy + fs * 1.25, sub, fs=fs - 2, mono=mono, cls="dg-on2" if solid else "dg-t2",
                          anchor=align)
            else:
                cy = y + h / 2 + fs * 0.35 if ty is None else ty
                self.text(tx, cy, text, fs=fs, mono=mono, bold=bold, anchor=align,
                          cls="dg-on" if solid else "dg-t")

    def rect(self, x, y, w, h, cls: str, r: int = 0):
        self.items.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>')

    def text(self, x, y, s: str, fs: int = 14, cls: str = "dg-t", anchor: str = "middle",
             mono: bool = False, bold: bool = False):
        cls += " dg-mono" if mono else ""
        weight = ' font-weight="600"' if bold else ""
        anchor = {"middle": "middle", "start": "start", "end": "end"}[anchor]
        lines = _t(s).split("\n")
        for i, line in enumerate(lines):
            self.items.append(f'<text class="{cls}" x="{x}" y="{y + i * fs * 1.3:.1f}" font-size="{fs}"'
                              f' text-anchor="{anchor}"{weight}>{escape(line)}</text>')

    def line(self, x1, y1, x2, y2, c: str = "gray", dash: bool = False, width: float = 1.5,
             arrow: bool = False, both: bool = False):
        cls = f"dg-ln-{c}" + (" dg-dash" if dash else "")
        mk = f' marker-end="url(#{self.name}-a-{c})"' if arrow or both else ""
        mk += f' marker-start="url(#{self.name}-s-{c})"' if both else ""
        self.items.append(f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
                          f' stroke-width="{width}"{mk}/>')

    def arrow(self, x1, y1, x2, y2, c: str = "gray", label: str | None = None, dash: bool = False,
              lx: float | None = None, ly: float | None = None, fs: int = 12, both: bool = False,
              anchor: str = "middle", width: float = 1.6):
        self.line(x1, y1, x2, y2, c, dash, width=width, arrow=True, both=both)
        if label:
            mx = (x1 + x2) / 2 if lx is None else lx
            my = (y1 + y2) / 2 - 6 if ly is None else ly
            self.text(mx, my, label, fs=fs, cls=f"dg-tc-{c}" if c != "gray" else "dg-t2", anchor=anchor)

    def path(self, d: str, c: str = "gray", arrow: bool = False, dash: bool = False, width: float = 1.6):
        cls = f"dg-ln-{c}" + (" dg-dash" if dash else "")
        mk = f' marker-end="url(#{self.name}-a-{c})"' if arrow else ""
        self.items.append(f'<path class="{cls}" d="{d}" fill="none" stroke-width="{width}"{mk}/>')

    def pill(self, x, y, s: str, c: str = "blue", fs: int = 11):
        s = _t(s)
        w = 8 + len(s) * fs * 0.62 + (s.count("") - len(s)) * 0
        w = max(w, fs * 2)
        w = 10 + sum(fs if ord(ch) > 0x2E80 else fs * 0.6 for ch in s)
        self.items.append(f'<rect class="dg-solid-{c}" x="{x}" y="{y - fs}" width="{w:.0f}" height="{fs + 7}" rx="{(fs + 7) / 2}"/>')
        self.text(x + w / 2, y + 1, s, fs=fs, cls="dg-on", bold=True)
        return w

    def cell(self, x, y, w, h, s: str = "", c: str | None = None, fs: int = 11, mono: bool = True,
             cls_text: str = "dg-t"):
        cls = f"dg-fill-{c}" if c else "dg-cell"
        self.items.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        if s != "":
            self.text(x + w / 2, y + h / 2 + fs * 0.36, str(s), fs=fs, mono=mono, cls=cls_text)

    # ---------------------------------------------------------------- 输出
    def svg(self) -> str:
        defs = []
        for c in COLORS:
            defs.append(f'<marker id="{self.name}-a-{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"'
                        f' markerHeight="7" orient="auto-start-reverse"><path class="dg-mk-{c}" d="M0 1 10 5 0 9z"/></marker>')
            defs.append(f'<marker id="{self.name}-s-{c}" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="7"'
                        f' markerHeight="7" orient="auto"><path class="dg-mk-{c}" d="M10 1 0 5 10 9z"/></marker>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" class="dg-svg"'
                f' style="max-width:{self.w}px" role="img"><defs>{"".join(defs)}</defs>'
                + "\n".join(self.items) + "</svg>\n")


DIAGRAMS: Dict[str, Callable[[], D]] = {}


def diagram(fn: Callable[[], D]) -> Callable[[], D]:
    DIAGRAMS[fn.__name__.replace("_", "-")] = fn
    return fn


# ====================================================================== 导读
@diagram
def processes() -> D:
    d = D("processes", 980, 470)
    d.box(20, 175, 110, 70, "客户端", "gray", "OpenAI SDK / curl")
    d.box(190, 150, 190, 120, "API Server", "blue", None, ty=180)
    d.text(285, 208, "FastAPI + uvicorn", fs=12, cls="dg-t2")
    d.text(285, 228, "FrontendManager", fs=12, cls="dg-t2", mono=True)
    d.text(285, 248, "（主进程）", fs=12, cls="dg-t2")
    d.box(450, 150, 190, 120, "tokenizer", "orange", None, ty=180)
    d.text(545, 208, "分词 / 套对话模板", fs=12, cls="dg-t2")
    d.text(545, 228, "增量反分词", fs=12, cls="dg-t2")
    d.text(545, 248, "（默认两者同一进程）", fs=12, cls="dg-t2")
    # 调度器 rank 0 / rank 1
    for i, y in enumerate((40, 290)):
        d.box(720, y, 240, 140, f"调度器进程 rank {i}", "green", None, dash=False, ty=y + 26)
        d.box(735, y + 42, 100, 80, "Scheduler", "green", "调度、KV 分配", fs=13, solid=False)
        d.box(845, y + 42, 100, 80, "Engine", "purple", f"模型 · GPU {i}", fs=13)
    d.arrow(130, 200, 188, 200, "gray", both=True)
    d.text(159, 190, "HTTP", fs=11, cls="dg-t2")
    d.text(159, 230, "SSE", fs=11, cls="dg-t2")
    d.arrow(380, 190, 448, 190, "blue", "TokenizeMsg", ly=180)
    d.arrow(448, 232, 382, 232, "orange", "UserReply", ly=252)
    d.arrow(640, 180, 735, 125, "orange", "UserMsg", lx=672, ly=140)
    d.arrow(735, 150, 640, 205, "green", "DetokenizeMsg", lx=712, ly=198, anchor="middle")
    d.arrow(752, 162, 752, 330, "green", dash=True)
    d.text(744, 300, "ZMQ PUB：原样转发", fs=11, cls="dg-tc-green", anchor="end")
    d.text(744, 316, "+ gloo 广播条数", fs=11, cls="dg-tc-green", anchor="end")
    d.arrow(928, 164, 928, 330, "purple", both=True)
    d.text(920, 226, "NCCL", fs=11, cls="dg-tc-purple", anchor="end")
    d.text(920, 242, "all-reduce", fs=11, cls="dg-tc-purple", anchor="end")
    d.text(920, 258, "all-gather", fs=11, cls="dg-tc-purple", anchor="end")
    d.text(420, 360, "控制消息：ZMQ（msgpack 编码的 dataclass）", fs=12, cls="dg-t2", anchor="middle")
    d.text(420, 380, "张量：NCCL（GPU）/ gloo（CPU）", fs=12, cls="dg-t2", anchor="middle")
    d.text(420, 430, "TP=N 时共 N + 2 个进程：API Server、tokenizer、N 个调度器", fs=13, cls="dg-t", anchor="middle",
           bold=True)
    return d


@diagram
def request_lifecycle() -> D:
    d = D("request-lifecycle", 980, 610)
    lanes = [("客户端", "gray"), ("API Server", "blue"), ("tokenizer", "orange"), ("Scheduler", "green"),
             ("Engine", "purple")]
    xs = [80, 280, 480, 680, 880]
    for (name, c), x in zip(lanes, xs):
        d.box(x - 70, 14, 140, 38, name, c, fs=14)
        d.line(x, 52, x, 596, "gray", dash=True, width=1)
    rows = [
        (0, 1, "① POST /v1/chat/completions", "gray"),
        (1, 2, "② TokenizeMsg(uid, 对话)", "blue"),
        (2, 3, "③ UserMsg(uid, token 张量)", "orange"),
        (3, 3, "④ 排队 → 匹配前缀 → 准入", "green"),
        (3, 4, "⑤ prefill batch：位置、out_loc、注意力元数据", "green"),
        (4, 3, "⑥ 采样出第一个 token（GPU 上写回 token pool）", "purple"),
        (3, 2, "⑦ DetokenizeMsg(uid, token, finished)", "green"),
        (2, 1, "⑧ UserReply(uid, 增量文本)", "orange"),
        (1, 0, "⑨ SSE：data: {delta}", "blue"),
        (3, 4, "⑩ decode batch（每轮每个请求 1 个 token）", "green"),
        (4, 3, "⑪ 下一个 token …… 直到 EOS 或 max_tokens", "purple"),
        (3, 3, "⑫ 结束：归还请求槽，KV 进 Radix Cache", "green"),
    ]
    y = 88
    for a, b, label, c in rows:
        if a == b:
            d.path(f"M{xs[a]} {y - 8} h 36 v 24 h -34", c, arrow=True)
            d.text(xs[a] + 44, y + 8, label, fs=12, cls=f"dg-tc-{c}" if c != "gray" else "dg-t2", anchor="start")
            y += 50
            continue
        d.arrow(xs[a], y, xs[b] + (-4 if b > a else 4), y, c)
        lx = (xs[a] + xs[b]) / 2
        if abs(a - b) == 1:
            d.text(lx, y - 7, label, fs=12, cls=f"dg-tc-{c}" if c != "gray" else "dg-t2")
        else:
            d.text(lx, y - 7, label, fs=12, cls=f"dg-tc-{c}" if c != "gray" else "dg-t2")
        y += 42
    return d


@diagram
def modules() -> D:
    d = D("modules", 980, 440)
    layers = [
        ("服务化", "orange", [("server", "API Server、启动器"), ("llm", "离线接口"), ("tokenizer", "分词进程"),
                               ("message", "消息与序列化")]),
        ("调度", "green", [("scheduler", "Scheduler 与四个 Manager")]),
        ("引擎", "purple", [("engine", "Engine · Sampler · GraphRunner")]),
        ("计算组件", "blue", [("models", "模型与权重"), ("attention", "注意力后端"), ("kvcache", "KV 池 · 前缀缓存"),
                              ("moe", "MoE 后端")]),
        ("基础", "gray", [("layers", "BaseOP 与各种层"), ("kernel", "算子"), ("distributed", "TP 通信"),
                           ("core", "Req · Batch · Context"), ("utils", "设备抽象 · ZMQ · 注册表")]),
    ]
    y = 16
    for title, c, mods in layers:
        d.box(16, y, 948, 68, None, c, r=12)
        d.text(34, y + 40, title, fs=15, bold=True, anchor="start", cls=f"dg-tc-{c}" if c != "gray" else "dg-t")
        n = len(mods)
        x0, w_total = 130, 820
        gap = 12
        bw = (w_total - gap * (n - 1)) / n
        for i, (m, desc) in enumerate(mods):
            x = x0 + i * (bw + gap)
            d.box(x, y + 11, bw, 46, m, c, desc, fs=13, mono=False, solid=False)
        y += 80
    d.text(490, 428, "上层依赖下层；模型的每一层通过 core.Context 读取当前 batch，通过注册表选择后端", fs=12,
           cls="dg-t2")
    return d

# ====================================================================== 第一部分：算得对
@diagram
def req_lengths() -> D:
    d = D("req-lengths", 980, 340)
    cw, x0 = 62, 150
    toks = ["p0", "p1", "p2", "p3", "p4", "p5", "t0", "t1", "t2"]
    stages = [("接纳时", 2, 6), ("prefill 之后", 6, 7), ("decode 1 之后", 7, 8)]
    for row, (name, cached, device) in enumerate(stages):
        y = 36 + row * 95
        d.text(x0 - 16, y + 26, name, fs=13, anchor="end", bold=True)
        for i, t in enumerate(toks):
            if i < cached:
                c = "green"
            elif i < device:
                c = "blue"
            else:
                c = None
            d.cell(x0 + i * cw, y + 6, cw, 30, t if i < device or i < 6 else t, c, fs=12)
        # 标尺
        d.line(x0, y + 46, x0 + cached * cw, y + 46, "green", width=2)
        d.text(x0 + cached * cw / 2, y + 62, f"cached_len = {cached}", fs=12, cls="dg-tc-green")
        d.line(x0 + cached * cw, y + 46, x0 + device * cw, y + 46, "blue", width=2)
        d.text(x0 + (cached + device) * cw / 2, y + 62, f"extend_len = {device - cached}", fs=12, cls="dg-tc-blue")
        d.line(x0 + device * cw, y, x0 + device * cw, y + 42, "gray", width=2)
        d.text(x0 + device * cw, y - 4, f"device_len = {device}", fs=12, cls="dg-t2")
    d.line(x0 + 9 * cw, 20, x0 + 9 * cw, 300, "orange", dash=True)
    d.text(x0 + 9 * cw + 8, 32, "max_device_len = 9", fs=12, cls="dg-tc-orange", anchor="start")
    d.text(x0 + 9 * cw + 8, 48, "（提示词 6 + max_tokens 3）", fs=11, cls="dg-tc-orange", anchor="start")
    d.pill(40, 330, "已在 KV 缓存", "green")
    d.pill(150, 330, "本轮要算", "blue")
    return d


@diagram
def kv_layout() -> D:
    d = D("kv-layout", 980, 440)
    ps, cw, ch = 4, 22, 26
    # KV 池：12 页 × 4
    d.text(20, 24, "KV 池（某一层的 K；page_size = 4）", fs=13, anchor="start", bold=True)
    owner = {0: "gray", 2: "blue", 5: "blue", 1: "green", 7: "green", 9: "green", 12: "yellow"}
    for pg in range(13):
        col, row = pg % 4, pg // 4
        x, y = 20 + col * (ps * cw + 14), 40 + row * (ch + 24)
        c = owner.get(pg)
        for k in range(ps):
            used = c is not None and not (pg == 5 and k >= 2) and not (pg == 9 and k >= 1)
            d.cell(x + k * cw, y, cw, ch, str(pg * ps + k), c if used else None, fs=9)
        d.text(x + ps * cw / 2, y + ch + 13, f"第 {pg} 页" + ("（dummy）" if pg == 12 else ""), fs=10, cls="dg-t2")
    d.pill(20, 262, "请求 A", "blue")
    d.pill(92, 262, "请求 B", "green")
    d.pill(164, 262, "Radix Cache 中", "gray")
    d.pill(290, 262, "dummy 页", "yellow")
    d.text(20, 300, "形状 [2, 层, 页, 页大小, KV 头, head_dim]", fs=12, cls="dg-t2", anchor="start", mono=True)
    d.text(20, 320, "按 token 展平后：第 j 行 = 池中第 j 个 token 的位置", fs=12, cls="dg-t2", anchor="start")
    # page table
    x1 = 470
    d.text(x1, 24, "page table（按 token 存位置）", fs=13, anchor="start", bold=True)
    rows = [("A", "blue", [8, 9, 10, 11, 20, 21]), ("B", "green", [4, 5, 6, 7, 28, 29, 30, 31, 36])]
    for r, (name, c, locs) in enumerate(rows):
        y = 44 + r * 44
        d.text(x1 + 18, y + 18, f"行 {r}", fs=12, cls="dg-t2")
        for j in range(10):
            d.cell(x1 + 40 + j * 42, y, 42, 28, str(locs[j]) if j < len(locs) else "", c if j < len(locs) else None, fs=11)
    d.text(x1 + 18, 150, "…", fs=14, cls="dg-t2")
    d.text(x1 + 40, 172, "out_loc = page_table[行, cached_len:device_len]", fs=12, cls="dg-t2", anchor="start", mono=True)
    d.text(x1 + 40, 190, "FlashAttention 需要页号：page_table[行, ::page_size] // page_size", fs=12, cls="dg-t2",
           anchor="start", mono=True)
    # token pool
    d.text(x1, 232, "token pool（同形状，存 token id，在设备上）", fs=13, anchor="start", bold=True)
    ids = [("A", "blue", [785, 6722, 315, 9625, 374, 12095, "→"]), ("B", "green", [16, 488, 220, 16, 284, 220, 17, 11, 220, "→"])]
    for r, (name, c, tk) in enumerate(ids):
        y = 250 + r * 44
        d.text(x1 + 18, y + 18, f"行 {r}", fs=12, cls="dg-t2")
        for j in range(10):
            v = tk[j] if j < len(tk) else ""
            d.cell(x1 + 40 + j * 42, y, 42, 28, str(v), c if (j < len(tk) and v != "→") else None, fs=10)
    d.text(x1 + 40, 358, "input_ids = token_pool[行, 位置]", fs=12, cls="dg-t2", anchor="start", mono=True)
    d.text(x1 + 40, 376, "token_pool[行, device_len] = 采样结果   ← 下一轮的输入，不经过 CPU", fs=12, cls="dg-t2",
           anchor="start", mono=True)
    d.arrow(x1 + 40 + 5 * 42 + 21, 72, 20 + 1 * (ps * cw + 14) + 1.5 * cw, 90, "blue", dash=True)
    return d


@diagram
def varlen_batch() -> D:
    d = D("varlen-batch", 980, 330)
    x0, cw = 40, 44
    d.text(x0, 26, "本轮的 q（各请求首尾相接，不做 padding）", fs=13, anchor="start", bold=True)
    segs = [("A", "blue", 6), ("B", "green", 3), ("C", "orange", 1)]
    i = 0
    for name, c, n in segs:
        for k in range(n):
            d.cell(x0 + i * cw, 40, cw, 30, f"{name}{k}", c, fs=11)
            i += 1
    for pos, v in [(0, 0), (6, 6), (9, 9), (10, 10)]:
        d.line(x0 + pos * cw, 36, x0 + pos * cw, 80, "gray")
        d.text(x0 + pos * cw, 94, str(v), fs=12, cls="dg-t2", mono=True)
    d.text(x0 + 10 * cw + 16, 60, "cu_seqlens_q = [0, 6, 9, 10]", fs=13, anchor="start", mono=True)
    d.text(x0 + 10 * cw + 16, 80, "get_last_indices → [5, 8, 9]", fs=13, anchor="start", mono=True, cls="dg-t2")
    # KV
    d.text(x0, 132, "每个请求的全部 KV（按 page table 从池中取）", fs=13, anchor="start", bold=True)
    kv = [("A", "blue", 6, 0, [0, 1, 2, 3, 4, 5]), ("B", "green", 7, 4, [40, 41, 42, 43, 20, 21, 22]),
          ("C", "orange", 9, 8, [50, 51, 52, 53, 54, 55, 56, 57, 58])]
    for r, (name, c, n, cached, locs) in enumerate(kv):
        y = 146 + r * 40
        d.text(x0 - 10, y + 20, name, fs=13, anchor="end", bold=True)
        for k in range(n):
            d.cell(x0 + k * cw, y, cw, 30, str(locs[k]), c if k >= cached else "gray", fs=11)
        d.text(x0 + 9 * cw + 16, y + 20, f"cache_seqlens = {n}" + (f"（前 {cached} 个已缓存）" if cached else ""), fs=12,
               anchor="start", cls="dg-t2")
    d.pill(x0, 286, "已在缓存", "gray")
    d.pill(x0 + 80, 286, "本轮新写入（out_loc）", "blue")
    # 因果掩码 B
    mx, my, s = 690, 150, 26
    d.text(mx, 132, "请求 B 的因果掩码（右下角对齐）", fs=13, anchor="start", bold=True)
    for r in range(3):
        for k in range(7):
            ok = k <= 4 + r
            d.cell(mx + k * s, my + r * s, s, s, "", "green" if ok else None)
        d.text(mx - 8, my + r * s + 17, f"q{r}", fs=11, anchor="end", cls="dg-t2", mono=True)
    for k in range(7):
        d.text(mx + k * s + s / 2, my + 3 * s + 16, str(k), fs=10, cls="dg-t2", mono=True)
    d.text(mx, my + 3 * s + 38, "q0 的位置是 4：能看到 key 0～4", fs=12, anchor="start", cls="dg-t2")
    d.text(mx, my + 3 * s + 58, "本轮的 n 个 query 是序列的最后 n 个位置", fs=12, anchor="start", cls="dg-t2")
    return d


# ====================================================================== 第二部分：排得好
@diagram
def scheduler_loop() -> D:
    d = D("scheduler-loop", 980, 420)
    steps = [(90, 40, "① 收消息", "receive_msg（无事可做时阻塞）", "orange"),
             (90, 150, "② 选 batch", "有等待的请求 → prefill；否则 decode", "green"),
             (90, 260, "③ 准备并前向", "_prepare_batch → Engine.forward_batch", "purple"),
             (90, 370, "④ 处理结果", "追加 token、判断结束、回复、释放", "blue")]
    for x, y, t, sub, c in steps:
        d.box(x, y - 26, 300, 60, t, c, sub, fs=14)
    for a, b in ((40, 150), (150, 260), (260, 370)):
        d.arrow(240, a + 34, 240, b - 28, "gray")
    d.path("M 90 374 C 20 374 20 44 88 44", "gray", arrow=True, dash=True)
    d.text(40, 210, "下一轮", fs=12, cls="dg-t2")
    mgr = [(540, 14, 420, "SchedulerIOMixin", "ZMQ 收发 · 多 rank 同步（第 14 章）", "orange"),
           (540, 110, 200, "PrefillManager", "等待队列 · PrefillAdder", "green"),
           (760, 110, 200, "DecodeManager", "运行中的请求集合", "green"),
           (540, 206, 200, "CacheManager", "页分配 · 前缀缓存", "teal"),
           (760, 206, 200, "TableManager", "请求槽 · token pool", "teal"),
           (540, 302, 420, "Engine", "模型 · KV 池 · 注意力后端 · 采样 · CUDA Graph", "purple")]
    for x, y, w, t, sub, c in mgr:
        d.box(x, y, w, 64, t, c, sub, fs=14)
    d.arrow(392, 44, 538, 44, "orange")
    d.arrow(392, 150, 538, 142, "green")
    d.arrow(392, 258, 538, 238, "teal", dash=True)
    d.arrow(392, 268, 538, 330, "purple")
    d.text(750, 400, "Scheduler = 四个 Manager + Engine + IO；主循环见 normal_loop / overlap_loop", fs=12, cls="dg-t2")
    return d


@diagram
def cache_regions() -> D:
    d = D("cache-regions", 980, 230)
    x0, w = 40, 900
    parts = [(0.22, "gray", "接纳时命中的前缀", "锁着 → 解锁即可"),
             (0.2, "red", "自己算的，但缓存里已有", "重复的一份 → 释放"),
             (0.38, "green", "本次插入缓存", "所有权交给 Radix Cache"),
             (0.2, "orange", "不足一页的尾巴", "结束时释放，否则保留")]
    x = x0
    labels = ["0", "old.cached_len", "cached_len", "new.cached_len", "req.cached_len"]
    for i, (frac, c, t, sub) in enumerate(parts):
        ww = w * frac
        d.box(x, 60, ww, 70, t, c, sub, r=0, fs=13)
        d.text(x, 48, labels[i], fs=11, cls="dg-t2", mono=True, anchor="start" if i == 0 else "middle")
        x += ww
    d.text(x0 + w, 48, labels[-1], fs=11, cls="dg-t2", mono=True, anchor="end")
    d.text(x0, 170, "CacheManager.cache_req(req, finished)：prefill 刚结束时 finished=False，请求结束时 finished=True。", fs=12,
           cls="dg-t2", anchor="start")
    d.text(x0, 192, "红色段来自\"两个前缀相同的请求同时 prefill\"：先结束的已把前缀插入缓存，后结束的那份就是多余的。", fs=12,
           cls="dg-t2", anchor="start")
    return d


def _tree(d: D, x0: float, y0: float, title: str, nodes, edges, note: str):
    d.text(x0 + 140, y0, title, fs=13, bold=True)
    pos = {}
    for key, (x, y, label, c, ref) in nodes.items():
        pos[key] = (x0 + x, y0 + y)
        d.box(x0 + x - 44, y0 + y - 17, 88, 34, label, c, f"ref={ref}", fs=12, mono=True)
    for a, b in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        d.line(x1, y1 + 17, x2, y2 - 17, "gray")
    d.text(x0 + 140, y0 + 300, note, fs=12, cls="dg-t2")


@diagram
def radix_tree() -> D:
    d = D("radix-tree", 980, 360)
    _tree(d, 40, 26, "插入 [1,2,3,4]、[1,2,5,6]、[7,8]",
          {"r": (140, 40, "root", "gray", 1), "a": (80, 120, "[1,2]", "blue", 0), "b": (30, 210, "[3,4]", "blue", 0),
           "c": (130, 210, "[5,6]", "blue", 0), "d": (220, 120, "[7,8]", "blue", 0)},
          [("r", "a"), ("a", "b"), ("a", "c"), ("r", "d")], "可淘汰 6 · 受保护 0")
    _tree(d, 360, 26, "匹配 [1,2,3,9] 并加锁",
          {"r": (140, 40, "root", "gray", 1), "a": (80, 120, "[1,2]", "green", 1), "b": (30, 205, "[3]", "green", 1),
           "e": (30, 275, "[4]", "blue", 0), "c": (130, 205, "[5,6]", "blue", 0), "d": (220, 120, "[7,8]", "blue", 0)},
          [("r", "a"), ("a", "b"), ("b", "e"), ("a", "c"), ("r", "d")], "[3,4] 在匹配处分裂；路径上 ref+1")
    _tree(d, 680, 26, "淘汰 2 个，再淘汰 1 个",
          {"r": (140, 40, "root", "gray", 1), "a": (80, 120, "[1,2]", "green", 1), "b": (30, 205, "[3]", "green", 1)},
          [("r", "a"), ("a", "b")], "LRU：[4]、[5,6] 先走，然后 [7,8]")
    d.pill(20, 350, "受保护（ref>0）", "green")
    d.pill(150, 350, "可淘汰叶子", "blue")
    return d


@diagram
def chunked_prefill() -> D:
    d = D("chunked-prefill", 980, 310)
    x0, unit, y0 = 120, 22, 40
    rounds = [[("uid0", "blue", 16, "0:16")], [("uid0", "blue", 16, "16:32")], [("uid0", "blue", 16, "32:48")],
              [("uid0", "blue", 13, "48:61"), ("uid1", "green", 3, "0:3")],
              [("uid1", "green", 2, "3:5"), ("uid2", "orange", 5, "0:5")]]
    for i, parts in enumerate(rounds):
        y = y0 + i * 42
        d.text(x0 - 12, y + 20, f"第 {i + 1} 轮", fs=12, anchor="end", cls="dg-t2")
        x = x0
        for name, c, n, rng in parts:
            full = rng.endswith("61") or rng.endswith("3:5") or name == "uid2"
            label = f"{name} [{rng}]" if n * unit >= 90 else rng
            d.box(x, y, n * unit, 30, label, c, fs=11, bold=False, dash=not full)
            x += n * unit
    d.line(x0 + 16 * unit, y0 - 12, x0 + 16 * unit, y0 + 5 * 42, "red", dash=True)
    d.text(x0 + 16 * unit + 6, y0 - 16, "max_extend_tokens = 16", fs=12, cls="dg-tc-red", anchor="start")
    d.text(x0 + 16 * unit + 40, y0 + 40, "虚线框：ChunkedReq（不采样、不进 decode）", fs=12, cls="dg-t2", anchor="start")
    d.text(x0 + 16 * unit + 40, y0 + 62, "实线框：本块做完了这个请求的提示词", fs=12, cls="dg-t2", anchor="start")
    d.text(x0 + 16 * unit + 40, y0 + 84, "分块请求回到等待队列队首，下一轮优先继续", fs=12, cls="dg-t2", anchor="start")
    d.text(x0 + 16 * unit + 40, y0 + 106, "后面的块通过 page table 读前面块的 KV，与前缀命中相同", fs=12, cls="dg-t2", anchor="start")
    d.text(x0, 270, "三个请求都 prefill 完之后才开始 decode：一轮只做一种（prefill 优先）", fs=12, cls="dg-t2", anchor="start")
    d.pill(x0, 294, "uid0：61 个 token", "blue")
    d.pill(x0 + 140, 294, "uid1：5 个", "green")
    d.pill(x0 + 236, 294, "uid2：5 个", "orange")
    return d


@diagram
def overlap_timeline() -> D:
    d = D("overlap-timeline", 980, 330)
    def lane(y, title):
        d.text(20, y + 22, title, fs=12, anchor="start", cls="dg-t2")
    # 普通循环
    d.text(20, 24, "普通循环：CPU 和 GPU 轮流工作", fs=13, anchor="start", bold=True)
    lane(36, "CPU")
    lane(76, "GPU")
    x = 80
    for n in range(1, 5):
        d.box(x, 36, 70, 30, f"调度{n}", "orange", fs=11, bold=False)
        d.box(x + 70, 76, 90, 30, f"计算 {n}", "purple", fs=11, bold=False)
        d.box(x + 160, 36, 50, 30, f"处理{n}", "blue", fs=11, bold=False)
        x += 210
    d.text(930, 96, "GPU 空闲", fs=11, cls="dg-t2", anchor="end")
    # 重叠循环
    d.text(20, 164, "重叠循环：发射第 N+1 轮后再处理第 N 轮", fs=13, anchor="start", bold=True)
    lane(176, "CPU")
    lane(216, "GPU")
    d.box(80, 176, 70, 30, "调度1", "orange", fs=11, bold=False)
    x = 150
    for n in range(1, 7):
        d.box(x, 216, 110, 30, f"计算 {n}", "purple", fs=11, bold=False)
        if n < 6:
            d.box(x + 50, 176, 60, 30, f"调度{n + 1}", "orange", fs=11, bold=False)
            if n > 1:
                d.box(x, 176, 50, 30, f"处理{n - 1}", "blue", fs=11, bold=False)
        x += 110
    d.text(80, 280, "前提：下一轮的输入由上一轮的采样结果在 GPU 上写进 token pool；请求状态在发射时就推进（complete_one）。", fs=12,
           cls="dg-t2", anchor="start")
    d.text(80, 302, "代价：处理第 N 轮时 CPU 状态已领先一轮；在第 N 轮遇到 EOS 的请求会被多调度一轮。", fs=12, cls="dg-t2",
           anchor="start")
    return d


@diagram
def overlap_hazard() -> D:
    d = D("overlap-hazard", 980, 300)
    d.text(20, 24, "请求 X 在第 N 轮采样出 EOS", fs=13, anchor="start", bold=True)
    cols = [(120, "发射 N+1\n（含 X）", "orange"), (320, "处理 N：\nX 结束、释放", "blue"), (520, "调度 N+2：\n请求槽可以复用吗？", "orange"),
            (760, "处理 N+1：\nX 的结果是过期的", "blue")]
    for x, t, c in cols:
        d.box(x - 80, 46, 170, 56, t.split("\n")[0], c, t.split("\n")[1], fs=12)
    d.box(40, 132, 560, 34, "GPU：第 N+1 轮计算中（会往 X 那一行 token pool 写采样结果）", "purple", fs=12, bold=False)
    d.box(600, 132, 250, 34, "第 N+2 轮", "purple", fs=12, bold=False)
    d.arrow(605, 108, 520, 128, "red")
    d.text(470, 196, "官方：立即把 X 的请求槽分给新请求 → 调度器 stream 与引擎 stream 写同一行，没有先后保证", fs=12,
           cls="dg-tc-red")
    d.text(470, 220, "本书：推迟到\"处理 N+1\"（在途 batch 已算完）再归还请求槽；X 的过期结果整体丢弃", fs=12,
           cls="dg-tc-green")
    d.text(470, 256, "同一套逻辑也修正了：finished 提前一个 token、过期消息、prefill 在途时 abort 导致页重复释放", fs=12,
           cls="dg-t2")
    return d


# ====================================================================== 第四部分：更快、更大
@diagram
def tp_sharding() -> D:
    d = D("tp-sharding", 980, 400)
    y0 = 40
    d.text(20, 24, "一个 decoder 层在 TP=2 下（颜色 = rank）", fs=13, anchor="start", bold=True)
    d.box(20, y0 + 110, 70, 60, "x", "gray", "完整", fs=14)
    blocks = [(120, "qkv_proj", "列并行\n按头切", True), (300, "注意力", "各算各的头\n无通信", None),
              (480, "o_proj", "行并行\n部分和", False)]
    for x, t, sub, col in blocks:
        for r, c in enumerate(("blue", "green")):
            yy = y0 + 30 + r * 110
            d.box(x, yy, 140, 90, t, c, sub.split("\n")[0], fs=13)
            d.text(x + 70, yy + 72, sub.split("\n")[1], fs=11, cls="dg-t2")
    for x in (90, 260, 440):
        for r in range(2):
            d.arrow(x + (0 if x == 90 else 0), y0 + 140 if x == 90 else y0 + 75 + r * 110,
                    x + 28, y0 + 75 + r * 110, "gray")
    d.box(660, y0 + 90, 120, 80, "all-reduce", "red", "部分和相加", fs=13)
    d.arrow(620, y0 + 75, 658, y0 + 115, "gray")
    d.arrow(620, y0 + 185, 658, y0 + 150, "gray")
    d.box(820, y0 + 90, 140, 80, "MLP 同理", "gray", "gate_up 列 → down 行", fs=13)
    d.arrow(780, y0 + 130, 818, y0 + 130, "gray")
    d.text(890, y0 + 190, "再一次 all-reduce", fs=11, cls="dg-t2")
    d.text(20, 320, "每层 2 次 all-reduce；嵌入层按词表切后 all-reduce，LM head 按词表切后 all-gather。", fs=12, cls="dg-t2",
           anchor="start")
    d.text(20, 342, "KV 头少于 rank 数时，几个 rank 复制同一个 KV 头（Qwen3-0.6B：8 个 KV 头，TP=16 时每两个 rank 共用一个）。", fs=12,
           cls="dg-t2", anchor="start")
    d.text(20, 364, "合并的 qkv 必须先分别切 q、k、v 再拼接：每个 rank 拿到 [q_本地; k_本地; v_本地]。", fs=12, cls="dg-t2",
           anchor="start")
    return d


@diagram
def cuda_graph() -> D:
    d = D("cuda-graph", 980, 380)
    d.text(20, 24, "录制（每个批大小一次）", fs=13, anchor="start", bold=True)
    d.box(20, 40, 200, 70, "dummy batch", "gray", "bs 个 dummy 请求", fs=13)
    d.box(290, 40, 300, 70, "固定缓冲区", "teal", "input_ids · positions · out_loc · logits", fs=13)
    d.box(660, 40, 300, 70, "注意力后端的捕获缓冲区", "teal", "seq_lens · page_table（或 FlashInfer wrapper）", fs=13)
    d.arrow(222, 75, 288, 75, "gray", "指向")
    d.arrow(592, 75, 658, 75, "gray")
    d.box(290, 140, 670, 50, "torch.cuda.graph：录下 model.forward() 的所有 kernel（指针固定）", "purple", fs=13, bold=False)
    d.text(20, 230, "replay（每轮 decode）", fs=13, anchor="start", bold=True)
    d.box(20, 246, 200, 70, "当前 batch", "blue", "3 个请求 → 补齐到 4", fs=13)
    d.box(290, 246, 300, 70, "copy_from", "blue", "把输入拷进固定缓冲区", fs=13)
    d.box(660, 246, 300, 70, "prepare_for_replay", "blue", "把元数据拷进捕获缓冲区", fs=13)
    d.arrow(222, 281, 288, 281, "blue")
    d.arrow(592, 281, 658, 281, "blue")
    d.arrow(810, 244, 810, 192, "purple", "graph.replay()", lx=870, ly=222)
    d.text(20, 356, "漏拷任何一个输入，graph 读到的就是旧值，而且不会报错——CPU 仿真 EmulatedGraph 就是为了抓住这类问题。", fs=12,
           cls="dg-tc-red", anchor="start")
    return d


@diagram
def fused_moe() -> D:
    d = D("fused-moe", 980, 360)
    d.text(20, 24, "5 个 token，每个选 2 个专家 → 10 个 (token, 专家) 对", fs=13, anchor="start", bold=True)
    ids = [[4, 3], [6, 1], [1, 7], [6, 4], [5, 4]]
    cols = {1: "blue", 3: "green", 4: "orange", 5: "purple", 6: "pink", 7: "teal"}
    for t, pair in enumerate(ids):
        for k, e in enumerate(pair):
            i = t * 2 + k
            d.cell(20 + i * 44, 40, 44, 30, f"t{t}→{e}", cols[e], fs=10)
            d.text(20 + i * 44 + 22, 86, str(i), fs=10, cls="dg-t2", mono=True)
    d.text(20, 122, "按专家排序，每段补齐到 BLOCK_M = 4（· 是占位符）", fs=13, anchor="start", bold=True)
    blocks = [(1, [3, 4]), (3, [1]), (4, [0, 7, 9]), (5, [8]), (6, [2, 6]), (7, [5])]
    x = 20
    for e, members in blocks:
        for k in range(4):
            v = str(members[k]) if k < len(members) else "·"
            d.cell(x + k * 36, 138, 36, 30, v, cols[e] if k < len(members) else None, fs=11)
        d.text(x + 72, 186, f"专家 {e}", fs=11, cls=f"dg-tc-{cols[e]}")
        x += 4 * 36 + 10
    d.box(20, 210, 460, 56, "GEMM 1：每个 [BLOCK_M, BLOCK_N] 输出块只属于一个专家", "purple", "A 行 = hidden[对 // top_k]，B = w1[专家]", fs=12)
    d.box(500, 210, 150, 56, "silu × up", "gray", None, fs=12)
    d.box(670, 210, 290, 56, "GEMM 2（乘路由权重）", "purple", "B = w2[专家]", fs=12)
    d.arrow(480, 238, 498, 238, "gray")
    d.arrow(650, 238, 668, 238, "gray")
    d.text(20, 300, "整层 MoE 只需两次 kernel 启动；最后按 token 把 top_k 个结果相加。", fs=12, cls="dg-t2", anchor="start")
    d.text(20, 322, "块越大补齐浪费越多（decode 时尤其明显），块越小矩阵乘效率越低。", fs=12, cls="dg-t2", anchor="start")
    return d


def main(argv: List[str]) -> None:
    en = "--lang" in argv and argv[argv.index("--lang") + 1] == "en"
    missing_only = "--missing" in argv
    names = [a for i, a in enumerate(argv)
             if not a.startswith("--") and (i == 0 or argv[i - 1] != "--lang")]
    if en:
        tr = BOOK.parent / "i18n" / "en" / "figures.json"
        _TR["map"] = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else {}
        _TR["map"].setdefault("\u0000", "")          # 让 _t 在没有任何译文时也记录缺失
    out = (BOOK / "docs-en" / "assets" / "diagrams") if en else OUT
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in DIAGRAMS.items():
        if names and not any(a in name for a in names):
            continue
        svg = fn().svg()
        if missing_only:
            continue
        (out / f"{name}.svg").write_text(svg, encoding="utf-8")
        print("wrote", name + (" (en)" if en else ""))
    if en and _TR["missing"]:
        print(json.dumps({k: "" for k in sorted(_TR["missing"])}, ensure_ascii=False, indent=1))
        print(f"还有 {len(_TR['missing'])} 段文字没有英文译文", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
