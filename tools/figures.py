"""生成各手册里的示意图（SVG）。每张图是一个函数，画在一个小画布上，输出到对应手册的 docs/assets/figures/。

图里的文字、线条用 currentColor，形状的填充用 CSS 类（blue、green、orange、purple、red、gray、bx），
由 hooks/figures.py 内嵌进页面后，跟随站点的亮色 / 暗色主题。页面里这样引用：

    ![图：说明文字](../assets/figures/名字.svg){.aig-svg}

用法：python3 tools/figures.py          # 重新生成全部
      python3 tools/figures.py kv-cache # 只生成名字里含 kv-cache 的
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
FIGS: dict[tuple[str, str], callable] = {}


def figure(book: str, name: str):
    def deco(fn):
        FIGS[(book, name)] = fn                # 不同的书可以有同名的图（比如推理系统和 C++ 各有一张 paged-kv）
        return fn
    return deco


# 英文版（--lang en）：图里的每一段文字按 i18n/en/figures.json（中文原文 → 英文）替换，写到 <书>/docs-en/assets/figures/；
# 没有译文的原样保留，--missing 会列出来。中文版不受影响。
_TR: dict = {"map": {}, "missing": set()}


def _t(s):
    s = str(s)
    if not _TR["map"] or not re.search(r"[\u3400-\u9fff]", s):
        return s
    if s in _TR["map"]:
        return _TR["map"][s]
    _TR["missing"].add(s)
    return s


class Fig:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title, self.el = w, h, _t(title), []

    # ------------------------------------------------------------------ 基本图元
    def text(self, x, y, s, cls="tx", size=13, anchor="middle", weight=None, family=None):
        lines = _t(s).split("\n")
        attrs = f'x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}" dominant-baseline="middle"'
        if weight:
            attrs += f' font-weight="{weight}"'
        if family == "mono":
            attrs += ' font-family="ui-monospace, SFMono-Regular, Menlo, monospace"'
        if len(lines) == 1:
            self.el.append(f"<text {attrs}>{escape(lines[0])}</text>")
        else:
            dy0 = -(len(lines) - 1) * size * 0.62
            spans = "".join(f'<tspan x="{x:.1f}" dy="{dy0 if i == 0 else size * 1.24:.1f}">{escape(t)}</tspan>'
                            for i, t in enumerate(lines))
            self.el.append(f"<text {attrs}>{spans}</text>")

    def rect(self, x, y, w, h, cls="bx", rx=7, text=None, size=13, tcls="tx", weight=None, sw=1.3, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" stroke-width="{sw}"{d}/>')
        if text is not None:
            self.text(x + w / 2, y + h / 2, text, cls=tcls, size=size, weight=weight)

    def circle(self, x, y, r, cls="bx", text=None, size=12, tcls="tx", sw=1.3):
        self.el.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" class="{cls}" stroke-width="{sw}"/>')
        if text is not None:
            self.text(x, y, text, cls=tcls, size=size)

    def line(self, x1, y1, x2, y2, cls="ln", sw=1.4, dash=None, opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' stroke-opacity="{opacity}"' if opacity else ""
        self.el.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{sw}"{d}{o}/>')

    def path(self, d, cls="ln", sw=1.4, dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.el.append(f'<path d="{d}" class="{cls}" stroke-width="{sw}"{da}/>')

    def poly(self, pts, cls="ah"):
        self.el.append(f'<polygon points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" class="{cls}"/>')

    def head(self, x2, y2, ang, cls="ah", size=7):
        a1, a2 = ang + math.pi * 0.84, ang - math.pi * 0.84
        self.poly([(x2, y2), (x2 + size * math.cos(a1), y2 + size * math.sin(a1)),
                   (x2 + size * math.cos(a2), y2 + size * math.sin(a2))], cls)

    def arrow(self, x1, y1, x2, y2, cls="ln", hcls="ah", sw=1.4, dash=None, label=None, lx=0, ly=-8, lsize=11.5,
              both=False, opacity=None):
        ang = math.atan2(y2 - y1, x2 - x1)
        shorten = 5
        ex, ey = x2 - shorten * math.cos(ang), y2 - shorten * math.sin(ang)
        sx, sy = (x1 + shorten * math.cos(ang), y1 + shorten * math.sin(ang)) if both else (x1, y1)
        self.line(sx, sy, ex, ey, cls, sw, dash, opacity)
        self.head(x2, y2, ang, hcls)
        if both:
            self.head(x1, y1, ang + math.pi, hcls)
        if label:
            self.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, cls="mu", size=lsize)

    def elbow(self, pts, cls="ln", hcls="ah", sw=1.4, dash=None):
        """折线箭头：经过若干个点，最后一段带箭头"""
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        self.path(d, cls, sw, dash)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        self.head(x2, y2, math.atan2(y2 - y1, x2 - x1), hcls)

    def svg(self) -> str:
        body = "\n  ".join(self.el)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'role="img" aria-label="{escape(self.title)}">\n  {body}\n</svg>\n')


# ====================================================================== 大模型原理
@figure("llm", "decoder-block")
def decoder_block():
    f = Fig(560, 470, "一个 pre-norm 的 decoder 层")
    cx, w = 190, 200
    f.text(cx, 18, "残差流 x（每个 token 一个 d 维向量）", cls="mu", size=12)
    f.arrow(cx, 30, cx, 440, sw=2.2)
    f.text(cx, 458, "输出到下一层", cls="mu", size=12)
    # 注意力子层
    f.rect(330, 70, 190, 40, "gray", text="RMSNorm")
    f.rect(330, 132, 190, 56, "blue", text="注意力（GQA + RoPE）\n只在这里交换 token 间的信息")
    f.elbow([(cx, 60), (425, 60), (425, 70)])
    f.arrow(425, 110, 425, 132)
    f.elbow([(425, 188), (425, 206), (cx + 7, 206)])
    f.circle(cx, 206, 11, "bx", "+", size=15)
    # MLP 子层
    f.rect(330, 250, 190, 40, "gray", text="RMSNorm")
    f.rect(330, 312, 190, 56, "orange", text="SwiGLU MLP\n逐个 token 独立计算")
    f.elbow([(cx, 240), (425, 240), (425, 250)])
    f.arrow(425, 290, 425, 312)
    f.elbow([(425, 368), (425, 386), (cx + 7, 386)])
    f.circle(cx, 386, 11, "bx", "+", size=15)
    f.text(cx - 20, 206, "x = x + Attn(Norm(x))", cls="mu", size=12, anchor="end")
    f.text(cx - 20, 386, "x = x + MLP(Norm(x))", cls="mu", size=12, anchor="end")
    f.text(60, 120, "残差连接让梯度\n直接流回前面的层", cls="mu", size=11.5)
    return f


@figure("llm", "causal-attention")
def causal_attention():
    f = Fig(640, 300, "因果注意力：每个位置只看自己和之前的位置")
    n, s, x0, y0 = 6, 30, 70, 60
    toks = ["我", "爱", "北", "京", "天", "安"]
    f.text(x0 + n * s / 2, 30, "注意力分数 QKᵀ / √d（行：query，列：key）", size=12.5)
    for i in range(n):
        f.text(x0 - 16, y0 + i * s + s / 2, toks[i], cls="mu", size=12)
        f.text(x0 + i * s + s / 2, y0 - 12, toks[i], cls="mu", size=12)
        for j in range(n):
            cls = "blue" if j <= i else "gray"
            f.rect(x0 + j * s + 1, y0 + i * s + 1, s - 2, s - 2, cls, rx=3, sw=0.8)
            if j > i:
                f.text(x0 + j * s + s / 2, y0 + i * s + s / 2, "−∞", cls="mu", size=10)
    f.text(x0 + n * s / 2, y0 + n * s + 22, "上三角置为 −∞，softmax 之后为 0", cls="mu", size=11.5)
    # 右侧：softmax 和加权求和
    f.arrow(x0 + n * s + 20, y0 + 90, x0 + n * s + 70, y0 + 90, label="softmax\n（按行）", ly=-24)
    bx = x0 + n * s + 80
    f.rect(bx, y0 + 40, 130, 100, "green", text="每一行是一个\n概率分布\n（和为 1）")
    f.arrow(bx + 130, y0 + 90, bx + 180, y0 + 90, label="× V", ly=-12)
    f.rect(bx + 190, y0 + 40, 120, 100, "orange", text="输出：\n对 V 的加权平均\n（每个位置一行）")
    return f


@figure("llm", "gqa")
def gqa():
    f = Fig(660, 260, "MHA、GQA、MQA：多少个 query 头共用一组 K、V")
    groups = [("MHA", 8, 8, "每个头一组 KV\nKV 最大"), ("GQA", 8, 2, "4 个头共用一组\n（LLaMA-3、Qwen）"), ("MQA", 8, 1, "所有头共用一组\nKV 最小")]
    for g, (name, nq, nkv, note) in enumerate(groups):
        x0 = 20 + g * 215
        f.text(x0 + 90, 20, name, weight="600", size=14)
        for i in range(nq):
            f.rect(x0 + i * 23, 40, 18, 30, "blue", rx=3, sw=0.9)
        per = nq // nkv
        for k in range(nkv):
            kx = x0 + (k * per + per / 2) * 23 - 11
            f.rect(kx, 140, 18, 30, "orange", rx=3, sw=0.9)
            for i in range(per):
                qx = x0 + (k * per + i) * 23 + 9
                f.line(qx, 70, kx + 9, 140, sw=0.8, opacity=0.5)
        f.text(x0 + 90, 206, note, cls="mu", size=11.5)
    f.rect(214, 234, 14, 14, "blue", rx=2, sw=0.8)
    f.text(234, 241, "query 头", cls="mu", size=11.5, anchor="start")
    f.rect(330, 234, 14, 14, "orange", rx=2, sw=0.8)
    f.text(350, 241, "K、V 头（每个都要进 KV Cache）", cls="mu", size=11.5, anchor="start")
    return f


@figure("llm", "kv-cache")
def kv_cache():
    f = Fig(680, 290, "prefill 与 decode：KV Cache 让 decode 每步只算一个新 token")
    f.text(170, 20, "prefill：提示词的 T 个 token 一次算完", weight="600", size=13)
    for i in range(6):
        f.rect(40 + i * 44, 40, 38, 28, "blue", rx=4, text=f"t{i + 1}", size=11)
    f.arrow(170, 72, 170, 100)
    f.rect(40, 104, 260, 38, "gray", text="Q、K、V：T 行，矩阵乘（算力受限）", size=12)
    f.arrow(170, 146, 170, 172)
    for i in range(6):
        f.rect(40 + i * 44, 176, 38, 26, "orange", rx=4, text="K,V", size=10)
    f.text(170, 218, "全部 K、V 写入 KV Cache", cls="mu", size=11.5)
    f.text(510, 20, "decode：每步只有 1 个新 token", weight="600", size=13)
    f.rect(488, 40, 44, 28, "green", rx=4, text="t7", size=11)
    f.arrow(510, 72, 510, 100)
    f.rect(390, 104, 240, 38, "gray", text="q 只有 1 行：矩阵 × 向量（访存受限）", size=12)
    f.arrow(510, 146, 510, 172)
    for i in range(6):
        f.rect(360 + i * 40, 176, 34, 26, "orange", rx=4, text="K,V", size=10)
    f.rect(600, 176, 34, 26, "green", rx=4, text="新", size=10)
    f.text(495, 218, "读出全部历史 K、V，追加 1 行", cls="mu", size=11.5)
    f.line(340, 30, 340, 230, dash="4 4", opacity=0.4)
    f.text(340, 262, "每个 token 的 KV = 2 × 层数 × KV 头数 × 头维 × 字节数；随上下文和并发线性增长，决定能同时服务多少请求",
           cls="mu", size=11.5)
    return f


@figure("llm", "rope")
def rope():
    f = Fig(620, 260, "RoPE：把 q、k 的每一对维度按位置旋转")
    cx, cy, r = 140, 140, 90
    f.circle(cx, cy, r, "gray", sw=1)
    f.line(cx - r - 10, cy, cx + r + 10, cy, opacity=0.4)
    f.line(cx, cy - r - 10, cx, cy + r + 10, opacity=0.4)
    for k, (col, lbl) in enumerate([("blue-l", "位置 m：旋转 mθ"), ("orange-l", "位置 n：旋转 nθ")]):
        ang = -(0.45 + 0.75 * k)
        x2, y2 = cx + r * math.cos(ang), cy + r * math.sin(ang)
        f.arrow(cx, cy, x2, y2, cls=col, hcls=col.replace("-l", "-s"), sw=2)
        f.text(x2 + (18 if k == 0 else -8), y2 - 12, lbl, cls="mu", size=11.5, anchor="start" if k == 0 else "end")
    f.path(f"M {cx + 35 * math.cos(-0.45):.1f} {cy + 35 * math.sin(-0.45):.1f} A 35 35 0 0 0 {cx + 35 * math.cos(-1.2):.1f} {cy + 35 * math.sin(-1.2):.1f}", sw=1.2)
    f.text(cx + 52, cy - 44, "(n−m)θ", cls="mu", size=11)
    f.text(cx, 250, "一对维度 (x₂ᵢ, x₂ᵢ₊₁) 看成平面上的向量", cls="mu", size=11.5)
    tx = 290
    f.text(tx, 50, "q·k 只取决于两个旋转角之差：", anchor="start", size=13)
    f.text(tx, 78, "⟨R(mθ)q, R(nθ)k⟩ = ⟨q, R((n−m)θ)k⟩", anchor="start", size=13, family="mono")
    f.text(tx, 116, "→ 注意力分数只依赖相对位置 n − m", anchor="start", size=12.5, cls="mu")
    f.text(tx, 150, "不同的维度对转得快慢不同：", anchor="start", size=13)
    f.text(tx, 176, "θᵢ = base^(−2i/d)，高频对看近处，低频对看远处", anchor="start", size=12, cls="mu")
    f.text(tx, 202, "base 调大（10⁴ → 10⁶）能支持更长的上下文", anchor="start", size=12, cls="mu")
    return f


# ====================================================================== 推理系统
@figure("serving", "request-life")
def request_life():
    f = Fig(720, 230, "一个请求在推理引擎里的旅程")
    steps = [("HTTP / API", "gray"), ("分词、对话模板", "gray"), ("调度器\n排队 · 组批", "blue"),
             ("模型前向\nprefill / decode", "orange"), ("采样", "green"), ("反分词\n流式返回", "gray")]
    x, w, gap = 16, 100, 17
    for i, (s, c) in enumerate(steps):
        f.rect(x + i * (w + gap), 60, w, 58, c, text=s, size=12)
        if i:
            f.arrow(x + i * (w + gap) - gap, 89, x + i * (w + gap), 89)
    f.elbow([(x + 4 * (w + gap) + w / 2, 118), (x + 4 * (w + gap) + w / 2, 160), (x + 2 * (w + gap) + w / 2, 160),
             (x + 2 * (w + gap) + w / 2, 122)], dash="5 4")
    f.text(x + 3 * (w + gap) + w / 2, 176, "decode：每生成一个 token 就回到调度器，和别的请求一起组下一批", cls="mu", size=11.5)
    f.text(x + 0.5 * (w + gap) + w / 2, 36, "前端进程（CPU）", cls="mu", size=11.5)
    f.text(x + 3 * (w + gap) + w / 2, 36, "引擎核心（GPU 上的计算）", cls="mu", size=11.5)
    f.text(x + 5 * (w + gap) + w / 2, 36, "前端进程", cls="mu", size=11.5)
    f.text(360, 210, "TTFT = 排队 + prefill；TPOT = 每一轮 decode 的时间（被同批的 prefill 拉长）", cls="mu", size=11.5)
    return f


@figure("serving", "paged-kv")
def paged_kv():
    f = Fig(700, 280, "分页 KV：块表把请求的逻辑块映射到物理块")
    f.text(120, 20, "请求的逻辑块", weight="600")
    reqs = [("请求 A", [3, 7, 1], "blue"), ("请求 B", [5, 0], "orange"), ("请求 C", [6, 2, 4], "green")]
    for r, (name, blocks, c) in enumerate(reqs):
        y = 44 + r * 70
        f.text(30, y + 18, name, cls="mu", size=12)
        for i, pb in enumerate(blocks):
            f.rect(70 + i * 56, y, 50, 36, c, rx=5, text=f"块 {i}", size=11)
    f.text(330, 20, "块表", weight="600")
    for r, (name, blocks, c) in enumerate(reqs):
        y = 44 + r * 70
        f.rect(290, y, 84, 36, "gray", rx=5, text="[" + ", ".join(map(str, blocks)) + "]", size=12)
    f.text(560, 20, "GPU 上的物理块（每块 16 个 token）", weight="600")
    owner = {}
    for name, blocks, c in reqs:
        for pb in blocks:
            owner[pb] = c
    for pb in range(8):
        col, row = pb % 4, pb // 4
        x, y = 440 + col * 62, 50 + row * 70
        f.rect(x, y, 54, 50, owner.get(pb, "bx"), rx=5, text=str(pb), size=13)
    f.rect(440, 190, 240, 36, "gray", rx=5, text="空闲块链表：随用随取，用完归还", size=11.5)
    for r in range(3):
        f.arrow(234 if r != 1 else 178, 62 + r * 70, 288, 62 + r * 70, opacity=0.6)
        f.arrow(376, 62 + r * 70, 436, 62 + r * 70, opacity=0.6)
    f.text(350, 262, "逻辑上连续、物理上分散：不需要预留最大长度，碎片只出现在每个请求的最后一块", cls="mu", size=11.5)
    return f


@figure("serving", "continuous-batching")
def continuous_batching():
    f = Fig(700, 260, "静态批处理与连续批处理")
    lens = [6, 3, 8, 4]
    cols = ["blue", "orange", "green", "purple"]
    s = 22
    f.text(170, 18, "静态批处理：一批里最长的请求结束，整批才结束", size=12.5)
    for r, L in enumerate(lens):
        for t in range(8):
            cls = cols[r] if t < L else "gray"
            f.rect(30 + t * s, 36 + r * 26, s - 3, 20, cls, rx=3, sw=0.8)
    f.text(30 + 4 * s, 150, "灰色：已经结束、却还占着位置的请求", cls="mu", size=11.5)
    f.text(510, 18, "连续批处理：每一步都可以换人", size=12.5)
    queue = [(0, 6, 0), (1, 3, 0), (2, 8, 0), (3, 4, 0), (1, 4, 3), (3, 3, 4), (1, 2, 7)]
    extra = ["red", "blue", "orange"]
    for k, (row, L, start) in enumerate(queue):
        cls = cols[k] if k < 4 else extra[k - 4]
        for t in range(start, min(start + L, 10)):
            f.rect(380 + t * s, 36 + row * 26, s - 3, 20, cls, rx=3, sw=0.8)
    f.text(380 + 5 * s, 150, "一个请求结束，下一步就补上新的请求", cls="mu", size=11.5)
    f.arrow(30, 176, 30 + 8 * s, 176, label="时间（decode 步）", ly=12)
    f.arrow(380, 176, 380 + 10 * s, 176, label="时间（decode 步）", ly=12)
    f.text(350, 236, "GPU 始终满载：吞吐成倍提高；调度的单位从\"一批请求\"变成\"一步\"（iteration-level scheduling）", cls="mu", size=11.5)
    return f


@figure("serving", "radix-tree")
def radix_tree():
    f = Fig(660, 300, "基数树：共享前缀的请求复用同一段 KV")
    f.rect(260, 16, 140, 36, "gray", text="根", size=12)
    f.rect(230, 80, 200, 40, "blue", text="系统提示词（所有请求共享）", size=12)
    f.rect(30, 156, 170, 40, "orange", text="用户 A：第一轮提问", size=12)
    f.rect(250, 156, 160, 40, "green", text="用户 B：提问", size=12)
    f.rect(460, 156, 180, 40, "purple", text="few-shot 示例 + 问题", size=12)
    f.rect(30, 228, 170, 40, "orange", text="A 的回答 + 第二轮", size=11.5)
    f.rect(236, 228, 90, 40, "green", text="回答（采样 1）", size=11)
    f.rect(334, 228, 90, 40, "green", text="回答（采样 2）", size=11)
    f.arrow(330, 52, 330, 80)
    f.arrow(305, 120, 115, 156)
    f.arrow(330, 120, 330, 156)
    f.arrow(355, 120, 550, 156)
    f.arrow(115, 196, 115, 228)
    f.arrow(315, 196, 281, 228)
    f.arrow(345, 196, 379, 228)
    f.text(118, 284, "多轮对话沿一条路径增长", cls="mu", size=11)
    f.text(330, 284, "并行采样在这里分叉", cls="mu", size=11)
    f.text(550, 236, "命中前缀的部分不用再 prefill；\n空间不够时从叶子开始\n按 LRU 淘汰", cls="mu", size=11.5)
    return f


@figure("serving", "pd-disagg")
def pd_disagg():
    f = Fig(700, 250, "PD 分离：prefill 和 decode 在不同的 GPU 组上")
    f.rect(16, 90, 100, 60, "gray", text="路由 /\n调度器", size=12)
    f.rect(170, 40, 180, 70, "orange", text="Prefill 实例\n算力受限：大矩阵乘\n关心 TTFT", size=12)
    f.rect(510, 40, 180, 70, "green", text="Decode 实例\n访存受限：读权重和 KV\n关心 TPOT", size=12)
    f.arrow(116, 110, 170, 80)
    f.arrow(350, 75, 510, 75, label="KV Cache\n（RDMA / NVLink）", ly=-20)
    f.rect(170, 160, 520, 44, "blue", text="两边各自选择并行方式和实例数（xPyD）：prefill 和 decode 不再互相干扰", size=12)
    f.arrow(600, 110, 600, 160, dash="4 4", opacity=0.6)
    f.arrow(260, 110, 260, 160, dash="4 4", opacity=0.6)
    f.text(360, 232, "代价：KV 的传输（每 token 几十到几百 KB）、两套实例的容量规划", cls="mu", size=11.5)
    return f


@figure("serving", "tp-mlp")
def tp_mlp():
    f = Fig(700, 260, "张量并行的 MLP：先按列切、再按行切，每层只需一次 all-reduce")
    f.rect(20, 100, 70, 50, "gray", text="X\n（完整）", size=12)
    for r in range(2):
        y = 40 + r * 110
        f.rect(140, y, 130, 56, "blue", text=f"GPU {r}：W₁ 的\n第 {r} 组列", size=12)
        f.rect(330, y, 100, 56, "gray", text=f"H{r}\n（一半的列）", size=12)
        f.rect(480, y, 120, 56, "orange", text=f"GPU {r}：W₂ 的\n第 {r} 组行", size=12)
        f.arrow(90, 125, 140, y + 28)
        f.arrow(270, y + 28, 330, y + 28)
        f.arrow(430, y + 28, 480, y + 28)
        f.arrow(600, y + 28, 640, 125)
    f.circle(655, 125, 16, "red", "Σ", size=15)
    f.text(655, 160, "all-reduce", cls="mu", size=11.5)
    f.text(300, 236, "列并行的输出不用通信就能接行并行；两次矩阵乘之间的激活函数逐元素计算，也不需要通信", cls="mu", size=11.5)
    return f


@figure("serving", "ep-a2a")
def ep_a2a():
    f = Fig(700, 280, "专家并行：token 按路由结果发到专家所在的卡，算完再发回来")
    for r in range(4):
        y = 40 + r * 52
        f.rect(20, y, 110, 40, "blue", text=f"GPU {r} 的 token", size=11.5)
        f.rect(290, y, 120, 40, "orange", text=f"GPU {r}：专家 {2 * r}、{2 * r + 1}", size=11.5)
        f.rect(570, y, 110, 40, "green", text=f"GPU {r} 的输出", size=11.5)
    for a in range(4):
        for b in range(4):
            f.line(130, 60 + a * 52, 290, 60 + b * 52, sw=0.8, opacity=0.35)
            f.line(410, 60 + a * 52, 570, 60 + b * 52, sw=0.8, opacity=0.35)
    f.text(210, 26, "dispatch（all-to-all）", cls="mu", size=12)
    f.text(490, 26, "combine（all-to-all）", cls="mu", size=12)
    f.text(350, 258, "每个 token 只激活 top-k 个专家；通信量正比于 token 数 × k × 隐藏维度，DeepEP 等库专门优化这两次 all-to-all",
           cls="mu", size=11.5)
    return f


@figure("serving", "spec-decode")
def spec_decode():
    f = Fig(700, 290, "投机解码：草稿模型猜几个 token，目标模型一次验证")
    toks = ["北京", "是", "中国", "的"]
    f.text(90, 24, "草稿（小模型 / MTP 头）逐个生成 4 个 token", anchor="start", size=12.5)
    for i, t in enumerate(toks):
        f.rect(90 + i * 80, 40, 70, 34, "purple", text=t, size=12)
    f.arrow(245, 78, 245, 112)
    f.text(258, 95, "目标模型：一次前向，算出这 4 个位置（外加下一个位置）的分布", anchor="start", cls="mu", size=11.5)
    for i, t in enumerate(toks):
        cls = "green" if i < 3 else "red"
        f.rect(90 + i * 80, 116, 70, 34, cls, text=("✓ " if i < 3 else "✗ ") + t, size=12)
    f.arrow(365, 154, 365, 180)
    f.rect(330, 184, 70, 34, "blue", text="首都", size=12)
    f.text(420, 150, "第 4 个被拒绝：从修正后的分布\n重新采一个 token 代替它，\n后面的草稿全部作废", anchor="start", cls="mu", size=11.5)
    f.text(350, 244, "这一步前进了 4 个 token（3 个接受 + 1 个修正），目标模型只做了 1 次前向", size=12.5)
    f.text(350, 270, "按拒绝采样的规则接受，输出分布与只用目标模型完全相同；收益取决于接受率和验证的额外开销", cls="mu", size=11.5)
    return f


@figure("math", "roofline")
def roofline():
    f = Fig(640, 320, "屋顶线：decode 访存受限，prefill 算力受限（H100，bf16）")
    x0, y0, W, H = 70, 270, 520, 230
    f.line(x0, y0, x0 + W, y0, sw=1.2)
    f.line(x0, y0, x0, y0 - H, sw=1.2)
    f.text(x0 + W / 2, y0 + 30, "算术强度（FLOP / 字节，对数）", cls="mu", size=12)
    f.text(24, y0 - H / 2, "可达\n算力", cls="mu", size=12)

    def X(ai):
        return x0 + (math.log10(ai) - 0) / 4 * W        # 1 ~ 10^4

    def Y(t):
        return y0 - (math.log10(t) - 0) / 4 * H          # 1 ~ 10^4 TFLOPS

    peak, bw = 989, 3.35                                  # TFLOPS，TB/s
    ridge = peak / bw
    f.path(f"M {X(1):.1f} {Y(bw):.1f} L {X(ridge):.1f} {Y(peak):.1f} L {X(10 ** 4):.1f} {Y(peak):.1f}", cls="blue-l", sw=2.4)
    f.text(X(ridge) + 6, Y(peak) - 14, f"峰值 {peak} TFLOPS", cls="mu", size=11.5, anchor="start")
    f.text(X(6), Y(bw * 6) - 14, "斜率 = 带宽 3.35 TB/s", cls="mu", size=11.5)
    f.line(X(ridge), Y(peak), X(ridge), y0, dash="4 4", opacity=0.5)
    f.text(X(ridge), y0 + 12, f"拐点 ≈ {ridge:.0f}", cls="mu", size=11)
    for ai, lbl, cls in [(1, "decode\nbatch 1", "red-s"), (32, "decode\nbatch 32", "orange-s"), (2000, "prefill\n2K token", "green-s")]:
        t = min(peak, bw * ai)
        f.circle(X(ai), Y(t), 5, cls)
        f.text(X(ai) + (10 if ai < 1000 else -10), Y(t) + 22, lbl, size=11.5, anchor="start" if ai < 1000 else "end")
    return f


@figure("serving", "mla")
def mla():
    f = Fig(700, 260, "MLA：每个 token 只缓存一个低维的潜向量")
    f.rect(20, 100, 90, 50, "gray", text="h\n（隐藏状态）", size=12)
    f.rect(170, 60, 150, 56, "orange", text="c：512 维潜向量\n+ 64 维 RoPE 键", size=12)
    f.rect(170, 150, 150, 50, "blue", text="q（每个头）", size=12)
    f.arrow(110, 115, 170, 88, label="W_DKV", ly=-10)
    f.arrow(110, 135, 170, 175, label="W_Q", ly=12)
    f.rect(390, 40, 140, 44, "gray", text="KV Cache：只存 c", size=12)
    f.arrow(320, 80, 390, 62)
    f.rect(390, 120, 290, 84, "green", text="decode：把 W_UK 吸收进 q\nq·(W_UK c) = (W_UKᵀ q)·c\n直接和缓存的 c 做注意力，不必展开出每个头的 K", size=12)
    f.arrow(320, 175, 390, 162)
    f.text(350, 236, "DeepSeek-V3：每 token 每层 576 个数（MHA 要 2 × 128 头 × 128 维 = 32768 个）", cls="mu", size=11.5)
    return f


# ====================================================================== CUDA
@figure("cuda", "memory-hierarchy")
def memory_hierarchy():
    f = Fig(660, 300, "GPU 的存储层次（H100）")
    levels = [("寄存器", "每个 SM 256 KB", "每个线程私有，约 1 个周期", "green"),
              ("共享内存 / L1", "每个 SM 最多 228 KB", "同一个 block 共享，二三十个周期", "blue"),
              ("L2 缓存", "50 MB", "所有 SM 共享", "orange"),
              ("HBM（显存）", "80 GB，3.35 TB/s", "几百个周期", "purple")]
    cx = 250
    for i, (name, size, lat, cls) in enumerate(levels):
        w = 150 + i * 80
        y = 24 + i * 66
        f.rect(cx - w / 2, y, w, 52, cls, text=name, size=13, weight="600")
        f.text(480, y + 16, size, size=12, anchor="start")
        f.text(480, y + 36, lat, cls="mu", size=11.5, anchor="start")
    f.arrow(40, 280, 40, 24)
    f.text(54, 34, "越往上越快、\n越小", cls="mu", size=11.5, anchor="start")
    return f


@figure("cuda", "thread-hierarchy")
def thread_hierarchy():
    f = Fig(680, 260, "线程层次：grid → block → warp → thread")
    f.rect(20, 30, 250, 200, "gray", rx=10)
    f.text(145, 46, "grid（一次 kernel 启动）", size=12)
    for r in range(2):
        for c in range(3):
            f.rect(35 + c * 78, 64 + r * 80, 70, 66, "blue", text=f"block\n({c},{r})", size=11)
    f.arrow(270, 130, 330, 130)
    f.rect(330, 30, 180, 200, "blue", rx=10)
    f.text(420, 46, "一个 block（最多 1024 线程）", size=12)
    for w in range(4):
        f.rect(345, 64 + w * 40, 150, 32, "orange", text=f"warp {w}：32 个线程", size=11)
    f.arrow(510, 130, 560, 130)
    f.text(620, 96, "warp 里的 32 个线程\n同一时刻执行同一条指令\n（SIMT）", cls="mu", size=11.5)
    f.text(620, 170, "一个 block 只在\n一个 SM 上运行，\n可以用共享内存和\n__syncthreads()", cls="mu", size=11.5)
    return f


@figure("cuda", "coalescing")
def coalescing():
    f = Fig(680, 270, "合并访存：一个 warp 的线程访问连续地址，一次事务就能取回")
    # 合并：线程 i 读 a[i]
    f.text(30, 20, "合并：线程 i 读 a[i]", anchor="start", size=12.5, weight="600")
    f.rect(26, 86, 324, 28, "bx", rx=5, sw=0.9, dash="4 3")
    for i in range(16):
        f.rect(30 + i * 20, 34, 16, 20, "blue", rx=3, sw=0.8)
        f.rect(30 + i * 20, 90, 16, 20, "orange", rx=3, sw=0.8)
        f.arrow(38 + i * 20, 56, 38 + i * 20, 88, sw=0.9, opacity=0.55)
    f.text(380, 60, "相邻线程读相邻的地址：\nwarp 的请求合并成 1 次 128 字节的事务，\n取回的字节全部有用", anchor="start", cls="mu", size=11.5)
    # 跨步：线程 i 读 a[4i]
    f.text(30, 146, "跨步：线程 i 读 a[4·i]", anchor="start", size=12.5, weight="600")
    for seg in range(4):
        f.rect(28 + seg * 80, 212, 78, 28, "bx", rx=4, sw=0.9, dash="4 3")
    for i in range(8):
        f.rect(30 + i * 40, 160, 16, 20, "blue", rx=3, sw=0.8)
    for c in range(32):
        f.rect(30 + c * 10, 216, 8, 20, "orange" if c % 4 == 0 else "gray", rx=2, sw=0.6)
    for i in range(8):
        f.arrow(38 + i * 40, 182, 34 + i * 40, 214, sw=0.9, opacity=0.55)
    f.text(380, 196, "每个线程的数据隔得很开：\n要取回 4 倍的字节，其中 3/4 白读，\n有效带宽降到 1/4", anchor="start", cls="mu", size=11.5)
    f.text(340, 258, "虚线框是一次内存事务取回的范围：写 kernel 时让 threadIdx.x 连续的线程访问连续的地址", cls="mu", size=11.5)
    return f


@figure("cuda", "tiled-gemm")
def tiled_gemm():
    f = Fig(680, 290, "分块矩阵乘：每个 block 把 A、B 的小块搬进共享内存，反复使用")
    f.rect(40, 90, 160, 120, "gray", rx=3)
    f.text(120, 222, "A（M × K）", cls="mu", size=12)
    f.rect(40, 130, 160, 30, "blue", rx=2, sw=1.6)
    f.rect(230, 20, 120, 160, "gray", rx=3)
    f.text(290, 194, "B（K × N）", cls="mu", size=12)
    f.rect(290, 20, 30, 160, "orange", rx=2, sw=1.6)
    f.rect(380, 90, 120, 120, "gray", rx=3)
    f.text(440, 222, "C（M × N）", cls="mu", size=12)
    f.rect(440, 130, 30, 30, "green", rx=2, sw=1.8)
    f.text(560, 70, "沿 K 方向逐块推进：", size=12)
    f.text(560, 96, "① 搬 A、B 的一小块\n进共享内存", cls="mu", size=11.5)
    f.text(560, 146, "② 每个元素被复用\n tile 边长次", cls="mu", size=11.5)
    f.text(560, 196, "③ 累加到寄存器里的\nC 小块，最后写回", cls="mu", size=11.5)
    f.text(340, 268, "访存量从 O(MNK) 降到 O(MNK / tile)：算术强度随 tile 增大，才能从访存受限变成算力受限", cls="mu", size=11.5)
    return f


@figure("cuda", "flash-attention")
def flash_attention():
    f = Fig(700, 280, "FlashAttention：按块计算注意力，不写出 N × N 的分数矩阵")
    f.rect(30, 40, 60, 180, "blue", rx=4)
    f.text(60, 232, "Q", size=13)
    f.rect(30, 100, 60, 36, "blue", rx=3, sw=2)
    f.text(60, 118, "Qᵢ", size=12)
    f.rect(140, 40, 60, 180, "orange", rx=4)
    f.text(170, 232, "K、V", size=13)
    for j in range(4):
        f.rect(140, 44 + j * 44, 60, 40, "orange", rx=3, sw=1.2 if j != 1 else 2.2)
        f.text(170, 64 + j * 44, f"Kⱼ,Vⱼ", size=11)
    f.arrow(90, 118, 140, 108)
    f.rect(260, 70, 220, 110, "green", text="在共享内存 / 寄存器里：\nSᵢⱼ = Qᵢ Kⱼᵀ\n在线 softmax：更新行最大值 m\n和分母 l，把输出按比例修正\nOᵢ ← Oᵢ·e^(m_old−m) + P Vⱼ", size=11.5)
    f.arrow(200, 108, 260, 120)
    f.arrow(480, 125, 540, 125)
    f.rect(540, 100, 130, 50, "gray", text="只把 Oᵢ\n写回显存", size=12)
    f.text(350, 256, "显存读写从 O(N²) 降到 O(N²d²/M)（M 为片上存储大小）；计算量不变，但不再受显存带宽限制", cls="mu", size=11.5)
    return f


# ====================================================================== 分布式训练
@figure("train", "zero-stages")
def zero_stages():
    f = Fig(700, 270, "ZeRO：把优化器状态、梯度、参数依次切到各个数据并行 rank 上")
    rows = [("DDP", [1, 1, 1]), ("ZeRO-1", [1, 1, 0]), ("ZeRO-2", [1, 0, 0]), ("ZeRO-3 / FSDP", [0, 0, 0])]
    parts = [("参数", "blue", 2), ("梯度", "orange", 2), ("优化器状态", "green", 12)]
    f.text(430, 18, "每张卡上保存的内容（4 张卡，按字节数画宽度：2 + 2 + 12 字节 / 参数）", cls="mu", size=11.5)
    for r, (name, full) in enumerate(rows):
        y = 36 + r * 54
        f.text(80, y + 17, name, size=12.5, weight="600")
        x = 150
        for (lbl, cls, bytes_), keep in zip(parts, full):
            w = bytes_ * 24 if keep else bytes_ * 6
            f.rect(x, y, w, 34, cls, rx=4, text=lbl if w > 40 else None, size=11)
            x += w + 6
        mem = sum(b if k else b / 4 for (_, _, b), k in zip(parts, full))
        f.text(680, y + 17, f"{mem:g} 字节 / 参数", cls="mu", size=12, anchor="end")
    f.text(350, 258, "切得越多显存越省，通信越多：ZeRO-3 在前向、反向时都要 all-gather 参数", cls="mu", size=11.5)
    return f


def pipeline_times(order, p, dur):
    """按依赖关系排出流水线的时间表：F(s,i) 要等 F(s-1,i) 算完，B(s,i) 要等 B(s+1,i) 算完；每个 stage 按给定顺序串行执行"""
    done, t_stage, out = {}, [0] * p, [[] for _ in range(p)]
    pending = [list(o) for o in order]
    while any(pending):
        progressed = False
        for s in range(p):
            if not pending[s]:
                continue
            kind, i = pending[s][0]
            dep = (("F", s - 1, i) if s > 0 else None) if kind == "F" else (("B", s + 1, i) if s < p - 1 else ("F", s, i))
            if dep is not None and dep not in done:
                continue
            start = max(t_stage[s], done.get(dep, 0))
            end = start + dur[kind]
            done[(kind, s, i)] = end
            t_stage[s] = end
            out[s].append((kind, i, start, end))
            pending[s].pop(0)
            progressed = True
        assert progressed, "流水线的调度顺序死锁了"
    return out, max(t_stage)


@figure("train", "pipeline-1f1b")
def pipeline_1f1b():
    p, m, dur = 4, 4, {"F": 1, "B": 2}                     # 反向的耗时按前向的 2 倍画
    f = Fig(720, 290, "流水线并行：GPipe 与 1F1B（4 个 stage，4 个 micro-batch）")
    unit = 20
    gpipe = [[("F", i) for i in range(m)] + [("B", i) for i in range(m)] for _ in range(p)]
    ofob = []
    for s in range(p):
        warm = min(p - s - 1, m)
        seq = [("F", i) for i in range(warm)]
        f_next, b_next = warm, 0
        while b_next < m:
            if f_next < m:
                seq.append(("F", f_next))
                f_next += 1
            seq.append(("B", b_next))
            b_next += 1
        ofob.append(seq)

    def draw(y0, order, title):
        times, total = pipeline_times(order, p, dur)
        f.text(20, y0 - 14, title + f"：共 {total} 个时间单位", anchor="start", size=12.5)
        for s in range(p):
            f.text(14, y0 + s * 24 + 10, f"S{s}", cls="mu", size=11, anchor="start")
            f.rect(40, y0 + s * 24, total * unit, 20, "gray", rx=3, sw=0.5)
            for kind, i, a, b in times[s]:
                f.rect(40 + a * unit + 1, y0 + s * 24, (b - a) * unit - 2, 20, "blue" if kind == "F" else "orange",
                       rx=3, sw=0.8, text=f"{kind}{i}", size=10)
        return total

    draw(34, gpipe, "GPipe：先做完所有前向，再做所有反向，每个 stage 要存 4 份激活")
    draw(166, ofob, "1F1B：稳定阶段一个前向、一个反向交替，最多存 stage 数份激活")
    f.text(360, 278, "灰色是气泡（空闲）：两者的气泡一样多，约 (p−1)/(m+p−1)；1F1B 省的是激活显存，micro-batch 越多气泡越小",
           cls="mu", size=11.5)
    return f


@figure("train", "ring-allreduce")
def ring_allreduce():
    f = Fig(680, 270, "环形 all-reduce：reduce-scatter 之后接 all-gather")
    cx, cy, r, nr = 140, 132, 82, 27
    pos = [(cx + r * math.cos(-math.pi / 2 + k * math.pi / 2), cy + r * math.sin(-math.pi / 2 + k * math.pi / 2)) for k in range(4)]
    for k in range(4):
        (x1, y1), (x2, y2) = pos[k], pos[(k + 1) % 4]
        ang = math.atan2(y2 - y1, x2 - x1)
        f.arrow(x1 + nr * math.cos(ang), y1 + nr * math.sin(ang), x2 - nr * math.cos(ang), y2 - nr * math.sin(ang), sw=1.5)
    for k, (x, y) in enumerate(pos):
        f.circle(x, y, nr, "blue", f"GPU {k}", size=11)
    tx = 280
    f.text(tx, 36, "数据切成 N 份，沿环传 2(N−1) 步：", anchor="start", size=12.5)
    f.text(tx, 74, "① reduce-scatter（N−1 步）：每步把收到的一份累加进\n   自己的对应份，再传给下一张卡；结束时每张卡各有\n   一份完整的和", anchor="start", cls="mu", size=11.5)
    f.text(tx, 138, "② all-gather（N−1 步）：把各自那一份沿环传一圈，\n   每张卡都拿到全部的和", anchor="start", cls="mu", size=11.5)
    f.text(tx, 184, "每张卡收发的数据量 ≈ 2(N−1)/N × 数据大小：", anchor="start", size=12)
    f.text(tx, 206, "几乎与卡数无关，带宽用满，只有启动延迟随 N 增长", anchor="start", size=12)
    f.text(340, 252, "DDP 的梯度同步、张量并行每层的输出都靠它（或者它的树形、分层变体）", cls="mu", size=11.5)
    return f


@figure("train", "parallelism-3d")
def parallelism_3d():
    f = Fig(700, 280, "3D 并行：TP 放在机内，PP 跨机，DP 在最外层")
    for d in range(2):
        x0 = 20 + d * 345
        f.rect(x0, 30, 325, 200, "gray", rx=10)
        f.text(x0 + 162, 46, f"数据并行副本 {d}（DP 组之间只同步梯度）", size=12)
        for p in range(2):
            y0 = 64 + p * 82
            f.rect(x0 + 12, y0, 301, 70, "blue" if p == 0 else "orange", rx=8)
            f.text(x0 + 60, y0 + 35, f"stage {p}\n（一台机器）", size=11.5)
            for t in range(4):
                f.rect(x0 + 112 + t * 48, y0 + 14, 42, 42, "bx", rx=5, text=f"TP{t}", size=11)
        f.arrow(x0 + 162, 134, x0 + 162, 146, sw=1.2)
    f.text(350, 252, "TP 每层都要 all-reduce，需要 NVLink；PP 只在 stage 边界点对点发送，可以跨机器；DP 的梯度同步和反向重叠", cls="mu", size=11.5)
    return f


# ====================================================================== 数学（大模型原理）
@figure("math", "dot-product")
def dot_product():
    f = Fig(640, 300, "点积 = 投影长度 × 另一个向量的长度")
    ox, oy = 90, 240
    ax, ay = 300, -150            # 向量 a（屏幕坐标里 y 向上为负）
    bx, by = 330, -60             # 向量 b
    for gx in range(0, 7):
        f.line(ox + gx * 60, oy - 210, ox + gx * 60, oy + 20, cls="ln", sw=0.8, opacity=0.12)
    for gy in range(0, 5):
        f.line(ox - 30, oy - gy * 60, ox + 400, oy - gy * 60, cls="ln", sw=0.8, opacity=0.12)
    f.arrow(ox, oy, ox + ax, oy + ay, cls="blue-l", hcls="blue-s", sw=2.4)
    f.arrow(ox, oy, ox + bx, oy + by, cls="orange-l", hcls="orange-s", sw=2.4)
    f.text(ox + ax + 8, oy + ay - 6, "a", cls="blue-s", size=14, weight="700", anchor="start")
    f.text(ox + bx + 10, oy + by + 4, "b", cls="orange-s", size=14, weight="700", anchor="start")

    # a 在 b 上的投影
    bl2 = bx * bx + by * by
    t = (ax * bx + ay * by) / bl2
    px, py = ox + t * bx, oy + t * by
    f.line(ox + ax, oy + ay, px, py, cls="ln", sw=1.3, dash="4 4", opacity=0.6)
    f.path(f"M {ox:.1f} {oy:.1f} L {px:.1f} {py:.1f}", cls="purple-l", sw=5)
    f.text((ox + px) / 2 - 4, (oy + py) / 2 + 20, "投影长度 |a|cos θ", cls="purple-s", size=12)
    f.text(ox + 66, oy - 26, "θ", cls="mu", size=13)
    f.path(f"M {ox + 46:.1f} {oy - 8:.1f} A 48 48 0 0 0 {ox + 40:.1f} {oy - 26:.1f}", cls="ln", sw=1.2, dash="3 3")

    f.rect(430, 60, 190, 62, "purple", text="a · b = |a||b|cos θ", size=13)
    f.text(525, 150, "夹角越小，点积越大", cls="mu", size=12)
    f.text(525, 172, "垂直时点积为 0", cls="mu", size=12)
    f.text(525, 200, "注意力分数就是这个点积：", cls="mu", size=12)
    f.text(525, 222, "query 与 key 方向越一致，权重越高", cls="mu", size=12)
    return f


@figure("math", "matmul-views")
def matmul_views():
    f = Fig(720, 300, "矩阵乘法的三种视角：点积、列组合、外积之和")
    def grid(x, y, rows, cols, cell, cls, label, hi=None, hicls="blue"):
        for r in range(rows):
            for c in range(cols):
                c2 = hicls if hi and hi(r, c) else cls
                f.rect(x + c * cell, y + r * cell, cell, cell, c2, rx=2, sw=0.9)
        f.text(x + cols * cell / 2, y + rows * cell + 16, label, cls="mu", size=12)

    cell = 18
    # 1. 点积视角
    f.text(120, 34, "① 点积：一行 × 一列", size=13, weight="600")
    grid(46, 52, 4, 3, cell, "gray", "A", hi=lambda r, c: r == 1)
    f.text(116, 52 + 2 * cell, "×", cls="mu", size=14)
    grid(130, 52, 3, 5, cell, "gray", "B", hi=lambda r, c: c == 3, hicls="orange")
    f.text(232, 52 + 2 * cell, "=", cls="mu", size=14)
    grid(246, 52, 4, 5, cell, "bx", "C", hi=lambda r, c: r == 1 and c == 3, hicls="purple")

    # 2. 列组合视角
    f.text(120, 176, "② 列组合：C 的一列是 A 的列的线性组合", size=13, weight="600")
    grid(46, 194, 4, 3, cell, "blue", "A 的三列")
    f.text(116, 194 + 2 * cell, "×", cls="mu", size=14)
    grid(130, 194, 3, 5, cell, "gray", "B", hi=lambda r, c: c == 1, hicls="orange")
    f.text(232, 194 + 2 * cell, "=", cls="mu", size=14)
    grid(246, 194, 4, 5, cell, "bx", "C", hi=lambda r, c: c == 1, hicls="purple")

    # 3. 外积之和
    f.text(530, 34, "③ 外积之和：k 个秩 1 矩阵相加", size=13, weight="600")
    for t in range(3):
        x = 392 + t * 108
        grid(x, 58, 4, 1, cell, "blue", "")
        f.text(x + cell + 6, 58 + 2 * cell, "⊗", cls="mu", size=13)
        grid(x + cell + 14, 58, 1, 3, cell, "orange", "")
        f.text(x + 34, 58 + 4 * cell + 6, f"a{t + 1} ⊗ b{t + 1}", cls="mu", size=11.5)
        if t < 2:
            f.text(x + 96, 58 + 2 * cell, "+", cls="mu", size=15)
    f.text(556, 186, "切 m → 各算各的行（数据并行）", cls="mu", size=12)
    f.text(556, 208, "切 n → 列切分（张量并行）", cls="mu", size=12)
    f.text(556, 230, "切 k → 部分和要相加（行切分、split-K）", cls="mu", size=12)
    return f


@figure("math", "singular-values")
def singular_values():
    f = Fig(640, 340, "奇异值衰减：预训练权重不低秩，微调增量与 KV 低秩")
    x0, y0, W, H = 70, 260, 500, 210
    f.line(x0, y0, x0 + W, y0, sw=1.2)
    f.line(x0, y0, x0, y0 - H, sw=1.2)
    f.text(x0 + W / 2, y0 + 30, "第 i 个奇异值（按大小排序）", cls="mu", size=12)
    f.text(x0 - 6, y0 - H - 12, "σi / σ1", cls="mu", size=12, anchor="end")
    for frac, lbl in ((0.0, "0"), (0.5, "0.5"), (1.0, "1")):
        f.line(x0 - 4, y0 - frac * H, x0, y0 - frac * H, sw=1)
        f.text(x0 - 14, y0 - frac * H, lbl, cls="mu", size=11, anchor="end")

    def curve(fn, cls):
        pts = []
        for i in range(0, 101):
            x = x0 + i / 100 * W
            pts.append(f"{x:.1f} {y0 - fn(i / 100) * H:.1f}")
        f.path("M " + " L ".join(pts), cls=cls, sw=2.4)

    curve(lambda t: max(0.02, 1 - 0.75 * t), "blue-l")
    curve(lambda t: max(0.01, 2.718 ** (-9 * t)), "orange-l")
    curve(lambda t: max(0.01, 2.718 ** (-5.2 * t)), "purple-l")
    legend = [("blue-l", "预训练权重：衰减很慢，不是低秩"),
              ("purple-l", "KV 激活：能压成低维潜向量 → MLA"),
              ("orange-l", "微调增量 ΔW：几十个方向就够 → LoRA")]
    for i, (cls, lbl) in enumerate(legend):
        ly = y0 - H + 22 + i * 22
        f.line(x0 + W - 236, ly, x0 + W - 212, ly, cls=cls, sw=2.6)
        f.text(x0 + W - 204, ly, lbl, cls="mu", size=11.5, anchor="start")
    f.text(x0 + W / 2, y0 + 54, "保留前 r 个奇异值就是最优的秩 r 近似（Eckart-Young 定理）", cls="mu", size=11.5)
    return f


@figure("math", "entropy-kl")
def entropy_kl():
    f = Fig(680, 300, "熵与 KL 散度：分布有多分散，两个分布差多远")
    def bars(x0, y0, vals, cls, label, sub):
        w, gap, h = 26, 8, 120
        for i, v in enumerate(vals):
            f.rect(x0 + i * (w + gap), y0 - v * h, w, v * h, cls, rx=3, sw=1)
        f.line(x0 - 6, y0, x0 + len(vals) * (w + gap) - gap + 6, y0, sw=1.1)
        f.text(x0 + (len(vals) * (w + gap) - gap) / 2, y0 + 20, label, size=12.5, weight="600")
        f.text(x0 + (len(vals) * (w + gap) - gap) / 2, y0 + 40, sub, cls="mu", size=11.5)

    bars(60, 190, [0.62, 0.2, 0.1, 0.05, 0.03], "blue", "尖锐分布", "熵低：模型很有把握")
    bars(280, 190, [0.26, 0.22, 0.2, 0.18, 0.14], "orange", "平坦分布", "熵高：模型在犹豫")
    f.text(370, 246, "熵 H(p) = −Σ p log p", cls="mu", size=12)

    f.rect(470, 70, 180, 120, "purple", rx=10)
    f.text(560, 96, "KL(p‖q)", cls="purple-s", size=14, weight="700")
    f.text(560, 122, "= Σ p log(p / q)", cls="mu", size=12)
    f.text(560, 148, "用 q 代替 p 要多付", cls="mu", size=11.5)
    f.text(560, 166, "多少比特", cls="mu", size=11.5)
    f.text(560, 222, "投机解码的接受率、蒸馏的损失、", cls="mu", size=11.5)
    f.text(560, 242, "RL 的策略约束，都是这个量", cls="mu", size=11.5)
    return f


@figure("math", "backprop-graph")
def backprop_graph():
    f = Fig(700, 260, "反向模式自动微分：一次前向记录，一次反向按链式法则回传")
    xs = [70, 230, 390, 550]
    names = ["x", "h = Wx", "a = ReLU(h)", "L = loss(a)"]
    for i, (x, n) in enumerate(zip(xs, names)):
        f.rect(x, 60, 120, 48, "bx" if i else "gray", text=n, size=12.5)
        if i:
            f.arrow(xs[i - 1] + 120, 84, x, 84, sw=1.6)
    f.text(390, 36, "前向：算出每一步的值并留下需要的中间结果", cls="mu", size=12)

    for i in range(3, 0, -1):
        f.arrow(xs[i] + 10, 168, xs[i - 1] + 110, 168, cls="orange-l", hcls="orange-s", sw=1.8)
    for i, g in enumerate(["∂L/∂x", "∂L/∂h", "∂L/∂a", "∂L/∂L = 1"]):
        f.rect(xs[i], 144, 120, 46, "orange", text=g, size=12.5)
    f.text(390, 214, "反向：每一步把上游梯度乘上自己的局部导数（链式法则）", cls="mu", size=12)
    f.text(390, 236, "所以显存里要留住前向的中间结果——这就是激活显存的来源", cls="mu", size=11.5)
    return f


@figure("math", "queue-latency")
def queue_latency():
    f = Fig(640, 320, "利用率与排队延迟：越接近满载，延迟涨得越快")
    x0, y0, W, H = 70, 260, 500, 210
    f.line(x0, y0, x0 + W, y0, sw=1.2)
    f.line(x0, y0, x0, y0 - H, sw=1.2)
    f.text(x0 + W / 2, y0 + 30, "利用率 ρ（到达率 ÷ 处理能力）", cls="mu", size=12)
    f.text(x0 - 46, y0 - H - 12, "平均排队时间", cls="mu", size=12, anchor="start")
    for frac, lbl in ((0.0, "0"), (0.5, "50%"), (0.8, "80%"), (1.0, "100%")):
        f.line(x0 + frac * W, y0, x0 + frac * W, y0 + 5, sw=1)
        f.text(x0 + frac * W, y0 + 16, lbl, cls="mu", size=11)

    def curve(c, cls):
        pts = []
        for i in range(0, 96):
            rho = i / 100
            v = min(1.0, c * rho / (1 - rho) / 12)
            pts.append(f"{x0 + rho * W:.1f} {y0 - v * H:.1f}")
        f.path("M " + " L ".join(pts), cls=cls, sw=2.4)

    for i, (c, cls, lbl) in enumerate([(1.0, "red-l", "1 台机器"), (0.35, "orange-l", "4 台机器"),
                                       (0.12, "green-l", "16 台机器")]):
        curve(c, cls)
        ly = y0 - H + 22 + i * 22
        f.line(x0 + 30, ly, x0 + 54, ly, cls=cls, sw=2.6)
        f.text(x0 + 62, ly, lbl, cls="mu", size=11.5, anchor="start")
    f.line(x0 + 0.8 * W, y0, x0 + 0.8 * W, y0 - H, dash="4 4", opacity=0.45)
    f.text(x0 + 0.8 * W - 8, y0 - 150, "过了 80% 就开始起飞", cls="mu", size=11.5, anchor="end")
    f.text(x0 + 6, y0 - 16, "池子越大越抗：同样的利用率，机器多的排队短得多", cls="mu", size=11.5, anchor="start")
    return f


# ====================================================================== 运行
def main(argv: list[str]) -> None:
    """python3 tools/figures.py [名字片段 ...]                 生成中文图
       python3 tools/figures.py --lang en [--book B] [名字 ...]   生成英文图到 <书>/docs-en/assets/figures/
       加 --missing：只列出还没有英文译文的文字（写进 i18n/en/figures.json）"""
    en = "--lang" in argv and argv[argv.index("--lang") + 1] == "en"
    book_only = argv[argv.index("--book") + 1] if "--book" in argv else None
    missing_only = "--missing" in argv
    names = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--lang", "--book"))]
    if en:
        tr = ROOT / "i18n" / "en" / "figures.json"
        _TR["map"] = json.loads(tr.read_text(encoding="utf-8")) if tr.exists() else {}
        _TR["map"].setdefault("\u0000", "")                         # 让 _t 在没有任何译文时也记录缺失
    for (book, name), fn in FIGS.items():
        if names and not any(a in name for a in names):
            continue
        if book_only and book != book_only:
            continue
        fig = fn()
        if missing_only:
            continue
        out = ROOT / book / ("docs-en" if en else "docs") / "assets" / "figures" / f"{name}.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fig.svg(), encoding="utf-8")
        print(f"{book}/{name}.svg" + (" (en)" if en else ""))
    if en and _TR["missing"]:
        print(json.dumps({k: "" for k in sorted(_TR["missing"])}, ensure_ascii=False, indent=1))
        print(f"还有 {len(_TR['missing'])} 段文字没有英文译文", file=sys.stderr)



@figure("cuda", "reduction-addressing")
def reduction_addressing():
    f = Fig(640, 430, "归约的两种寻址方式：交错寻址会分支发散，顺序寻址不会")
    n, cw, gap, rh = 8, 60, 8, 48

    def board(y0, title, steps, note):
        x0 = 34
        f.text(x0, y0, title, cls="tx", size=13, weight="600", anchor="start")
        for r, (active, d) in enumerate(steps):
            y = y0 + 22 + r * rh
            f.text(x0 - 12, y + 12, f"{r + 1}", cls="mu", size=11, anchor="end")
            for i in range(n):
                x = x0 + i * (cw + gap)
                f.rect(x, y, cw, 24, "blue" if i in active else "gray", rx=4,
                       text=f"t{i}", size=11, tcls="tx" if i in active else "mu")
            for i in sorted(active):                          # 这一步谁把谁加过来
                xs = x0 + (i + d) * (cw + gap) + cw / 2
                xd = x0 + i * (cw + gap) + cw / 2
                yb = y + 24
                f.path(f"M {xs:.1f} {yb:.1f} C {xs:.1f} {yb + 15:.1f} {xd:.1f} {yb + 15:.1f} {xd:.1f} {yb:.1f}",
                       cls="ln", sw=1.1)
        f.text(x0, y0 + 22 + len(steps) * rh + 4, note, cls="mu", size=11.5, anchor="start")

    board(20, "① 交错寻址（v0）：if (tid % (2·s) == 0) —— 活跃线程在 warp 里是散的",
          [({0, 2, 4, 6}, 1), ({0, 4}, 2), ({0}, 4)],
          "同一个 warp 里一半线程闲着却要陪跑，这就是分支发散；tid % (2·s) 的取模本身也慢")
    board(232, "② 顺序寻址（v2）：if (tid < s) —— 活跃线程连成一段",
          [({0, 1, 2, 3}, 4), ({0, 1}, 2), ({0}, 1)],
          "整个 warp 要么全干、要么全闲，发散消失；共享内存访问也是连续的，没有 bank 冲突。\n"
          "剩下最后 32 个元素时换成 __shfl_down，连共享内存和 __syncthreads 都省了。")
    return f



# ====================================================================== 计算机基础
@figure("cs", "rtt-ladder")
def rtt_ladder():
    f = Fig(640, 370, "一个请求要花几个往返：新建连接 vs 复用连接")

    def ladder(x0, title, steps, total, note):
        cw = 200
        f.text(x0 + cw / 2, 22, title, cls="tx", size=13, weight="600")
        f.text(x0, 44, "客户端", cls="mu", size=11)
        f.text(x0 + cw, 44, "服务端", cls="mu", size=11)
        bottom = 72 + len(steps) * 56 - 12
        f.line(x0, 56, x0, bottom, cls="ln", sw=1.2, opacity=0.5)
        f.line(x0 + cw, 56, x0 + cw, bottom, cls="ln", sw=1.2, opacity=0.5)
        y = 72
        for label, cls in steps:
            f.arrow(x0 + 3, y, x0 + cw - 3, y + 16, cls=cls + "-l", hcls=cls + "-s", sw=1.8)
            f.arrow(x0 + cw - 3, y + 20, x0 + 3, y + 36, cls=cls + "-l", hcls=cls + "-s", sw=1.8)
            f.text(x0 + cw / 2, y - 6, label, cls="mu", size=11)
            y += 56
        f.rect(x0 - 4, 300, cw + 8, 30, "gray", rx=5, text=total, size=12)
        f.text(x0 + cw / 2, 350, note, cls="mu", size=11.5)

    ladder(70, "第一次请求：4 个往返",
           [("DNS 查询", "blue"), ("TCP 三次握手", "blue"), ("TLS 1.3 握手", "blue"), ("发请求 → 第一个字节", "orange")],
           "跨城 RTT 30 ms → 120 ms", "每一段都是一个完整的来回")
    ladder(380, "连接复用 + DNS 缓存：1 个往返",
           [("发请求 → 第一个字节", "orange")],
           "跨城 RTT 30 ms → 30 ms", "省掉的 90 ms 是白送的")
    return f


@figure("cs", "io-models")
def io_models():
    f = Fig(720, 320, "三种 I/O 模型：处理 1000 个就绪连接各要几次系统调用")
    rows = [
        ("一连接一线程", "gray", ["每个连接一个线程，阻塞在 recv 上", "1000 个线程的栈和上下文切换都要钱"], 1000, "1000 次"),
        ("epoll", "blue", ["一次 epoll_wait 拿回就绪列表", "每个就绪连接再各来一次 read", "对普通文件无效"], 1001, "1001 次"),
        ("io_uring", "green", ["请求批量写进和内核共享的提交队列", "一次 io_uring_enter 全部交出去", "结果直接从完成队列读；开 SQPOLL 连这一次都省"], 1, "1 次（SQPOLL 下 0 次）"),
    ]
    bx, bw = 420, 180
    f.text(bx, 26, "系统调用次数（对数刻度）", cls="mu", size=11.5, anchor="start")
    for r, (name, cls, bullets, calls, label) in enumerate(rows):
        y = 44 + r * 84
        f.rect(30, y, 118, 62, cls, rx=6, text=name, size=12.5, weight="600")
        for i, b in enumerate(bullets):
            f.text(164, y + 14 + i * 19, "· " + b, cls="mu", size=11.5, anchor="start")
        w = max(6, math.log10(calls + 1) / math.log10(1001) * bw)
        f.rect(bx, y + 18, w, 22, cls, rx=3)
        f.text(bx + w + 8, y + 29, label, cls="tx", size=11.5, anchor="start", weight="600")
    f.text(30, 300, "连接多、每次数据少的时候，系统调用本身就是主要开销——这正是 io_uring 想省掉的那一部分。",
           cls="mu", size=11.5, anchor="start")
    return f


@figure("train", "ring-attention")
def ring_attention():
    f = Fig(680, 310, "Ring Attention：query 不动，KV 沿环传一圈")
    P, cw, ch = 4, 120, 44
    x0, y0 = 40, 60
    f.text(x0, 30, "每张卡只存 1/P 的 query；KV 块每一步往右边传一格，P 步之后每段 query 都见过全部 KV",
           cls="mu", size=11.5, anchor="start")
    for r in range(P):
        y = y0 + r * (ch + 16)
        f.rect(x0, y, cw, ch, "blue", rx=5, text=f"卡 {r}\nquery 块 {r}", size=11.5)
        for step in range(P):
            x = x0 + cw + 40 + step * (cw - 10)
            kv = (r - step) % P
            cls = "orange" if step == 0 else "gray"
            f.rect(x, y + 8, cw - 26, ch - 16, cls, rx=4, text=f"KV 块 {kv}", size=11)
            if step < P - 1:
                f.arrow(x + cw - 26, y + ch / 2, x + cw - 10, y + ch / 2, sw=1.2)
    for step in range(P):
        f.text(x0 + cw + 40 + step * (cw - 10) + (cw - 26) / 2, y0 - 8, f"第 {step + 1} 步", cls="mu", size=11)
    f.text(x0, 286, "每一步算出的是「这段 query 对某一块 KV」的部分注意力，用 log-sum-exp 合并，\n"
                    "和 FlashAttention 的 online softmax 是同一个公式；网络上传的只有 KV。",
           cls="mu", size=11.5, anchor="start")
    return f



# ====================================================================== C++
@figure("cpp", "memory-order")
def memory_order():
    f = Fig(680, 330, "relaxed 与 release/acquire：另一个线程看到标志时，数据到了没有")

    def panel(x0, title, ok):
        cw, gap = 112, 56
        pw = cw * 2 + gap
        f.text(x0 + pw / 2, 24, title, cls="tx", size=12.5, weight="600")
        f.text(x0 + cw / 2, 52, "线程 A（写）", cls="mu", size=11)
        f.text(x0 + cw + gap + cw / 2, 52, "线程 B（读）", cls="mu", size=11)
        cls = "green" if ok else "red"
        f.rect(x0, 68, cw, 32, "blue", rx=5, text="写 config", size=11.5)
        f.rect(x0, 114, cw, 32, cls, rx=5, text="ready = true\n" + ("release" if ok else "relaxed"), size=10.5)
        f.rect(x0 + cw + gap, 68, cw, 32, cls, rx=5, text="读 ready\n" + ("acquire" if ok else "relaxed"), size=10.5)
        f.rect(x0 + cw + gap, 114, cw, 32, "blue", rx=5, text="读 config", size=11.5)
        f.arrow(x0 + cw / 2, 100, x0 + cw / 2, 114, sw=1.4)
        f.arrow(x0 + cw + gap + cw / 2, 100, x0 + cw + gap + cw / 2, 114, sw=1.4)
        if ok:
            f.arrow(x0 + cw + 4, 128, x0 + cw + gap - 4, 86, cls="green-l", hcls="green-s", sw=2)
            f.text(x0 + pw / 2, 172, "release 之前的所有写，\n对读到它的线程一定可见", cls="mu", size=11)
            f.rect(x0, 212, pw, 32, "green", rx=5, text="读到的 config 一定是新的", size=11.5)
        else:
            f.path(f"M {x0 - 10} 76 C {x0 - 34} 92 {x0 - 34} 122 {x0 - 10} 138", cls="red-l", sw=1.8, dash="4 3")
            f.text(x0 + pw / 2, 172, "两条写之间没有任何约束，\n编译器和 CPU 都可以换顺序", cls="mu", size=11)
            f.rect(x0, 212, pw, 32, "red", rx=5, text="可能先看到 ready，读到的还是旧 config", size=11)

    panel(52, "relaxed：有数据竞争", False)
    panel(388, "release / acquire：建立同步边", True)
    f.text(340, 274, "原子只保证「这一个变量」不被撕裂；跨变量的先后关系要靠内存序来建立。", cls="mu", size=11.5)
    f.text(340, 300, "acquire / release 只约束这一对变量，比 seq_cst 便宜，是发布-订阅场景的默认选择。", cls="mu", size=11.5)
    return f


# ====================================================================== Python
@figure("python", "name-binding")
def name_binding():
    f = Fig(680, 320, "名字是贴在对象上的标签，不是装值的盒子")
    f.text(30, 24, "可变对象：两个名字指向同一个列表", cls="tx", size=13, weight="600", anchor="start")
    f.rect(40, 44, 80, 30, "gray", rx=5, text="a", size=13)
    f.rect(40, 88, 80, 30, "gray", rx=5, text="b", size=13)
    f.rect(230, 56, 170, 50, "blue", rx=6, text="[1, 2, 3, 4]\nid = 0x7f…a0", size=11.5)
    f.arrow(120, 59, 230, 74, sw=1.5)
    f.arrow(120, 103, 230, 88, sw=1.5)
    f.text(430, 66, "b = a 不复制任何东西；", cls="mu", size=11.5, anchor="start")
    f.text(430, 86, "b.append(4) 之后 a 也变了。", cls="mu", size=11.5, anchor="start")

    f.text(30, 150, "不可变对象：给名字重新贴标签，不改动对象", cls="tx", size=13, weight="600", anchor="start")
    f.rect(40, 170, 80, 30, "gray", rx=5, text="x", size=13)
    f.rect(40, 214, 80, 30, "gray", rx=5, text="y", size=13)
    f.rect(230, 164, 110, 34, "orange", rx=6, text="1", size=13)
    f.rect(230, 210, 110, 34, "orange", rx=6, text="2", size=13)
    f.arrow(120, 185, 230, 181, sw=1.5)
    f.arrow(120, 229, 230, 227, sw=1.5)
    f.text(360, 181, "y = x 之后 y += 1 做的是", cls="mu", size=11.5, anchor="start")
    f.text(360, 201, "「算出新对象 2，再让 y 指向它」，", cls="mu", size=11.5, anchor="start")
    f.text(360, 221, "对象 1 一点没动，所以 x 还是 1。", cls="mu", size=11.5, anchor="start")

    f.rect(30, 268, 620, 36, "gray", rx=6,
           text="赋值 = 让名字指向对象；== 比的是值，is 比的是「同一个对象」；函数传参传的也是这条标签",
           size=12)
    return f


@figure("python", "gil-timeline")
def gil_timeline():
    f = Fig(680, 330, "GIL：CPU 密集型任务多线程不会更快，I/O 密集型会")
    x0, W, unit = 120, 480, 77

    def track(y, title, rows, note):
        f.text(30, y - 16, title, cls="tx", size=12.5, weight="600", anchor="start")
        for ti, (name, blocks) in enumerate(rows):
            yy = y + ti * 26
            f.text(x0 - 10, yy + 10, name, cls="mu", size=11, anchor="end")
            f.rect(x0, yy, W, 20, "gray", rx=3, sw=0)
            for (s0, ln, kind) in blocks:
                cls = {"run": "blue", "wait": "orange", "idle": "gray"}[kind]
                f.rect(x0 + s0 * unit, yy, ln * unit, 20, cls, rx=3,
                       text=("跑" if kind == "run" else "等 I/O" if kind == "wait" else ""), size=10.5)
        f.text(x0, y + len(rows) * 26 + 12, note, cls="mu", size=11.5, anchor="start")

    track(46, "CPU 密集：两个线程轮流拿 GIL，总时间和单线程一样",
          [("线程 1", [(0, 1, "run"), (2, 1, "run"), (4, 1, "run")]),
           ("线程 2", [(1, 1, "run"), (3, 1, "run"), (5, 1, "run")])],
          "字节码同一时刻只有一个线程在跑，切换还要额外开销")
    track(146, "I/O 密集：等待时释放 GIL，两个线程真的重叠了",
          [("线程 1", [(0, 0.5, "run"), (0.5, 3, "wait"), (3.5, 0.5, "run")]),
           ("线程 2", [(0.5, 0.5, "run"), (1, 3, "wait"), (4, 0.5, "run")])],
          "socket.recv、time.sleep、读文件，以及 NumPy 的大块计算，都会释放 GIL")
    track(246, "多进程：每个进程一把自己的 GIL，CPU 密集才真正并行",
          [("进程 1", [(0, 3, "run")]), ("进程 2", [(0, 3, "run")])],
          "同样的计算量，墙钟时间只要一半；代价是进程间要序列化传数据，启动也更慢")
    f.text(x0 + W, 322, "时间 →", cls="mu", size=11, anchor="end")
    return f


@figure("python", "generator-pipeline")
def generator_pipeline():
    f = Fig(680, 312, "生成器：一次只在内存里留一条数据")
    f.text(30, 24, "① 先全读进列表：内存里同时存着整份数据", cls="tx", size=13, weight="600", anchor="start")
    f.rect(40, 40, 180, 40, "orange", rx=6, text="读全部行 → list", size=12)
    f.rect(240, 40, 150, 40, "orange", rx=6, text="过滤 → 新 list", size=12)
    f.rect(410, 40, 150, 40, "orange", rx=6, text="解析 → 新 list", size=12)
    f.arrow(220, 60, 240, 60, sw=1.5)
    f.arrow(390, 60, 410, 60, sw=1.5)
    f.text(660, 60, "峰值内存\n≈ 整份数据", cls="mu", size=11, anchor="end")

    f.text(30, 122, "② 生成器管道：每次只拉一条，用完就扔", cls="tx", size=13, weight="600", anchor="start")
    f.rect(40, 138, 180, 40, "blue", rx=6, text="逐行 yield", size=12)
    f.rect(240, 138, 150, 40, "blue", rx=6, text="过滤 yield", size=12)
    f.rect(410, 138, 150, 40, "blue", rx=6, text="解析 yield", size=12)
    f.arrow(240, 158, 220, 158, cls="blue-l", hcls="blue-s", sw=1.6, label="要一条", ly=-10)
    f.arrow(410, 158, 390, 158, cls="blue-l", hcls="blue-s", sw=1.6, label="要一条", ly=-10)
    f.text(660, 158, "峰值内存\n≈ 一条", cls="mu", size=11, anchor="end")

    f.text(30, 214, "生成器函数的状态", cls="tx", size=13, weight="600", anchor="start")
    states = [("调用函数", "gray"), ("暂停在 yield", "blue"), ("next() 恢复", "blue"), ("return / 耗尽", "gray")]
    for i, (name, cls) in enumerate(states):
        x = 40 + i * 155
        f.rect(x, 232, 128, 34, cls, rx=6, text=name, size=11.5)
        if i < len(states) - 1:
            f.arrow(x + 128, 249, x + 155, 249, sw=1.4)
    f.path("M 168 266 C 168 282 355 282 355 266", cls="ln", sw=1.2, dash="4 3")
    f.text(262, 296, "每次 next() 从上次暂停的地方继续，局部变量都还在", cls="mu", size=11)
    return f


@figure("python", "event-loop")
def event_loop():
    f = Fig(680, 320, "事件循环：一个线程上怎么同时跑三个任务")
    x0, W, unit = 110, 470, 47
    tasks = [
        ("任务 A", [(0, 1, "run"), (1, 4, "wait"), (5, 1, "run")]),
        ("任务 B", [(1, 1, "run"), (2, 3, "wait"), (6, 1, "run")]),
        ("任务 C", [(2, 1, "run"), (3, 5, "wait"), (8, 1, "run")]),
    ]
    f.text(30, 24, "同一个线程，谁在 await 就让出来给别人跑", cls="mu", size=11.5, anchor="start")
    for i, (name, blocks) in enumerate(tasks):
        y = 44 + i * 32
        f.text(x0 - 10, y + 11, name, cls="mu", size=11, anchor="end")
        f.rect(x0, y, W, 22, "gray", rx=3, sw=0)
        for (s0, ln, kind) in blocks:
            f.rect(x0 + s0 * unit, y, ln * unit, 22, "blue" if kind == "run" else "orange", rx=3,
                   text=("跑代码" if kind == "run" else "await：把控制权交回去"), size=10.5)
    f.text(x0, 152, "整段时间里 CPU 其实只忙了最上面那几小块——剩下的都在等网络。",
           cls="mu", size=11.5, anchor="start")

    f.text(30, 192, "循环本身在做的事", cls="tx", size=13, weight="600", anchor="start")
    steps = [("就绪队列取一个", "blue"), ("跑到下一个 await", "blue"), ("登记到 selector", "orange"),
             ("epoll_wait 等就绪", "orange"), ("回调放回就绪队列", "green")]
    for i, (name, cls) in enumerate(steps):
        x = 30 + i * 128
        f.rect(x, 212, 112, 40, cls, rx=6, text=name, size=11)
        if i < len(steps) - 1:
            f.arrow(x + 112, 232, x + 128, 232, sw=1.4)
    f.path("M 86 252 C 86 274 600 274 600 252", cls="ln", sw=1.2, dash="4 3")
    f.text(343, 284, "一圈又一圈；单线程，没有锁，也没有线程切换", cls="mu", size=11)
    f.text(343, 308, "任何一个协程里写出阻塞调用（requests.get、time.sleep），整个循环就卡住",
           cls="mu", size=11.5)
    return f


@figure("llm", "train-vs-infer")
def train_vs_infer():
    f = Fig(700, 300, "训练一次前向算出所有位置的预测（teacher forcing）；推理只能一个一个地生成")
    toks = ["北京", "是中国", "的", "首都", "，"]
    # 左：训练
    f.text(30, 22, "训练：整句已知，一次前向，所有位置并行", cls="tx", size=13, weight="600", anchor="start")
    x0, y_in, y_out, w = 40, 70, 160, 58
    for i, t in enumerate(toks):
        x = x0 + i * 64
        f.rect(x, y_in, w, 30, "blue", rx=5, text=t, size=12)
        f.rect(x, y_out, w, 30, "green", rx=5, text="预测 " + (toks[i + 1] if i + 1 < len(toks) else "…"), size=10.5)
        f.arrow(x + w / 2, y_in + 30, x + w / 2, y_out, sw=1.2)
    f.rect(x0 - 8, y_in - 8, 64 * 5 + 10, 30 + 16 + 60 + 30 - 60, "gray", rx=8, sw=1, dash="4 3")
    f.text(x0 + 160, 128, "一次前向：第 t 个位置只看得到前 t 个 token（因果掩码）", cls="mu", size=10.5)
    f.text(x0 + 160, 215, "损失 = 5 个位置的 −log P(真实的下一个 token) 取平均", cls="mu", size=11)
    f.text(x0 + 160, 240, "算力随序列长度线性增长，但只过一遍权重", cls="mu", size=11)
    # 右：推理
    f.text(390, 22, "推理：下一个 token 要等上一个生成出来", cls="tx", size=13, weight="600", anchor="start")
    steps = [["北京", "是中国"], ["北京", "是中国", "的"], ["北京", "是中国", "的", "首都"]]
    for r, seq in enumerate(steps):
        y = 56 + r * 62
        f.text(392, y + 15, f"第 {r + 1} 步", cls="mu", size=10.5, anchor="start")
        for i, t in enumerate(seq):
            x = 440 + i * 50
            new = i == len(seq) - 1
            f.rect(x, y, 46, 30, "orange" if new else "blue", rx=5, text=t, size=11)
        f.arrow(440 + len(seq) * 50 + 4, y + 15, 440 + len(seq) * 50 + 26, y + 15, sw=1.2)
        f.text(440 + len(seq) * 50 + 46, y + 15, "前向", cls="mu", size=10.5)
        if r < len(steps) - 1:
            f.path(f"M {440 + len(seq) * 50 + 46} {y + 24} C {440 + len(seq) * 50 + 46} {y + 50} {440 + len(seq) * 50 + 28} {y + 50} {440 + len(seq) * 50 + 26} {y + 62}", cls="ln", sw=1.1, dash="3 3")
    f.text(540, 250, "每一步都要把全部权重读一遍，只算一个 token：", cls="mu", size=11)
    f.text(540, 272, "500 个 token = 500 次前向，这就是 decode 的访存瓶颈", cls="mu", size=11)
    return f


@figure("math", "float-numberline")
def float_numberline():
    f = Fig(700, 262, "三种 8 位格式在 0～16 之间能表示的数：浮点近 0 处密、远处疏，整数处处均匀")
    x0, x1 = 150, 680

    def X(v):
        return x0 + v / 16 * (x1 - x0)

    def fp_values(ebits, mbits, bias, vmax):
        vals = set()
        for e in range(0, 2 ** ebits):
            for m in range(0, 2 ** mbits):
                if e == 0:
                    v = m / 2 ** mbits * 2 ** (1 - bias)
                else:
                    v = (1 + m / 2 ** mbits) * 2 ** (e - bias)
                if 0 <= v <= vmax:
                    vals.add(v)
        return sorted(vals)

    rows = [
        ("FP8 E4M3", fp_values(4, 3, 7, 16), "blue", "1 位符号 + 4 位指数 + 3 位尾数：[1, 2) 里 8 个数，[8, 16) 里也是 8 个"),
        ("FP8 E5M2", fp_values(5, 2, 15, 16), "purple", "1 + 5 + 2：每个二进制区间只有 4 个数，范围更大、更稀"),
        ("INT8 × (16/127)", [i * 16 / 127 for i in range(0, 128)], "orange", "整数格式配一个缩放因子：间隔处处相同（0.126）"),
    ]
    for r, (name, vals, cls, note) in enumerate(rows):
        y = 50 + r * 64
        f.text(x0 - 8, y, name, cls="tx", size=12, anchor="end", weight="600")
        f.line(x0, y, x1, y, cls="ln", sw=1, opacity=0.5)
        for v in vals:
            f.line(X(v), y - 9, X(v), y + 9, cls=cls + "-l", sw=1.1)
        f.text(x0, y + 24, note, cls="mu", size=10.5, anchor="start")
    y = 50 + 3 * 64 - 24
    for v in (0, 1, 2, 4, 8, 16):
        f.line(X(v), y - 4, X(v), y + 4, cls="ln", sw=1)
        f.text(X(v), y + 14, str(v), cls="mu", size=10.5)
    f.line(x0, y, x1, y, cls="ln", sw=1)
    f.text(415, 252, "浮点：相对精度固定（E4M3 约 6%），所以能容忍离群值；整数：绝对精度固定，大数小数一视同仁，缩放因子必须选好",
           cls="mu", size=10.5)
    return f


@figure("llm", "multihead")
def multihead():
    f = Fig(700, 300, "多头注意力：把 d 维拆成 n_h 个头，各自做注意力，拼回去再过输出投影")
    f.rect(20, 120, 90, 60, "gray", rx=8, text="x\n[T, d]", size=12)
    f.arrow(110, 150, 150, 150, sw=1.4)
    for i, (name, cls) in enumerate([("W_Q", "blue"), ("W_K", "green"), ("W_V", "orange")]):
        y = 60 + i * 66
        f.rect(150, y, 70, 40, cls, rx=6, text=name, size=12)
        f.line(150, 150, 150, y + 20, cls="ln", sw=1.2)
        f.arrow(220, y + 20, 262, y + 20, sw=1.2)
    f.text(240, 36, "拆成 n_h 份", cls="mu", size=11)
    heads = 4
    for h in range(heads):
        x = 268 + h * 78
        f.rect(x, 56, 66, 170, "gray", rx=8, sw=1, dash="4 3")
        f.text(x + 33, 70, f"头 {h + 1}", cls="tx", size=11, weight="600")
        f.rect(x + 8, 84, 50, 24, "blue", rx=4, text="q·kᵀ", size=10.5)
        f.arrow(x + 33, 108, x + 33, 126, sw=1.1)
        f.rect(x + 8, 126, 50, 24, "purple", rx=4, text="softmax", size=10)
        f.arrow(x + 33, 150, x + 33, 168, sw=1.1)
        f.rect(x + 8, 168, 50, 24, "orange", rx=4, text="× v", size=10.5)
        f.text(x + 33, 210, "[T, d_h]", cls="mu", size=10)
    f.text(425, 246, "每个头 d_h = d / n_h 维，各自关注不同的关系（前一个词、主语、重复的内容……）", cls="mu", size=11)
    f.arrow(580, 150, 612, 150, sw=1.4)
    f.rect(612, 120, 66, 60, "blue", rx=8, text="拼接\n→ W_O", size=11.5)
    f.text(645, 200, "[T, d]", cls="mu", size=10.5)
    f.text(350, 280, "n_h 个头的 QKᵀ 是一次 batch 矩阵乘；GQA 让几个 query 头共用一组 K、V", cls="mu", size=11)
    return f


@figure("llm", "pre-post-norm")
def pre_post_norm():
    f = Fig(700, 270, "Post-Norm 把归一化放在相加之后；Pre-Norm 让残差流成为一条不被归一化的加法通道")

    def column(x0, title, pre):
        f.text(x0 + 130, 24, title, cls="tx", size=13, weight="600")
        f.rect(x0 + 90, 40, 80, 28, "gray", rx=6, text="x", size=12)
        if pre:
            f.arrow(x0 + 130, 68, x0 + 130, 88, sw=1.3)
            f.rect(x0 + 90, 88, 80, 28, "purple", rx=6, text="Norm", size=11.5)
            f.arrow(x0 + 130, 116, x0 + 130, 136, sw=1.3)
            f.rect(x0 + 70, 136, 120, 30, "blue", rx=6, text="注意力 / FFN", size=11.5)
            f.arrow(x0 + 130, 166, x0 + 130, 190, sw=1.3)
            f.circle(x0 + 130, 202, 12, "green", text="+", size=14)
            f.path(f"M {x0 + 170} 54 C {x0 + 240} 54 {x0 + 240} 202 {x0 + 142} 202", cls="green-l", sw=2.2)
            f.text(x0 + 235, 128, "残差流\n（原样加回）", cls="mu", size=10.5)
            f.arrow(x0 + 130, 214, x0 + 130, 240, sw=1.3)
            f.text(x0 + 130, 252, "x + Sublayer(Norm(x))", cls="mu", size=11)
        else:
            f.arrow(x0 + 130, 68, x0 + 130, 92, sw=1.3)
            f.rect(x0 + 70, 92, 120, 30, "blue", rx=6, text="注意力 / FFN", size=11.5)
            f.arrow(x0 + 130, 122, x0 + 130, 146, sw=1.3)
            f.circle(x0 + 130, 158, 12, "green", text="+", size=14)
            f.path(f"M {x0 + 170} 54 C {x0 + 240} 54 {x0 + 240} 158 {x0 + 142} 158", cls="green-l", sw=2.2)
            f.arrow(x0 + 130, 170, x0 + 130, 190, sw=1.3)
            f.rect(x0 + 90, 190, 80, 28, "purple", rx=6, text="Norm", size=11.5)
            f.arrow(x0 + 130, 218, x0 + 130, 240, sw=1.3)
            f.text(x0 + 130, 252, "Norm(x + Sublayer(x))", cls="mu", size=11)

    column(30, "Post-Norm（原始 Transformer）", False)
    column(380, "Pre-Norm（现代大模型）", True)
    f.line(350, 30, 350, 250, cls="ln", sw=1, dash="4 4", opacity=0.4)
    return f


@figure("llm", "moe-structure")
def moe_structure():
    f = Fig(700, 300, "MoE 层：路由器给每个 token 打分，只把它送进分数最高的 k 个专家，再按权重求和")
    f.rect(20, 125, 70, 50, "gray", rx=8, text="token\nx", size=12)
    f.arrow(90, 150, 130, 150, sw=1.4)
    f.rect(130, 110, 90, 80, "purple", rx=8, text="路由器\nd → E\nsoftmax", size=11.5)
    experts = 8
    chosen = {2: "0.62", 5: "0.38"}
    for i in range(experts):
        y = 28 + i * 31
        cls = "orange" if i in chosen else "gray"
        f.rect(330, y, 110, 24, cls, rx=5, text=f"专家 {i + 1}（小 FFN）", size=10.5, sw=1.6 if i in chosen else 1)
        if i in chosen:
            f.arrow(220, 150, 330, y + 12, cls="orange-l", hcls="orange-s", sw=1.6, label=chosen[i], lx=-22, ly=-6 if i == 2 else 12)
            f.arrow(440, y + 12, 500, 150, cls="orange-l", hcls="orange-s", sw=1.6)
        else:
            f.line(220, 150, 330, y + 12, cls="ln", sw=0.8, opacity=0.18)
    f.rect(300, 272, 170, 22, "green", rx=5, text="共享专家（所有 token 都过）", size=10.5)
    f.arrow(130, 150, 300, 283, cls="green-l", hcls="green-s", sw=1.2)
    f.arrow(470, 283, 500, 160, cls="green-l", hcls="green-s", sw=1.2)
    f.circle(512, 150, 14, "blue", text="Σ", size=14)
    f.arrow(526, 150, 570, 150, sw=1.4)
    f.rect(570, 125, 110, 50, "gray", rx=8, text="y = Σ wᵢ·Expertᵢ(x)", size=10.5)
    f.text(250, 232, "top-2：只有 2 个专家真的算\n其余 6 个这一步不动", cls="mu", size=10.5)
    f.text(600, 215, "激活参数 ≈ k 个专家 + 共享专家\n总参数 = 全部 E 个专家", cls="mu", size=10.5)
    return f


@figure("llm", "residual-stream")
def residual_stream():
    f = Fig(700, 290, "残差流：一条贯穿所有层的 d 维向量，每层从中读、算、再加回去；logit lens 在中途把它解码成词")
    y = 150
    f.rect(20, y - 16, 70, 32, "gray", rx=6, text="嵌入", size=11.5)
    f.rect(90, y - 10, 500, 20, "green", rx=10, sw=1.2)
    f.text(340, y, "残差流 x  [T, d]", cls="tx", size=11.5, weight="600")
    f.rect(610, y - 16, 70, 32, "gray", rx=6, text="Norm\nLM Head", size=10.5)
    f.arrow(590, y, 610, y, sw=1.2)
    for i, (x, name, cls) in enumerate(((160, "注意力", "blue"), (300, "FFN", "orange"), (440, "注意力", "blue"))):
        cy = y - 72
        f.arrow(x - 30, y - 10, x - 30, cy + 16, cls="ln", sw=1.1, label="读（Norm）" if i == 0 else None, lx=-36, ly=0, lsize=9.5)
        f.rect(x - 26, cy - 16, 52, 32, cls, rx=6, text=name, size=10.5)
        f.arrow(x + 30, cy + 16, x + 30, y - 10, cls="ln", sw=1.1, label="加回（+）" if i == 0 else None, lx=34, ly=0, lsize=9.5)
    f.text(300, 36, "层 1 … 层 N：每层只是在残差流上\"做一点修改\"，不是重建它", cls="mu", size=11)
    for x, word in ((200, "的 (0.21)"), (360, "首 (0.33)"), (520, "首都 (0.72)")):
        f.arrow(x, y + 10, x, y + 44, cls="purple-l", hcls="purple-s", sw=1.2, dash="3 3")
        f.rect(x - 42, y + 44, 84, 22, "purple", rx=5, text=word, size=10)
    f.text(360, 242, "logit lens：中途用最终的 Norm + LM Head 解码残差流，对\"北京是中国的\"的预测逐层变清晰", cls="mu", size=10.5)
    f.text(360, 268, "注意力子层在 token 之间搬信息，FFN 子层对每个 token 独立加工；两者都通过同一条残差流交流", cls="mu", size=10.5)
    return f


@figure("llm", "mla-compress")
def mla_compress():
    f = Fig(700, 300, "MLA：把 K、V 压成一个 512 维的潜向量来缓存，推理时 query 直接和潜向量做点积")
    f.text(170, 22, "MHA：每个 token 缓存 n_h 个头的 K 和 V", cls="tx", size=12, weight="600")
    for r, name in enumerate(("K", "V")):
        y = 44 + r * 46
        f.text(30, y + 14, name, cls="mu", size=11)
        for h in range(8):
            f.rect(44 + h * 34, y, 30, 28, "blue" if name == "K" else "orange", rx=4, text=f"h{h + 1}", size=9.5)
        f.text(330, y + 14, "…", cls="mu", size=12)
    f.text(170, 150, "DeepSeek-V3 规模：128 头 × 128 维 × 2 = 32768 个数 / token / 层", cls="mu", size=10.5)

    f.text(520, 22, "MLA：只缓存潜向量 c", cls="tx", size=12, weight="600")
    f.rect(400, 44, 60, 28, "gray", rx=5, text="h", size=11)
    f.arrow(460, 58, 500, 58, sw=1.2, label="W_DKV", ly=-9, lsize=10)
    f.rect(500, 44, 90, 28, "green", rx=5, text="c（512 维）", size=10.5, sw=1.8)
    f.text(545, 88, "KV Cache 里只有它\n（+ 64 维 RoPE 键）", cls="mu", size=10)
    f.arrow(545, 102, 545, 128, cls="ln", sw=1, dash="3 3", opacity=0.6)
    f.rect(470, 128, 150, 26, "blue", rx=5, text="k⁽ⁱ⁾ = c·W_UK⁽ⁱ⁾，v⁽ⁱ⁾ = c·W_UV⁽ⁱ⁾", size=9.5)
    f.text(545, 168, "需要时每个头各自从 c 还原（朴素做法）", cls="mu", size=10)

    f.text(350, 206, "权重吸收：不还原 K、V，把 W_UK 并进 query、W_UV 并进输出投影", cls="tx", size=12, weight="600")
    f.rect(60, 228, 150, 30, "blue", rx=6, text="q̃⁽ⁱ⁾ = W_UK⁽ⁱ⁾ᵀ q⁽ⁱ⁾", size=10.5)
    f.arrow(210, 243, 260, 243, sw=1.2, label="点积", ly=-9, lsize=10)
    f.rect(260, 228, 110, 30, "green", rx=6, text="缓存的 c_j", size=11)
    f.arrow(370, 243, 420, 243, sw=1.2, label="softmax", ly=-9, lsize=10)
    f.rect(420, 228, 110, 30, "green", rx=6, text="Σ pⱼ c_j", size=11)
    f.arrow(530, 243, 580, 243, sw=1.2, label="W_UV", ly=-9, lsize=10)
    f.rect(580, 228, 100, 30, "orange", rx=6, text="输出", size=11)
    f.text(350, 282, "decode 时相当于 128 个 query 头共享同一份 512 维的 K = V：缓存 576 个数 / token / 层，是 MHA 的 1/57", cls="mu", size=10.5)
    return f


@figure("llm", "sparsity-24")
def sparsity_24():
    f = Fig(700, 240, "2:4 半结构化稀疏：每 4 个连续权重保留 2 个，压缩存储加 2 位索引，稀疏 Tensor Core 直接跳过零")
    vals = [[0.8, -0.1, 0.05, -0.9, 0.3, -0.02, 0.7, 0.1], [-0.4, 0.6, -0.03, 0.02, 0.9, -0.5, 0.0, 0.1],
            [0.1, -0.7, 0.5, 0.0, -0.2, 0.8, -0.1, 0.4], [0.9, 0.2, -0.6, 0.1, 0.05, 0.3, -0.8, 0.02]]
    f.text(150, 22, "稠密权重（每行 8 个，分成 2 组）", cls="tx", size=12, weight="600")
    cell = 30
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            grp = row[c // 4 * 4: c // 4 * 4 + 4]
            keep = abs(v) >= sorted(abs(t) for t in grp)[2]
            x, y = 30 + c * cell + (8 if c >= 4 else 0), 36 + r * cell
            f.rect(x, y, cell - 3, cell - 3, "blue" if keep else "gray", rx=3, text=f"{v:g}" if keep else "0", size=9, sw=1)
    f.text(150, 166, "每组 4 个里按重要性留 2 个（蓝），其余置零", cls="mu", size=10.5)
    f.text(150, 186, "剪谁：|w| 或 Wanda 的 |w|·‖x‖（激活大的通道权重更重要）", cls="mu", size=10.5)

    f.arrow(300, 95, 345, 95, sw=1.4, label="压缩", ly=-10, lsize=11)
    f.text(500, 22, "稀疏格式：只存非零值 + 2 位索引", cls="tx", size=12, weight="600")
    for r, row in enumerate(vals):
        y = 36 + r * cell
        kept = []
        for gi in range(2):
            grp = row[gi * 4: gi * 4 + 4]
            thr = sorted(abs(t) for t in grp)[2]
            kept += [(c, v) for c, v in enumerate(grp) if abs(v) >= thr][:2]
        for i, (c, v) in enumerate(kept):
            f.rect(360 + i * cell, y, cell - 3, cell - 3, "blue", rx=3, text=f"{v:g}", size=9, sw=1)
        for i, (c, v) in enumerate(kept):
            f.rect(500 + i * 22, y + 4, 19, cell - 11, "orange", rx=3, text=str(c), size=9, sw=1)
    f.text(420, 166, "非零值：一半的空间", cls="mu", size=10.5)
    f.text(540, 166, "索引：每个 2 位", cls="mu", size=10.5)
    f.text(500, 190, "A100 / H100 的稀疏 Tensor Core 读索引、跳过零，矩阵乘算力 2×", cls="mu", size=10.5)
    f.text(350, 222, "非结构化稀疏精度最好但普通 kernel 加速不了；结构化（整头、整层）谁都能加速但精度掉得多；2:4 是中间的折中",
           cls="mu", size=10.5)
    return f


@figure("llm", "token-journey")
def token_journey():
    f = Fig(700, 440, "一个 token 的旅程：每一站的张量形状，prefill（T = 17）与 decode（T = 1）")
    f.text(130, 20, "站点", cls="tx", size=12, weight="600")
    f.text(400, 20, "prefill：一次送入整段提示词", cls="tx", size=12, weight="600")
    f.text(600, 20, "decode：每步一个 token", cls="tx", size=12, weight="600")
    stations = [
        ("对话模板 + 分词", "文本 → [1, 17]", "[1, 1]", "gray"),
        ("嵌入 embed_tokens", "[1, 17, 1024]", "[1, 1, 1024]", "gray"),
        ("RMSNorm", "[1, 17, 1024]", "[1, 1, 1024]", "purple"),
        ("q_proj / k_proj / v_proj", "q [1, 17, 2048]\nk、v [1, 17, 1024]", "q [1, 1, 2048]\nk、v [1, 1, 1024]", "blue"),
        ("RoPE，写入 KV Cache", "K Cache [1, 8, 17, 128]", "追加一列 → [1, 8, 18, 128]", "blue"),
        ("注意力（16 个 query 头，GQA 8 组）", "[1, 17, 2048]，分数 17×17", "[1, 1, 2048]，分数 1×18", "blue"),
        ("o_proj，加回残差流", "[1, 17, 1024]", "[1, 1, 1024]", "blue"),
        ("RMSNorm → gate / up", "[1, 17, 3072] × 2", "[1, 1, 3072] × 2", "orange"),
        ("SiLU(gate)·up → down，加回残差流", "[1, 17, 1024]", "[1, 1, 1024]", "orange"),
        ("最终 RMSNorm → LM Head", "[1, 17, 151936]，只用最后一行", "[1, 1, 151936]", "green"),
        ("采样 → 下一个 token", "'首'", "'都'", "green"),
    ]
    y0, h = 34, 34
    for i, (name, pre, dec, cls) in enumerate(stations):
        y = y0 + i * h
        f.rect(20, y, 220, h - 8, cls, rx=5, text=name, size=10.5)
        f.text(400, y + (h - 8) / 2, pre, cls="tx", size=10.5, family="mono")
        f.text(600, y + (h - 8) / 2, dec, cls="tx", size=10.5, family="mono")
        if i < len(stations) - 1:
            f.arrow(130, y + h - 8, 130, y + h, sw=1.1)
    f.rect(12, y0 + 2 * h - 4, 236, 7 * h - 2, "gray", rx=8, sw=1, dash="5 4")
    f.text(248, y0 + 2 * h + 4, "× 28 层", cls="mu", size=10.5, anchor="start")
    f.text(350, 424, "prefill 的每一站都是 [T, ·] 的矩阵乘（算力瓶颈）；decode 的每一站都是 [1, ·] 的矩阵–向量乘（访存瓶颈），KV Cache 让注意力只多算新的一列",
           cls="mu", size=10.5)
    return f


@figure("llm", "rlhf-dpo-flow")
def rlhf_dpo_flow():
    f = Fig(700, 300, "RLHF 要训练奖励模型再做 PPO（同时维护四个模型）；DPO 把奖励写成策略与参考模型的对数概率之比，直接在偏好对上训练")
    f.rect(20, 40, 90, 40, "gray", rx=7, text="预训练模型", size=11)
    f.arrow(110, 60, 150, 60, sw=1.3)
    f.rect(150, 40, 70, 40, "blue", rx=7, text="SFT", size=11.5)
    f.arrow(220, 60, 262, 60, sw=1.3)
    f.rect(262, 30, 150, 60, "orange", rx=7, text="偏好数据\n(x, y_w ≻ y_l)", size=11)
    f.text(337, 108, "人类（或模型）对同一问题的两个回答排序", cls="mu", size=10)

    f.text(140, 150, "RLHF", cls="tx", size=13, weight="600")
    f.arrow(300, 90, 160, 170, cls="ln", sw=1.2)
    f.rect(70, 170, 140, 40, "purple", rx=7, text="奖励模型 r(x, y)\n−log σ(r_w − r_l)", size=10)
    f.arrow(140, 210, 140, 236, sw=1.2)
    f.rect(20, 236, 240, 50, "red", rx=7, text="PPO：策略 + 参考（KL）+ 奖励 + 价值\n四个模型同时在显存里，rollout 要靠推理引擎", size=10)

    f.text(560, 150, "DPO", cls="tx", size=13, weight="600")
    f.arrow(380, 90, 540, 170, cls="ln", sw=1.2)
    f.rect(430, 170, 250, 40, "green", rx=7, text="r(x, y) = β · log( πθ(y|x) / π_ref(y|x) )", size=10.5)
    f.arrow(555, 210, 555, 236, sw=1.2)
    f.rect(430, 236, 250, 50, "green", rx=7, text="−log σ( β·[margin_w − margin_l] )\n只有策略和参考两个模型，一个分类损失", size=10)
    f.text(340, 230, "同一个目标：\n带 KL 约束的奖励最大化", cls="mu", size=10.5)
    return f


@figure("media", "pipeline-anatomy")
def pipeline_anatomy():
    f = Fig(700, 260, "文生图 pipeline 的三个部件：文本编码器算一次，去噪网络算几十次，VAE 解码一次")
    f.rect(20, 40, 90, 40, "gray", rx=7, text="提示词", size=11.5)
    f.arrow(110, 60, 150, 60, sw=1.3)
    f.rect(150, 30, 130, 60, "green", rx=7, text="文本编码器\nCLIP / T5", size=11)
    f.text(215, 104, "调用 1 次（CFG 再编码一次空提示词）\n输出的条件向量整个过程不变 → 可缓存", cls="mu", size=9.5)
    f.arrow(280, 60, 340, 120, sw=1.3)
    f.rect(20, 140, 90, 40, "gray", rx=7, text="随机噪声\n（种子）", size=10.5)
    f.arrow(110, 160, 300, 160, sw=1.3)
    f.rect(300, 118, 170, 84, "blue", rx=9, sw=1.8)
    f.text(385, 140, "去噪网络", cls="tx", size=12, weight="600")
    f.text(385, 160, "UNet / DiT", cls="mu", size=10.5)
    f.text(385, 184, "× N 步 × (CFG ? 2 : 1)", cls="tx", size=10.5)
    f.path("M 470 150 C 520 150 520 100 470 100 C 440 100 440 118 455 118", cls="blue-l", sw=1.6, dash="4 3")
    f.text(600, 96, "每一步：预测 → 调度器更新 x_t\n几乎全部计算量都在这里", cls="mu", size=9.5)
    f.arrow(470, 170, 520, 170, sw=1.3)
    f.rect(520, 150, 80, 40, "orange", rx=7, text="VAE 解码器", size=11)
    f.arrow(600, 170, 640, 170, sw=1.3)
    f.rect(640, 150, 50, 40, "gray", rx=7, text="图像", size=11)
    f.text(560, 212, "调用 1 次，但在像素分辨率上做卷积：\n激活是潜空间的 64 倍，常是显存峰值", cls="mu", size=9.5)
    f.text(350, 246, "拆成三个阶段之后服务层才能做的事：文本编码批处理与缓存、去噪每卡一个请求、VAE 解码和下一个请求的去噪重叠、中途预览与取消", cls="mu", size=10)
    return f


@figure("media", "unet-vs-dit")
def unet_vs_dit():
    f = Fig(700, 320, "UNet 在分辨率金字塔上做卷积（低分辨率层才有注意力）；DiT 把潜变量切成 token，整个网络是一叠 Transformer 块")
    f.text(170, 22, "UNet（SD 1.5 / SDXL）", cls="tx", size=12.5, weight="600")
    levels = [("128²，320 通道", 150, "blue"), ("64²，640 通道", 130, "blue"), ("32²，1280 通道，带注意力", 150, "purple")]
    for i, (name, w, cls) in enumerate(levels):
        y = 44 + i * 44
        f.rect(110 - w / 2, y, w, 30, cls, rx=5, text=name, size=9.5)
        f.rect(290 - w / 2, y, w, 30, cls, rx=5, text=name, size=9.5)
        f.arrow(110 + w / 2 + 2, y + 15, 290 - w / 2 - 2, y + 15, cls="ln", sw=1.1, dash="3 3", label="跳连" if i == 0 else None, ly=-9, lsize=9.5)
        if i < len(levels) - 1:
            f.arrow(110, y + 30, 110, y + 44, sw=1.1, label="下采样" if i == 0 else None, lx=-28, ly=0, lsize=9.5)
            f.arrow(290, y + 44, 290, y + 30, sw=1.1, label="上采样" if i == 0 else None, lx=30, ly=0, lsize=9.5)
    f.rect(120, 180, 100, 28, "red", rx=5, text="最底层：8²", size=10)
    f.text(170, 226, "卷积 + GroupNorm 为主，形状层层不同：\nMFU 低（15%～30%），难编译、难并行", cls="mu", size=10)
    f.text(170, 266, "注意力只在低分辨率层，输入分辨率翻倍\n时注意力 token 也翻 4 倍，但占比有限", cls="mu", size=10)

    f.line(350, 30, 350, 300, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(530, 22, "DiT（SD3 / FLUX / 视频模型）", cls="tx", size=12.5, weight="600")
    f.rect(400, 44, 110, 30, "gray", rx=5, text="潜变量 128²×16", size=10)
    f.arrow(510, 59, 550, 59, sw=1.2, label="patchify 2×2", ly=-13, lsize=9.5)
    f.rect(550, 44, 120, 30, "green", rx=5, text="4096 个 token × 64", size=10)
    f.arrow(610, 74, 610, 92, sw=1.2)
    f.rect(430, 92, 240, 150, "gray", rx=9, sw=1, dash="5 4")
    f.text(650, 104, "× N 层", cls="mu", size=10, anchor="end")
    blocks = [("AdaLN-Zero（时间步、条件 → 缩放 / 平移 / 门）", "orange"), ("自注意力（图像 token，或图文联合）", "blue"),
              ("交叉注意力（文本）或 MM-DiT 双流", "purple"), ("MLP", "blue")]
    for i, (name, cls) in enumerate(blocks):
        y = 112 + i * 31
        f.rect(445, y, 210, 24, cls, rx=4, text=name, size=9.5)
        if i < len(blocks) - 1:
            f.arrow(550, y + 24, 550, y + 31, sw=1)
    f.arrow(610, 242, 610, 258, sw=1.2)
    f.rect(550, 258, 120, 26, "gray", rx=5, text="unpatchify → 预测", size=10)
    f.text(525, 300, "形状全程不变、全是大矩阵乘：MFU 40%～55%\n编译、量化、序列并行都和 LLM 同一套", cls="mu", size=9.5)
    return f


@figure("media", "latent-shapes")
def latent_shapes():
    f = Fig(700, 230, "一张 1024² 的图在各层表示里有多少个数：像素 → VAE 潜变量 → DiT token")
    cols = [("像素", "1024 × 1024 × 3", "3,145,728 个数", 120, "gray"), ("SD VAE 潜变量（f8，4 通道）", "128 × 128 × 4", "65,536 个数（1/48）", 60, "blue"),
            ("FLUX VAE 潜变量（f8，16 通道）", "128 × 128 × 16", "262,144 个数（1/12）", 60, "green"), ("DiT token（patch 2×2）", "64 × 64 = 4096 个", "每个 64 维", 32, "orange")]
    x = 20
    for i, (name, shape, count, size, cls) in enumerate(cols):
        cx = x + 80
        f.rect(cx - size / 2, 110 - size / 2, size, size, cls, rx=4, sw=1.4)
        if i == 3:
            for gx in range(4):
                for gy in range(4):
                    f.rect(cx - 16 + gx * 8, 94 + gy * 8, 7, 7, "orange", rx=1, sw=0.6)
        f.text(cx, 30, name, cls="tx", size=10.5, weight="600")
        f.text(cx, 164, shape, cls="tx", size=10.5, family="mono")
        f.text(cx, 184, count, cls="mu", size=10)
        if i < len(cols) - 1:
            f.arrow(cx + size / 2 + 8, 110, cx + 160 - (cols[i + 1][3]) / 2 - 8, 110, sw=1.2,
                    label=["VAE 编码（×1/8 边长）", "", "打包 2×2"][i] if i != 1 else "", ly=-10, lsize=9.5)
        x += 160
    f.text(350, 214, "去噪网络在潜变量 / token 上跑几十步，便宜；VAE 解码只跑一次，却回到像素面积，激活是潜空间的 64 倍——显存峰值在这里", cls="mu", size=10)
    return f


@figure("media", "ulysses-ring")
def ulysses_ring():
    f = Fig(700, 330, "序列并行的两种做法：Ulysses 用 all-to-all 在「切序列」和「切头」之间转换；Ring 让 K、V 块绕环传递")
    f.text(170, 20, "Ulysses（切头）", cls="tx", size=12.5, weight="600")
    for g in range(4):
        y = 40 + g * 30
        f.text(30, y + 11, f"GPU {g}", cls="mu", size=10, anchor="start")
        f.rect(80, y, 60, 22, "blue", rx=3, text=f"序列块 {g}", size=9.5)
        f.text(110, y + 32, "" if g < 3 else "", cls="mu", size=9)
        f.rect(220, y, 70, 22, "orange", rx=3, text=f"所有序列·头{g}", size=9)
    f.arrow(140, 85, 220, 85, sw=1.4, label="all-to-all", ly=-10, lsize=10)
    f.text(255, 170, "每张卡对自己的 1/4 个头\n做完整序列的注意力", cls="mu", size=9.5)
    f.arrow(220, 210, 140, 210, sw=1.4, label="all-to-all 换回来", ly=14, lsize=10)
    f.text(170, 252, "通信量 ∝ 序列长度 × d，一步两次；\n要求头数能被卡数整除（Wan 40 头 → 最多 8 卡）", cls="mu", size=9.5)

    f.line(350, 30, 350, 300, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(530, 20, "Ring（切序列，K、V 绕环）", cls="tx", size=12.5, weight="600")
    centers = [(530, 100), (610, 160), (530, 220), (450, 160)]
    for g, (cx, cy) in enumerate(centers):
        f.rect(cx - 40, cy - 16, 80, 32, "blue", rx=5, text=f"GPU {g}\nQ{g}·K{g}·V{g}", size=9)
    for g in range(4):
        (x1, y1), (x2, y2) = centers[g], centers[(g + 1) % 4]
        f.arrow(x1 + (x2 - x1) * 0.3, y1 + (y2 - y1) * 0.3, x1 + (x2 - x1) * 0.7, y1 + (y2 - y1) * 0.7, cls="orange-l", hcls="orange-s", sw=1.8)
    f.text(530, 160, "K、V 块\n逐站传递", cls="mu", size=9.5)
    f.text(530, 262, "每收到一块 K、V 就算一块分数，用 online softmax 累加；\n传完一圈每张卡得到自己那段序列的完整输出", cls="mu", size=9.5)
    f.text(530, 300, "通信可以和计算重叠，对头数没要求；但卡多时环变长、延迟累加", cls="mu", size=9.5)
    return f


@figure("media", "smoothquant")
def smoothquant():
    f = Fig(700, 250, "SmoothQuant：把激活里少数通道的离群值按通道除掉，等量乘进权重，矩阵乘的结果不变")
    acts = [1, 1.2, 0.8, 9, 1.1, 0.9, 7.5, 1]
    ws = [1, 1.1, 0.9, 1, 1.2, 0.8, 1, 1.1]
    s = [max(a, 1) ** 0.5 for a in acts]

    def bars(x0, y0, vals, cls, label, scale):
        f.text(x0 + 70, y0 - 60, label, cls="tx", size=10.5)
        f.line(x0, y0, x0 + 150, y0, cls="ln", sw=1, opacity=0.5)
        for i, v in enumerate(vals):
            h = v * scale
            f.rect(x0 + 6 + i * 18, y0 - h, 13, h, cls, rx=2, sw=0.8)

    bars(30, 120, acts, "orange", "激活 X 的各通道幅度", 5)
    bars(200, 120, ws, "blue", "权重 W 对应的各行", 20)
    f.text(30, 142, "两个离群通道把 X 的量化步长撑大 9 倍，其余通道全被压扁", cls="mu", size=9.5, anchor="start")
    f.arrow(380, 85, 420, 85, sw=1.4, label="X / s，W × s", ly=-10, lsize=10)
    bars(430, 120, [a / si for a, si in zip(acts, s)], "orange", "X̂ = X · diag(s)⁻¹", 5)
    bars(560, 120, [w * si for w, si in zip(ws, s)], "blue", "Ŵ = diag(s) · W", 20)
    f.text(670, 164, "s_j = max|X_j|^α / max|W_j|^(1−α)，α 常取 0.5：难度在两边各担一半", cls="mu", size=9.5, anchor="end")
    f.text(350, 196, "X̂ · Ŵ = X · diag(s)⁻¹ · diag(s) · W = X · W：数学上完全等价，只是两边都变得「好量化」了", cls="tx", size=10.5)
    f.text(350, 226, "s 在校准时离线算好并合并进上一层的归一化权重，推理时没有额外算子；扩散模型里激活幅度还随时间步变，要按时间步校准", cls="mu", size=9.5)
    return f


@figure("serving", "server-topology")
def server_topology():
    f = Fig(700, 300, "一台 8 卡 H100 服务器里的数据通路：NVSwitch 把 8 张卡全连，每张卡经 PCIe 交换机接自己的网卡")
    f.rect(140, 20, 420, 34, "purple", rx=8, text="NVSwitch × 4：任意两卡之间 450 GB/s（单向），不分远近", size=11)
    for i in range(8):
        x = 60 + i * 76
        f.line(x + 28, 54, x + 28, 78, cls="purple-l", sw=2)
        f.rect(x, 78, 56, 44, "blue", rx=6, text=f"GPU {i}\nHBM 3.35 TB/s", size=8.5)
        f.line(x + 28, 122, x + 28, 150, cls="ln", sw=1.4)
        if i % 2 == 0:
            f.rect(x + 10, 150, 112, 26, "gray", rx=5, text="PCIe 交换机", size=9.5)
        f.line(x + 28, 176, x + 28, 204, cls="ln", sw=1.4)
        f.rect(x + 4, 204, 48, 30, "green", rx=5, text="NIC\n400 Gb/s", size=8.5)
    f.rect(240, 250, 220, 30, "orange", rx=6, text="CPU 与主机内存（卸载 KV、加载权重）", size=10)
    f.line(350, 176, 350, 250, cls="ln", sw=1.2, dash="4 3")
    f.text(120, 265, "GPU ↔ 交换机 ↔ CPU / 网卡：PCIe 5 x16 ≈ 55 GB/s", cls="mu", size=9.5)
    f.text(580, 265, "每卡一张网卡：≈ 50 GB/s 到别的机器", cls="mu", size=9.5)
    f.text(350, 294, "带宽阶梯：HBM 3350 ＞ NVLink 450 ＞ PCIe 55 ≈ 网卡 50（GB/s）——通信最频繁的切分放在 NVLink 覆盖的范围里", cls="mu", size=10)
    return f


@figure("serving", "kv-tiers")
def kv_tiers():
    f = Fig(700, 270, "KV Cache 的分层：越往下容量越大、带宽越低；块哈希是贯穿各层的键")
    tiers = [("L1  GPU 显存", "80 GB / 卡", "3.35 TB/s", "正在使用的 KV + 最热的前缀缓存", "blue", 340),
             ("L2  CPU 内存", "几百 GB ～ 1 TB", "经 PCIe 约 50 GB/s", "本机共享的前缀缓存；锁页才能 DMA", "green", 440),
             ("L3  SSD / 分布式存储", "TB ～ PB", "本地 NVMe 约 7 GB/s；网络 50 GB/s", "跨机器、跨实例共享（Mooncake Store、3FS、LMCache）", "orange", 540)]
    for i, (name, cap, bw, use, cls, w) in enumerate(tiers):
        y = 30 + i * 70
        f.rect(350 - w / 2, y, w, 50, cls, rx=8, sw=1.4)
        f.text(350, y + 16, name, cls="tx", size=12, weight="600")
        f.text(350, y + 36, f"容量 {cap}　带宽 {bw}", cls="tx", size=10)
        f.text(650, y + 25, use, cls="mu", size=9.5, anchor="end")
        if i < len(tiers) - 1:
            f.arrow(300, y + 50, 300, y + 70, sw=1.3, label="淘汰时写下去（写回）或算完就写（写穿）", lx=-120, ly=0, lsize=9)
            f.arrow(400, y + 70, 400, y + 50, sw=1.3, label="命中就预取，逐层读、边读边算", lx=110, ly=0, lsize=9)
    f.text(350, 250, "读回来还是重算？读的时间 = KV 字节数 / 带宽，重算的时间 = token 数 × 每 token 的 prefill 成本；短前缀重算更快，长前缀读更快", cls="mu", size=10)
    return f


@figure("serving", "pd-multiplex-sm")
def pd_multiplex_sm():
    f = Fig(700, 250, "同一张卡上 prefill 与 decode 的三种共存方式：时间上交替、空间上切 SM（green context）、分到不同的卡")
    f.text(120, 22, "时间上交替", cls="tx", size=12, weight="600")
    f.rect(30, 40, 180, 22, "gray", rx=3, sw=0.8)
    for x, w, kind in ((30, 70, "p"), (100, 18, "d"), (118, 60, "p"), (178, 32, "d")):
        f.rect(x, 40, w, 22, "orange" if kind == "p" else "blue", rx=3, text="prefill" if kind == "p" and w > 40 else ("d" if kind == "d" else ""), size=9)
    f.text(120, 78, "整卡轮流用：prefill 一来 decode 就被卡住", cls="mu", size=9.5)
    f.text(120, 96, "（分块 prefill：切小块混进 decode，每步都变慢一点）", cls="mu", size=9)

    f.text(350, 22, "空间上切 SM（PD 复用）", cls="tx", size=12, weight="600")
    f.rect(260, 40, 180, 22, "orange", rx=3, text="prefill：104 个 SM", size=9.5)
    f.rect(260, 66, 180, 14, "blue", rx=3, text="decode：28 个 SM", size=9)
    f.text(350, 100, "两条流各占一块 SM，同时跑：\ndecode 只和 prefill 争带宽与 L2，不被整段卡住", cls="mu", size=9.5)

    f.text(580, 22, "分到不同的卡（PD 分离）", cls="tx", size=12, weight="600")
    f.rect(500, 40, 76, 40, "orange", rx=5, text="prefill 实例", size=9.5)
    f.rect(600, 40, 76, 40, "blue", rx=5, text="decode 实例", size=9.5)
    f.arrow(576, 60, 600, 60, sw=1.3, label="传 KV", ly=-9, lsize=9)
    f.text(588, 100, "互不干扰，但要 RDMA / NVLink 传 KV，\n还要维护 xPyD 配比", cls="mu", size=9.5)
    f.line(30, 130, 670, 130, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(350, 152, "green context（CUDA 12.4+）：cuDevSmResourceSplitByCount 把 SM 切成几份，各建一个上下文和流，kernel 只在自己那份 SM 上跑", cls="tx", size=10)
    f.text(350, 178, "切分档位是调节旋钮：prefill 分得多 → 新请求的 TTFT 短；decode 分得多 → TPOT 稳。SGLang 的 PD 复用还让 prefill 每轮只跑几层，给 decode 让路", cls="mu", size=9.5)
    f.text(350, 204, "值不值：相比分块 prefill，decode 的 P99 TPOT 明显更稳；相比 PD 分离，不多买卡、不传 KV；代价是 prefill 变慢、两边都受带宽和 L2 的干扰", cls="mu", size=9.5)
    f.text(350, 230, "什么时候用：单卡或少卡部署、提示词中等长度、对 TPOT 抖动敏感的场景", cls="mu", size=9.5)
    return f


@figure("serving", "rdma-verbs")
def rdma_verbs():
    f = Fig(700, 300, "RDMA 的 verbs 模型：注册内存、建队列对，发送方 post 一个工作请求，网卡直接 DMA 到对方的内存，CPU 不参与搬运")
    for side, x0, title in ((0, 40, "发送方（prefill 实例）"), (1, 400, "接收方（decode 实例）")):
        f.text(x0 + 130, 22, title, cls="tx", size=12, weight="600")
        f.rect(x0, 40, 120, 60, "blue", rx=7, text="GPU 显存\nKV 块（已注册 MR）", size=9.5)
        f.rect(x0 + 140, 40, 120, 60, "gray", rx=7, text="CPU\n只负责 post / 收完成", size=9.5)
        f.rect(x0, 130, 260, 36, "green", rx=6, text="QP：发送队列 SQ + 接收队列 RQ　　CQ：完成队列", size=9.5)
        f.rect(x0 + 60, 190, 140, 40, "orange", rx=7, text="网卡（RNIC）", size=11)
        f.line(x0 + 60, 100, x0 + 60, 130, cls="ln", sw=1.2)
        f.line(x0 + 130, 166, x0 + 130, 190, cls="ln", sw=1.2)
    f.arrow(200, 210, 460, 210, cls="orange-l", hcls="orange-s", sw=2.2, label="RDMA WRITE：网卡到网卡，直接写进对方注册过的内存", ly=-12, lsize=10)
    f.text(330, 246, "① ibv_reg_mr 注册内存（固定页、拿到 lkey/rkey）　② 建 QP、交换地址与 rkey　③ ibv_post_send 一个 WR　④ 对方内存被写入，CQ 里出现完成事件", cls="mu", size=9.5)
    f.text(330, 270, "GPUDirect RDMA：网卡直接读写 GPU 显存，不经过主机内存；注册内存很慢（毫秒级），所以 KV 池要预先注册、反复复用", cls="mu", size=9.5)
    f.text(330, 292, "小消息多了会撞上网卡的消息速率上限：传 KV 要攒成大块，或者让 GPU 自己发起（IBGDA）", cls="mu", size=9.5)
    return f


@figure("serving", "ring-attention")
def ring_attention_serving():
    return ulysses_ring()


@figure("serving", "fsm-mask")
def fsm_mask():
    f = Fig(700, 270, "语法约束解码：自动机的当前状态决定允许的 token 集合，掩码加到 logits 上再采样，选中的 token 推进状态")
    states = ['{"city": "', "字符串", '", "temp_c": ', "整数", '"}']
    for i, s in enumerate(states):
        x = 30 + i * 132
        cls = "blue" if i % 2 == 0 else "orange"
        f.rect(x, 40, 112, 34, cls, rx=6, text=s, size=10, weight=None)
        if i < len(states) - 1:
            f.arrow(x + 112, 57, x + 132, 57, sw=1.2)
    f.text(350, 24, "模板的状态机：字面量段和字段段交替", cls="mu", size=10.5)
    f.rect(60, 110, 100, 34, "orange", rx=6, text="当前状态：整数", size=10.5, sw=1.6)
    f.arrow(160, 127, 220, 127, sw=1.3, label="沿词表前缀树\n算允许集合", ly=-16, lsize=9)
    toks = [("123", True), (" 45", False), ("-7", True), ("晴", False), ("}", False), ("2", True), ("abc", False), (",", True)]
    for i, (t, ok) in enumerate(toks):
        x = 225 + i * 54
        f.rect(x, 110, 48, 34, "green" if ok else "gray", rx=5, text=t, size=10.5, sw=1.4 if ok else 0.8)
        f.text(x + 24, 158, "允许" if ok else "−∞", cls="mu", size=9.5)
    f.text(350, 184, "掩码：不允许的 token 的 logit 设为 −∞，允许的 token 之间相对概率不变，模型在格式允许的范围内自由发挥", cls="mu", size=10)
    f.arrow(350, 196, 350, 214, sw=1.2)
    f.rect(230, 214, 240, 30, "blue", rx=6, text="采样 → 选中 \"-7\" → 状态推进（整数段仍可继续或进入下一段）", size=9.5)
    f.text(350, 262, "每一步都要算一遍允许集合：词表 15 万、状态机可能上千个状态，所以要预编译、缓存每个状态的掩码，并在 GPU 上并行应用", cls="mu", size=9.5)
    return f


@figure("serving", "float-order")
def float_order():
    f = Fig(700, 240, "同一行结果为什么会随 batch 变：kernel 按请求数选切分方式，加法顺序变了，浮点结果就变了")
    f.text(170, 22, "batch 小：split-K = 4，占满 SM", cls="tx", size=11.5, weight="600")
    for i in range(4):
        f.rect(40 + i * 66, 40, 58, 26, "blue", rx=4, text=f"段 {i + 1}", size=10)
        f.arrow(69 + i * 66, 66, 150 + i * 10, 96, sw=1)
    f.rect(100, 96, 130, 26, "orange", rx=4, text="((s1 + s2) + s3) + s4", size=9.5)
    f.text(170, 140, "= 0.30000001", cls="tx", size=11, family="mono")
    f.text(520, 22, "batch 大：不切分，顺序累加", cls="tx", size=11.5, weight="600")
    f.rect(400, 40, 240, 26, "blue", rx=4, text="整段 K 一次累加", size=10)
    f.arrow(520, 66, 520, 96, sw=1)
    f.rect(455, 96, 130, 26, "orange", rx=4, text="按 k 从头加到尾", size=9.5)
    f.text(520, 140, "= 0.29999998", cls="tx", size=11, family="mono")
    f.line(350, 30, 350, 150, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(350, 170, "同一个请求、同样的输入，和谁一起进 batch 决定了它的结果——这就是推理结果不逐位一致的根源之一", cls="mu", size=10)
    f.text(350, 192, "batch 无关（batch-invariant）kernel：切分方式固定，不随负载变；归约顺序固定；注意力的分块策略也固定", cls="tx", size=10)
    f.text(350, 214, "代价是放弃按形状选最优配置，吞吐降一到两成；强化学习要它，是因为训练侧和推理侧算出的概率必须对得上", cls="mu", size=9.5)
    return f


@figure("serving", "eplb")
def eplb():
    f = Fig(700, 300, "EPLB：先按负载把专家组打包到节点，再在节点内复制最热的专家，最后把专家摊到各卡——冷热差异靠冗余副本抹平")
    f.text(120, 22, "① 专家的冷热（路由统计）", cls="tx", size=11, weight="600")
    loads = [3, 9, 2, 4, 7, 2, 12, 3]
    for i, v in enumerate(loads):
        f.rect(30 + i * 24, 110 - v * 6, 18, v * 6, "orange" if v >= 7 else "blue", rx=2, sw=0.8)
        f.text(39 + i * 24, 122, f"E{i}", cls="mu", size=8.5)
    f.text(120, 140, "橙：热门专家", cls="mu", size=9.5)
    f.arrow(230, 80, 270, 80, sw=1.3)
    f.text(390, 22, "② 组打包到节点（分层策略）", cls="tx", size=11, weight="600")
    for n in range(2):
        y = 44 + n * 50
        f.rect(280, y, 220, 40, "gray", rx=6, sw=1)
        f.text(300, y + 20, f"节点 {n}", cls="mu", size=9.5, anchor="start")
        groups = ["组 1", "组 6", "组 3", "组 4"] if n == 0 else ["组 0", "组 7", "组 2", "组 5"]
        for g, name in enumerate(groups):
            f.rect(340 + g * 40, y + 8, 34, 24, "blue", rx=4, text=name, size=8.5)
    f.text(390, 150, "每个节点分到相同数量的组，组的负载之和尽量相等", cls="mu", size=9.5)
    f.arrow(500, 80, 540, 80, sw=1.3)
    f.text(620, 22, "③ 节点内复制热门专家", cls="tx", size=11, weight="600")
    for i, name in enumerate(["E6", "E6′", "E1", "E1′", "E4", "E0", "E3", "E2"]):
        x = 545 + (i % 4) * 36
        y = 44 + (i // 4) * 32
        f.rect(x, y, 32, 24, "orange" if "′" in name or name in ("E6", "E1") else "blue", rx=4, text=name, size=8.5)
    f.text(620, 125, "冗余副本后每张卡的负载接近", cls="mu", size=9.5)
    f.line(30, 170, 670, 170, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(350, 190, "分组限制路由：256 个专家分 8 组，先按组得分选 4 组、再在组内选 8 个\n一个 token 最多只跨 4 个节点，机间 all-to-all 的流量有上限", cls="tx", size=10)
    f.text(350, 228, "分层先保证「同一组尽量同一节点」，省机间带宽；全局不管节点，均衡更好但跨机流量更多\nprefill（大 batch，带宽敏感）和 decode（延迟敏感）常选不同策略", cls="mu", size=9.5)
    f.text(350, 262, "负载统计滚动更新，几分钟重排一次专家、期间权重要热搬；专家均衡了，DP attention 那侧还有各 rank KV 总量的不均衡（见下一节）", cls="mu", size=9.5)
    return f


@figure("serving", "fp8-scaling")
def fp8_scaling():
    f = Fig(700, 250, "FP8 的缩放粒度：激活按每个 token 每 128 个通道一个缩放，权重按 128×128 的块一个缩放，GEMM 沿 K 每 128 累加一次就乘上两个缩放")
    f.text(140, 22, "激活 A [M, K]", cls="tx", size=11, weight="600")
    for i in range(4):
        for j in range(6):
            f.rect(40 + j * 34, 40 + i * 18, 32, 16, "orange", rx=2, sw=0.6)
        f.text(252, 48 + i * 18, f"s_a[{i}, 0..5]", cls="mu", size=8.5, anchor="start")
    f.text(140, 128, "每行（token）每 128 个 K 一个缩放：1×128", cls="mu", size=9.5)
    f.text(430, 22, "权重 W [K, N]", cls="tx", size=11, weight="600")
    for i in range(3):
        for j in range(4):
            f.rect(360 + j * 36, 40 + i * 24, 34, 22, "blue", rx=2, sw=0.6, text="128²", size=8)
    f.text(430, 128, "每 128×128 的块一个缩放：离群值只影响一块", cls="mu", size=9.5)
    f.rect(540, 40, 130, 70, "green", rx=6, text="DeepGEMM：\nFP8 Tensor Core 乘\n每 128 个 K 在 fp32 里\n乘缩放后累加", size=9)
    f.line(30, 150, 670, 150, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(350, 172, "按张量一个缩放时，一个离群值把整个张量的步长撑大；按块缩放把影响局限在 128 个数里，精度接近 bf16——DeepSeek-V3 全程 FP8 的基础", cls="tx", size=9.5)
    f.text(350, 196, "MoE 的分组 GEMM：各专家的 token 数不同，把它们拼成一个大矩阵、按专家分段，一次 kernel 启动算完所有专家；token 要先按专家排序", cls="mu", size=9.5)
    f.text(350, 220, "Hopper 的 FP8 累加器精度有限（约 14 位），所以每 128 个 K 就要把部分和搬到 fp32 寄存器里——DeepGEMM 用 CUDA 核心做这一步", cls="mu", size=9.5)
    return f


@figure("serving", "nsa-branches")
def nsa_branches():
    f = Fig(700, 300, "两种原生稀疏注意力：NSA 用压缩、选择、滑窗三路加门控；DSA 用轻量索引器给每个 query 挑 2048 个 token 再做 MLA")
    f.text(170, 22, "NSA（三路 + 门控）", cls="tx", size=12, weight="600")
    f.rect(30, 44, 70, 34, "gray", rx=6, text="query", size=10.5)
    branches = [("压缩分支：每段 token 压成一个粗键值，看全局概览", "blue"), ("选择分支：按压缩分数挑最相关的几块，看细节", "orange"), ("滑窗分支：最近的 token，看局部", "green")]
    for i, (name, cls) in enumerate(branches):
        y = 44 + i * 44
        f.arrow(100, 61, 130, y + 17, sw=1.1)
        f.rect(130, y, 190, 34, cls, rx=6, text=name, size=8.5)
        f.arrow(320, y + 17, 340, 105, sw=1.1)
    f.rect(340, 88, 50, 34, "purple", rx=6, text="门控\n加权", size=9.5)
    f.text(175, 196, "三路都只看一小部分 token：总代价远小于全注意力\n训练时就按这个结构学（原生），kernel 按块组织（Triton）", cls="mu", size=9.5)

    f.line(410, 30, 410, 240, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(555, 22, "DSA（闪电索引器 + MLA）", cls="tx", size=12, weight="600")
    f.rect(430, 44, 70, 34, "gray", rx=6, text="query", size=10.5)
    f.arrow(500, 61, 530, 61, sw=1.1)
    f.rect(530, 44, 150, 34, "orange", rx=6, text="索引器：64 头 × 128 维（FP8）\n和所有历史 token 打分", size=8.5)
    f.arrow(605, 78, 605, 100, sw=1.1, label="top-2048", lx=40, ly=0, lsize=9)
    f.rect(530, 100, 150, 34, "blue", rx=6, text="只对选中的 2048 个 token\n做 MLA 注意力", size=8.5)
    f.text(555, 160, "每对 (query, key) 索引器只花 16K FLOPs，\nMLA 要 278K；上下文越长省得越多\n（128K 时 decode 的计算约 30 倍）", cls="mu", size=9.5)
    f.text(555, 215, "缓存每个 token 多存 132 字节的索引键", cls="mu", size=9.5)
    f.text(350, 262, "对推理引擎的新要求：注意力 kernel 要支持按索引取 KV（gather），分页 KV 的块表要能表达稀疏选择，前缀缓存和投机解码都要重新适配", cls="mu", size=9.5)
    f.text(350, 284, "两者都在训练时就稀疏（原生），而不是训好之后再剪——所以精度几乎不掉，这是和 StreamingLLM 这类事后淘汰最大的区别", cls="mu", size=9.5)
    return f


@figure("serving", "top-down")
def top_down():
    f = Fig(700, 250, "自顶向下地找瓶颈：先看端到端指标和理论下限，再看服务端指标，再看一步的时间线，最后才看 kernel")
    levels = [("① 端到端指标", "TTFT / TPOT / 吞吐 vs 理论下限：慢在哪一段？所有负载都慢还是高负载才慢？", "blue"),
              ("② 服务端指标", "排队数、KV 使用率、抢占次数、前缀缓存命中率、每步 batch 大小——很多\"慢\"是调度与容量问题", "green"),
              ("③ 一步的时间线", "profiler：CPU 调度与输入准备 → GPU 前向 → 采样 → 结果处理；GPU 是否一直在忙？", "orange"),
              ("④ 算子与 kernel", "时间花在哪些 kernel？离各自的屋顶线多远？访存受限还是算力受限？", "purple")]
    for i, (name, desc, cls) in enumerate(levels):
        y, x0, w = 30 + i * 50, 30 + i * 22, 640 - i * 22
        f.rect(x0, y, w, 40, cls, rx=8, sw=1.3)
        f.text(x0 + 14, y + 20, name, cls="tx", size=11, weight="600", anchor="start")
        f.text(x0 + 130, y + 20, desc, cls="tx", size=9, anchor="start")
    f.text(350, 240, "每一层都可能直接给出答案；跳过上面的层直接看 kernel，最常见的结果是优化了一个只占 5% 的东西", cls="mu", size=10)
    return f


@figure("serving", "weight-update-flow")
def weight_update_flow():
    f = Fig(700, 270, "不重启地换权重：在两步之间暂停调度，把检查点转成 kernel 的格式后原地写进同一块显存，CUDA Graph 记住的地址才不会失效")
    steps = [("调度器暂停\n在两步之间", "gray"), ("新权重到达\n（训练器广播 / 对象存储）", "blue"), ("转换格式\n合并 QKV、重排 GQA、\n量化、按 TP 切分", "orange"), ("原地拷贝进\n同一块显存", "green"), ("恢复调度\n在途请求继续", "gray")]
    for i, (name, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 40, 120, 60, cls, rx=7, text=name, size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 120, 70, x + 136, 70, sw=1.3)
    f.text(350, 124, "在途请求的 KV 还是旧权重算出来的：短的让它跑完，长的要么接受轻微不一致，要么丢弃重算——取决于用途", cls="mu", size=9.5)
    f.line(30, 146, 670, 146, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(120, 170, "为什么不能 new 一块显存", cls="tx", size=11, weight="600")
    f.rect(40, 186, 160, 60, "red", rx=7, text="CUDA Graph 回放时\n只认录制时的指针\n新地址 = 用旧权重", size=9.5)
    f.text(430, 170, "RL 训练的闭环", cls="tx", size=11, weight="600")
    f.rect(270, 186, 110, 60, "blue", rx=7, text="训练器\n（FSDP / Megatron）", size=9.5)
    f.arrow(380, 206, 450, 206, sw=1.3, label="参数广播（NCCL / RDMA）", ly=-10, lsize=9)
    f.rect(450, 186, 110, 60, "green", rx=7, text="推理引擎\n（rollout）", size=9.5)
    f.arrow(450, 236, 380, 236, sw=1.3, label="采样结果", ly=12, lsize=9)
    f.text(620, 216, "每几分钟一轮，\n更新要秒级完成", cls="mu", size=9.5)
    return f


@figure("serving", "multi-lora-batch")
def multi_lora_batch():
    f = Fig(700, 250, "多 LoRA 服务：基座矩阵乘所有 token 一起算，LoRA 的小矩阵按各自的适配器分组计算，再加回去")
    f.rect(30, 50, 110, 150, "gray", rx=7, sw=1.2)
    f.text(85, 40, "一个混合 batch", cls="mu", size=10)
    rows = [("请求 1，适配器 3", 1), ("请求 2，适配器 5", 1), ("请求 3，适配器 3", 37), ("请求 4，适配器 0", 1), ("请求 5，适配器 7", 12), ("请求 6，适配器 5", 1)]
    for i, (name, n) in enumerate(rows):
        f.rect(40, 58 + i * 23, 90, 18, "blue" if n > 1 else "orange", rx=3, text=name, size=7.5)
    f.arrow(140, 125, 180, 125, sw=1.3)
    f.rect(180, 60, 150, 50, "blue", rx=7, text="基座 GEMM\nx · W（全部 token 一次）", size=10)
    f.rect(180, 140, 150, 50, "orange", rx=7, text="LoRA 分支\nx · A_i · B_i（按适配器）", size=10)
    f.arrow(330, 85, 380, 118, sw=1.2)
    f.arrow(330, 165, 380, 132, sw=1.2)
    f.circle(392, 125, 12, "green", text="+", size=14)
    f.arrow(404, 125, 440, 125, sw=1.3)
    f.rect(440, 100, 80, 50, "gray", rx=7, text="y", size=12)
    f.text(600, 70, "SGMV：同一适配器的 token 排在一起，\n每一段一次小矩阵乘（prefill 友好）", cls="mu", size=9.5)
    f.text(600, 125, "BGMV：每个 token 按编号取自己的 A、B，\n逐 token 矩阵 × 向量（decode 友好）", cls="mu", size=9.5)
    f.text(600, 180, "秩 r 只有 8～64：LoRA 分支的 FLOP\n是基座的 1%～2%，显存按适配器数线性增长", cls="mu", size=9.5)
    f.text(350, 230, "适配器像 KV 一样要调度：常用的常驻显存，冷的放 CPU 按需搬；同一步里适配器种类太多会让 SGMV 的段变碎", cls="mu", size=9.5)
    return f


@figure("serving", "k8s-reconcile")
def k8s_reconcile():
    f = Fig(700, 260, "Kubernetes 的控制器模式：你声明期望状态，控制器不断比较期望与实际，差多少就补多少——apply 之后「什么都没发生」是正常的")
    f.rect(30, 50, 130, 60, "orange", rx=7, text="期望状态\n（YAML：3 个副本）", size=10)
    f.arrow(160, 80, 220, 80, sw=1.3, label="kubectl apply", ly=-10, lsize=9.5)
    f.rect(220, 50, 150, 60, "blue", rx=7, text="API Server + etcd\n（唯一的真相来源）", size=10)
    f.arrow(370, 80, 430, 80, sw=1.3, label="watch", ly=-10, lsize=9.5)
    f.rect(430, 40, 160, 80, "green", rx=7, text="控制器 reconcile 循环\n读期望 → 看实际 → 补差\n（创建 / 删除 Pod）", size=9.5)
    f.arrow(510, 120, 510, 160, sw=1.3, label="动作", lx=24, ly=0, lsize=9.5)
    f.rect(430, 160, 160, 50, "gray", rx=7, text="实际状态\n（节点上跑着的 Pod）", size=10)
    f.arrow(430, 185, 300, 185, sw=1.2, label="kubelet 上报状态", ly=-10, lsize=9)
    f.path("M 300 185 C 250 185 250 110 290 110", cls="ln", sw=1.2, dash="4 3")
    f.text(350, 240, "每一种对象都有自己的控制器：Deployment 管副本数，Service 管流量，LeaderWorkerSet 管一组互相认识的 Pod；出问题先看对象的 status 和 events", cls="mu", size=10)
    return f


@figure("serving", "gpu-to-pod")
def gpu_to_pod():
    f = Fig(700, 230, "一张卡怎么到达容器：驱动在节点上，device plugin 把卡登记成资源，调度器按 requests 选节点，容器运行时把设备文件挂进去")
    steps = [("节点：NVIDIA 驱动\n/dev/nvidia0..7", "gray"), ("device plugin\n向 kubelet 登记\nnvidia.com/gpu: 8", "blue"), ("调度器\n按 requests 选节点\n（拓扑、亲和、污点）", "orange"), ("kubelet + 容器运行时\n把设备文件、驱动库\n挂进容器", "green"), ("Pod 里的进程\nnvidia-smi 看到\n分到的那几张", "gray")]
    for i, (name, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 50, 120, 70, cls, rx=7, text=name, size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 120, 85, x + 136, 85, sw=1.3)
    f.text(350, 150, "GPU 只能按整卡申请、不能超卖；分到哪几张卡决定了 NVLink 是否连通——TP=4 的实例要同一个 NVSwitch 域里的 4 张卡", cls="tx", size=10)
    f.text(350, 174, "切分一张卡：MIG 把一张卡切成几个硬隔离的实例（各有自己的 SM 和显存），时间片则是分时复用、没有显存隔离", cls="mu", size=9.5)
    f.text(350, 198, "让 GPU 节点只跑该跑的东西：污点 + 容忍、节点亲和；驱动升级要先把节点排空（drain）", cls="mu", size=9.5)
    return f


@figure("serving", "lws-pd")
def lws_pd():
    f = Fig(700, 280, "多机实例在 Kubernetes 上的形状：LeaderWorkerSet 把一组互相认识的 Pod 当作一个实例；PD 分离就是两组实例加一个路由")
    f.text(170, 22, "LeaderWorkerSet：一个实例 = leader + workers", cls="tx", size=11, weight="600")
    f.rect(30, 40, 280, 110, "gray", rx=8, sw=1, dash="5 4")
    f.rect(45, 56, 80, 36, "orange", rx=6, text="leader Pod\nrank 0", size=9)
    for i in range(3):
        f.rect(140 + i * 56, 56, 50, 36, "blue", rx=6, text=f"worker\nrank {i + 1}", size=8.5)
    f.text(170, 112, "组内按稳定的主机名互相找到：TP / PP 跨 Pod 建通信组", cls="mu", size=9)
    f.text(170, 132, "整组一起创建、一起重启（gang），少一个都不算就绪", cls="mu", size=9)
    f.text(510, 22, "PD 分离的形状", cls="tx", size=11, weight="600")
    f.rect(360, 40, 100, 44, "green", rx=6, text="网关 / 路由\n（选实例、配比）", size=9)
    for i, (name, cls, y) in enumerate((("prefill 实例 ×P", "orange", 100), ("decode 实例 ×D", "blue", 100))):
        x = 480 + i * 110
        f.rect(x, y, 100, 44, cls, rx=6, text=name, size=9.5)
        f.arrow(410 + (i * 40), 84, x + 50, 100, sw=1.1)
    f.arrow(580, 122, 590, 122, cls="ln", sw=1.2)
    f.text(535, 160, "KV 经 RDMA 从 P 传到 D", cls="mu", size=9)
    f.line(30, 180, 670, 180, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(350, 200, "一个请求怎么找到实例：Service 做四层负载均衡只会轮询，推理要按前缀缓存、按负载选实例，所以前面要有自己的路由（网关）", cls="tx", size=10)
    f.text(350, 224, "有状态的东西：模型权重（挂 PVC 或对象存储 + 本地缓存）、KV 卸载用的本地盘、路由表——都不该随 Pod 的生死丢掉", cls="mu", size=9.5)
    f.text(350, 248, "扩缩容的单位是整个实例（一组 Pod），不是单个 Pod；滚动发布要按组、配合探针，别让半个实例接流量", cls="mu", size=9.5)
    return f


@figure("cuda", "warp-reduce")
def warp_reduce():
    f = Fig(700, 250, "warp 归约：__shfl_down_sync 每一步把 offset 之外的值加过来，5 步后 lane 0 拿到 32 个值的和")
    lanes = 16
    for step, off in enumerate((8, 4, 2, 1)):
        y = 36 + step * 46
        f.text(62, y + 11, f"offset = {off}", cls="mu", size=10, anchor="end")
        for i in range(lanes):
            x = 70 + i * 38
            active = i < off * 2
            cls = "blue" if i < off else ("orange" if active else "gray")
            f.rect(x, y, 32, 22, cls, rx=3, text=str(i), size=9, sw=0.8)
            if i >= off and active:
                f.path(f"M {x + 16} {y} C {x + 16} {y - 14} {x - off * 38 + 16} {y - 14} {x - off * 38 + 16} {y}", cls="orange-l", sw=1.2)
    f.text(360, 222, "示意 16 个 lane（真实是 32 个，从 offset = 16 开始）：蓝色 lane 收到右边 offset 处的值并累加，灰色 lane 的值不再需要", cls="mu", size=10)
    f.text(360, 242, "寄存器之间直接交换，不经过共享内存、不需要 __syncthreads；块内再把各 warp 的部分和经共享内存合并一次", cls="mu", size=10)
    return f


@figure("cuda", "warp-divergence")
def warp_divergence():
    f = Fig(700, 230, "warp 分歧：一个 warp 里的线程走了不同分支，硬件把两条路径先后执行，各自只有一部分 lane 活跃")
    f.text(40, 24, "if (lane % 4 == 0) A(); else B();", cls="tx", size=11, family="mono", anchor="start")
    for row, (label, pred) in enumerate((("执行 A：活跃的 lane", lambda i: i % 4 == 0), ("执行 B：活跃的 lane", lambda i: i % 4 != 0))):
        y = 50 + row * 60
        f.text(40, y + 12, label, cls="mu", size=10, anchor="start")
        for i in range(32):
            x = 40 + i * 20
            f.rect(x, y + 24, 17, 20, "green" if pred(i) else "gray", rx=2, sw=0.6)
    f.text(360, 180, "两段时间加起来，warp 一共跑了 A + B 的全部指令：分歧越均匀、分支越长，浪费越大（这里 A 时只有 1/4 的 lane 在干活）", cls="mu", size=10)
    f.text(360, 204, "Volta 之后每个线程有独立的 PC，分歧的两路可以交错，但仍不能同时执行；让分支按 warp 对齐（lane 整组走同一路）才没有代价", cls="mu", size=10)
    return f


@figure("cuda", "transpose-tile")
def transpose_tile():
    f = Fig(700, 240, "矩阵转置：按行读进共享内存的 tile，按列取出来写——两边都是合并访问；tile 多加一列 padding 避开 bank 冲突")
    def grid(x0, y0, n, cell, cls, hi_row=None, hi_col=None):
        for i in range(n):
            for j in range(n):
                c = cls
                if hi_row is not None and i == hi_row:
                    c = "orange"
                if hi_col is not None and j == hi_col:
                    c = "orange"
                f.rect(x0 + j * cell, y0 + i * cell, cell - 1.5, cell - 1.5, c, rx=1.5, sw=0.5)
    grid(40, 50, 8, 18, "blue", hi_row=2)
    f.text(112, 36, "全局内存 A（按行读）", cls="tx", size=10.5)
    f.text(112, 206, "一个 warp 读连续的一行：合并", cls="mu", size=9.5)
    f.arrow(190, 122, 240, 122, sw=1.3, label="读入", ly=-9, lsize=9.5)
    grid(250, 50, 8, 18, "green", hi_row=2)
    f.rect(250 + 8 * 18, 50, 16, 8 * 18 - 1.5, "gray", rx=1.5, sw=0.5)
    f.text(330, 36, "共享内存 tile[32][33]", cls="tx", size=10.5)
    f.text(330, 206, "多出一列：同一列的元素落在不同 bank", cls="mu", size=9.5)
    f.arrow(420, 122, 470, 122, sw=1.3, label="按列取", ly=-9, lsize=9.5)
    grid(480, 50, 8, 18, "purple", hi_col=2)
    f.text(552, 36, "全局内存 Aᵀ（按行写）", cls="tx", size=10.5)
    f.text(552, 206, "取出的一列正好是输出的一行：合并", cls="mu", size=9.5)
    f.text(350, 230, "朴素转置里读和写总有一边是跨步访问（每个线程隔一整行），带宽只剩几分之一；tile 中转把跨步留在共享内存里，那里不怕跨步、只怕 bank 冲突", cls="mu", size=9.5)
    return f


@figure("cuda", "tma-pipeline")
def tma_pipeline():
    f = Fig(700, 300, "Hopper 的异步流水：生产者 warp 用 TMA 发起整块拷贝，mbarrier 计数到达，消费者 warp 组用 wgmma 算上一级的数据")
    f.text(120, 22, "共享内存的多级缓冲", cls="tx", size=11.5, weight="600")
    for i, (name, cls) in enumerate((("级 0：wgmma 正在算", "orange"), ("级 1：TMA 正在填", "blue"), ("级 2：TMA 正在填", "blue"), ("级 3：空，等释放", "gray"))):
        f.rect(30, 40 + i * 36, 180, 28, cls, rx=5, text=name, size=9.5)
    f.text(120, 196, "每级一个 mbarrier：\nTMA 搬完自动 arrive，消费者 wait", cls="mu", size=9.5)
    f.rect(270, 40, 170, 60, "blue", rx=7, text="生产者 warp（1 个）\ncp.async.bulk.tensor\n给出坐标，硬件按张量映射搬整块", size=9)
    f.rect(270, 124, 170, 60, "orange", rx=7, text="消费者 warp 组（2～3 个）\nwgmma.mma_async\n直接从共享内存读 A、B", size=9)
    f.arrow(440, 70, 500, 70, sw=1.3, label="arrive", ly=-9, lsize=9)
    f.arrow(440, 154, 500, 154, sw=1.3, label="释放", ly=-9, lsize=9)
    f.rect(500, 40, 170, 144, "gray", rx=7, sw=1, dash="5 4")
    f.text(585, 60, "全局内存 → 共享内存", cls="tx", size=10)
    f.text(585, 84, "TMA：一条指令搬一个\n多维 tile，地址计算\n和边界处理都在硬件里", cls="mu", size=9)
    f.text(585, 140, "寄存器几乎不参与搬运，\n线程只负责发起和等待", cls="mu", size=9)
    f.text(350, 234, "warp 专门化：搬数据的和算的是不同 warp，各自用 setmaxnreg 调寄存器配额；搬运的延迟完全藏在计算后面", cls="tx", size=10)
    f.text(350, 258, "cp.async（Ampere）是每个线程搬自己那几个元素、靠 commit/wait 分组；TMA 把整块搬运交给硬件，线程块集群还能把 tile 多播给邻居", cls="mu", size=9.5)
    f.text(350, 282, "Blackwell 再进一步：tcgen05 的 mma 从 tensor memory 读写累加器，寄存器压力进一步下降", cls="mu", size=9.5)
    return f


@figure("cuda", "compile-pipeline")
def compile_pipeline():
    f = Fig(700, 230, "torch.compile 的三层：Dynamo 从字节码里抓出图，AOT Autograd 把前向和反向变成算子图，Inductor 融合并生成 Triton / C++ kernel")
    steps = [("Python 函数", "gray", "eager 代码\n（含控制流）"), ("Dynamo", "blue", "改字节码，抓 FX 图\n遇到不支持的就 graph break"), ("AOT Autograd", "purple", "展开成 ATen 算子图\n前向 + 反向"), ("Inductor", "orange", "融合、调度、\n生成 Triton / C++"), ("kernel", "green", "少量大 kernel\n+ 启动代码")]
    for i, (name, cls, desc) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 40, 120, 40, cls, rx=7, text=name, size=11)
        f.text(x + 60, 110, desc, cls="mu", size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 120, 60, x + 136, 60, sw=1.3)
    f.text(350, 160, "guard：Dynamo 记下图成立的条件（张量形状、dtype、Python 对象的属性），下次调用先检查 guard，不符就重新编译", cls="tx", size=10)
    f.text(350, 184, "推理引擎常用 mode=\"reduce-overhead\"（配 CUDA Graph）或只编译小算子的融合，形状分桶避免反复重编译", cls="mu", size=9.5)
    f.text(350, 208, "融合省下的是访存：几个逐元素算子变成一个 kernel，中间结果不再写回显存", cls="mu", size=9.5)
    return f


@figure("cuda", "dispatcher-keys")
def dispatcher_keys():
    f = Fig(700, 250, "一次 torch.add 怎么走到 kernel：按张量的 dispatch key 一层层分发——Autograd、Autocast、后端（CUDA / CPU / Meta）")
    f.rect(30, 50, 120, 40, "gray", rx=7, text="torch.add(a, b)", size=10.5)
    f.arrow(150, 70, 190, 70, sw=1.3)
    keys = [("Autograd", "purple", "记录反向\n（训练时）"), ("Autocast", "orange", "混合精度\n改 dtype"), ("后端：CUDA", "blue", "选 kernel\n（dtype、布局）"), ("kernel", "green", "add_kernel\n<float>")]
    for i, (name, cls, desc) in enumerate(keys):
        x = 190 + i * 125
        f.rect(x, 50, 108, 40, cls, rx=7, text=name, size=10.5)
        f.text(x + 54, 116, desc, cls="mu", size=9.5)
        if i < len(keys) - 1:
            f.arrow(x + 108, 70, x + 125, 70, sw=1.3)
    f.text(350, 160, "dispatch key 是张量上的一组位：按优先级取最高的那个先处理，处理完再把自己去掉、重新分发（redispatch）", cls="tx", size=10)
    f.text(350, 184, "注册自定义算子 = 给某个 key 注册一个实现：TORCH_LIBRARY_IMPL(myops, CUDA, m)；Meta / fake tensor 只算形状不算值，编译器靠它推导", cls="mu", size=9.5)
    f.text(350, 208, "一次分发几微秒：decode 一步上千个小算子时这就是可观的开销，所以要 CUDA Graph 或者把小算子融合掉", cls="mu", size=9.5)
    f.text(350, 232, "推理引擎里的自定义 kernel（FlashInfer、vLLM 的 custom ops）都是通过这套机制接进 PyTorch 的", cls="mu", size=9.5)
    return f


@figure("cuda", "pdl-overlap")
def pdl_overlap():
    f = Fig(700, 240, "PDL：下一个 kernel 的序言（加载权重、算地址）提前上场，和上一个 kernel 的尾巴重叠；griddepcontrol 在真正要用数据时才等")
    f.text(60, 24, "普通", cls="tx", size=11, weight="600", anchor="start")
    f.rect(60, 36, 200, 24, "blue", rx=3, text="kernel 1", size=10)
    f.rect(275, 36, 60, 24, "gray", rx=3, text="空隙", size=9)
    f.rect(335, 36, 200, 24, "orange", rx=3, text="kernel 2（序言 + 主体）", size=10)
    f.text(60, 100, "PDL", cls="tx", size=11, weight="600", anchor="start")
    f.rect(60, 112, 200, 24, "blue", rx=3, text="kernel 1　　　launch_dependents →", size=9.5)
    f.rect(200, 142, 100, 24, "orange", rx=3, text="kernel 2 序言", size=9.5)
    f.rect(300, 142, 160, 24, "orange", rx=3, text="wait → 主体", size=9.5)
    f.arrow(260, 128, 300, 142, sw=1.1)
    f.text(560, 128, "空隙 = 启动延迟 + 尾部\n只剩几 µs 的真正依赖", cls="mu", size=9.5)
    f.text(350, 190, "decode 一步几百个小 kernel，每个 2～5 µs 的空隙累计起来可占三成；PDL 把空隙压到接近零，CUDA Graph 里也能用", cls="tx", size=10)
    f.text(350, 214, "再往前一步是 megakernel：整个前向写进一个常驻 kernel，block 之间用全局内存的计数器同步，没有启动、没有空隙，但要自己做调度", cls="mu", size=9.5)
    return f


@figure("cuda", "triton-program-grid")
def triton_program_grid():
    f = Fig(700, 230, "Triton 的编程模型：一个 program 处理一个 BLOCK 的元素，program_id 决定它负责哪一块，越界用 mask 挡住")
    n, block, cw = 22, 8, 26
    for i in range(n):
        x = 40 + i * cw
        f.rect(x, 50, cw - 3, 24, ("blue", "green", "orange")[i // block], rx=3, text=str(i), size=9, sw=0.7)
    for i in range(n, 3 * block):
        x = 40 + i * cw
        f.rect(x, 50, cw - 3, 24, "gray", rx=3, text="×", size=9, sw=0.5)
    for p in range(3):
        x0 = 40 + p * block * cw
        f.rect(x0 - 2, 44, block * cw - 3 + 4, 36, "gray", rx=5, sw=1, dash="4 3")
        f.text(x0 + block * cw / 2, 100, f"program_id = {p}\noffsets = {p} × BLOCK + arange(BLOCK)", cls="mu", size=9)
    f.text(560, 128, "× = 越界，mask = offsets < n 挡住", cls="tx", size=9.5, family="mono")
    f.text(350, 160, "tl.load(ptr + offsets, mask) → 算 → tl.store：线程怎么分、怎么合并访问、怎么用共享内存，编译器决定；程序员只写\"一块\"的逻辑", cls="tx", size=10)
    f.text(350, 184, "矩阵乘就是二维的 program 网格，每个 program 算 C 的一个 BLOCK_M × BLOCK_N，沿 K 循环累加——和 CUDA 的 block tile 一一对应", cls="mu", size=9.5)
    f.text(350, 208, "autotune 在几组 BLOCK 大小、num_warps、num_stages 里挑最快的；融合 softmax、注意力、量化 GEMM 都是几十行", cls="mu", size=9.5)
    return f


@figure("cuda", "ir-lowering")
def ir_lowering():
    f = Fig(700, 250, "编译器的层层下降：图级 IR 做融合与布局，循环级 IR 做分块与交换，最后生成目标代码——每一层只关心自己那层的优化")
    levels = [("图级 IR（算子图）", "blue", "算子融合、常量折叠、布局选择、内存规划"), ("循环级 IR（嵌套循环）", "orange", "分块（tiling）、循环交换、向量化、并行映射"), ("目标 IR（LLVM / PTX）", "purple", "寄存器分配、指令选择、调度"), ("机器码", "green", "SASS / 可执行文件")]
    for i, (name, cls, desc) in enumerate(levels):
        y = 36 + i * 46
        f.rect(60, y, 220, 34, cls, rx=6, text=name, size=10.5)
        f.text(480, y + 17, desc, cls="mu", size=10)
        if i < len(levels) - 1:
            f.arrow(170, y + 34, 170, y + 46, sw=1.2, label="lowering", lx=46, ly=0, lsize=9)
    f.text(350, 232, "MLIR 把\"多层 IR + 各层的 pass\"做成了通用框架：Triton、TVM、IREE 都是这个形状；Inductor 的调度器相当于循环级那一层", cls="mu", size=9.5)
    return f


@figure("cuda", "autograd-graph")
def autograd_graph_cuda():
    return backprop_graph()


@figure("train", "moe-flow")
def moe_flow():
    f = Fig(700, 270, "专家并行的一次前向：路由 → 按目标专家排序 → 交换数量 → dispatch（all-to-all）→ 各卡的专家分组 GEMM → combine（反向的 all-to-all）→ 加权求和")
    steps = [("路由\n每个 token 选 top-k\n专家 + 门控权重", "blue"), ("排序\n按目标专家把\n(token, 专家) 排好", "gray"), ("交换数量\n小 all-to-all：\n各卡要收多少", "gray"), ("dispatch\nall-to-all 把 token\n送到专家所在的卡", "orange"), ("专家计算\n分组 GEMM：每个\n专家算自己那批", "green"), ("combine\n反向 all-to-all 送回，\n按门控权重求和", "orange")]
    for i, (name, cls) in enumerate(steps):
        x = 20 + i * 112
        f.rect(x, 50, 100, 70, cls, rx=7, text=name, size=9)
        if i < len(steps) - 1:
            f.arrow(x + 100, 85, x + 112, 85, sw=1.2)
    f.text(350, 150, "两次 all-to-all 每层都有，每次约 top-k × s·b·h 字节：EP 的通信在机内走 NVLink、跨机走 IB，DeepEP 这类库专门优化它", cls="tx", size=10)
    f.text(350, 174, "负载均衡：路由器偏心时某张卡的专家被挤爆、其他卡空等——辅助损失 E·Σ f_e·P_e 或无辅助损失的偏置调节，加上容量上限和 EPLB 的副本", cls="mu", size=9.5)
    f.text(350, 198, "反向传播正好反过来：combine 的反向是 dispatch、dispatch 的反向是 combine，带 autograd 的 all_to_all_single 自动处理", cls="mu", size=9.5)
    f.text(350, 222, "和其他并行的组合：注意力部分用 DP / TP，专家部分用 EP；专家数远大于卡数时每张卡放多个专家，分组 GEMM 一次算完", cls="mu", size=9.5)
    f.text(350, 246, "推理时同一套流程，只是没有反向；大规模 EP（每卡一个专家）时 all-to-all 的延迟成了 decode 的主要开销", cls="mu", size=9.5)
    return f


@figure("train", "fp8-training")
def fp8_training():
    f = Fig(700, 280, "FP8 训练的数据流：矩阵乘的输入量化成 FP8（前向 E4M3、反向梯度 E5M2），累加和主权重仍在高精度")
    f.rect(30, 50, 120, 44, "gray", rx=6, text="激活 x（bf16）", size=10)
    f.rect(30, 120, 120, 44, "gray", rx=6, text="权重 W（bf16 副本）", size=10)
    f.arrow(150, 72, 200, 72, sw=1.2, label="× 缩放 → E4M3", ly=-10, lsize=9)
    f.arrow(150, 142, 200, 142, sw=1.2, label="× 缩放 → E4M3", ly=-10, lsize=9)
    f.rect(200, 60, 130, 94, "orange", rx=7, text="FP8 GEMM\nTensor Core\n累加在 fp32", size=10)
    f.arrow(330, 107, 380, 107, sw=1.2, label="反量化", ly=-10, lsize=9)
    f.rect(380, 85, 110, 44, "gray", rx=6, text="输出 y（bf16）", size=10)
    f.rect(520, 50, 150, 44, "purple", rx=6, text="反向：梯度用 E5M2\n（范围大、精度低）", size=9.5)
    f.rect(520, 120, 150, 44, "green", rx=6, text="主权重 + 优化器状态\nfp32，从不量化", size=9.5)
    f.text(350, 192, "缩放因子怎么定：逐张量（延迟缩放，用上几步的 amax）简单但一个离群值毁一整张；细粒度（激活 1×128、权重 128×128，DeepSeek-V3）把影响局限在一小块", cls="tx", size=9.5)
    f.text(350, 216, "E4M3 最大 448、相对精度 1/8：数值必须先缩放进范围；E5M2 范围到 57344，给动态范围更大的梯度用", cls="mu", size=9.5)
    f.text(350, 240, "能省的是矩阵乘的时间和激活显存（存 FP8 版本）；归一化、softmax、优化器步骤仍在 fp32 / bf16，所以端到端加速通常 1.2～1.5 倍", cls="mu", size=9.5)
    f.text(350, 264, "FP4 训练再往下走：缩放更细、随机舍入、Hadamard 变换把离群值摊开——每一步都在和\"少数几个大数\"作斗争", cls="mu", size=9.5)
    return f


@figure("train", "tp-mlp")
def tp_mlp_train():
    return tp_mlp()


@figure("cs", "lru-cache")
def lru_cache():
    f = Fig(700, 250, "LRU = 哈希表 + 双向链表：哈希表 O(1) 找到节点，链表 O(1) 把它挪到头部或淘汰尾部")
    f.text(120, 24, "哈希表：key → 节点", cls="tx", size=11.5, weight="600")
    keys = ["k7", "k3", "k9", "k1"]
    for i, k in enumerate(keys):
        f.rect(40, 44 + i * 36, 60, 26, "blue", rx=4, text=k, size=10)
        f.arrow(100, 57 + i * 36, 170 + [2, 0, 3, 1][i] * 120 + 30, 100, cls="ln", sw=1, opacity=0.5)
    f.text(430, 24, "双向链表：最近用过的在头，最久没用的在尾", cls="tx", size=11.5, weight="600")
    f.rect(130, 86, 40, 28, "gray", rx=5, text="头", size=10)
    for i, k in enumerate(["k3", "k1", "k7", "k9"]):
        x = 200 + i * 120
        f.rect(x, 86, 60, 28, "orange" if i == 0 else "green", rx=5, text=k, size=10)
        f.arrow(x - 30 if i == 0 else x - 60, 100, x, 100, sw=1.2, both=True)
    f.rect(680 - 40, 86, 40, 28, "gray", rx=5, text="尾", size=10)
    f.arrow(620, 100, 640, 100, sw=1.2, both=True)
    f.text(230, 140, "get(k3)：命中 → 摘下来插到头部", cls="mu", size=9.5)
    f.text(560, 140, "put 满了：删掉尾部 k9，哈希表里也删", cls="mu", size=9.5)
    f.text(350, 180, "两个结构互相指：哈希表存节点指针，节点存 key（淘汰时反查哈希表）；带虚拟头尾节点，插入删除不用判空", cls="tx", size=10)
    f.text(350, 204, "推理引擎的前缀缓存就是这个形状的放大版：块哈希 → 块，LRU 决定淘汰谁；基数树让\"命中前缀\"和\"淘汰最久没用的叶子\"都 O(前缀长度)", cls="mu", size=9.5)
    f.text(350, 228, "Python 里 OrderedDict.move_to_end 就是\"挪到头\"，popitem(last=False) 就是\"淘汰尾\"；面试要求手写的是下面那个链表版本", cls="mu", size=9.5)
    return f


@figure("cs", "heap-ops")
def heap_ops():
    f = Fig(700, 260, "堆：用数组存的完全二叉树，父节点 i 的孩子是 2i+1 和 2i+2；插入沿路上浮，弹出堆顶后把末尾放到顶上下沉")
    vals = [3, 7, 5, 12, 9, 8, 15]
    pos = {0: (160, 50), 1: (90, 110), 2: (230, 110), 3: (55, 170), 4: (125, 170), 5: (195, 170), 6: (265, 170)}
    for i in range(1, 7):
        (x1, y1), (x2, y2) = pos[(i - 1) // 2], pos[i]
        f.line(x1, y1, x2, y2, cls="ln", sw=1.2)
    for i, v in enumerate(vals):
        x, y = pos[i]
        f.circle(x, y, 16, "blue" if i else "orange", text=str(v), size=11)
        f.text(x, y + 28, f"[{i}]", cls="mu", size=9)
    f.text(160, 222, "数组：[3, 7, 5, 12, 9, 8, 15]", cls="tx", size=10.5, family="mono")
    f.text(160, 244, "最小堆：每个父节点 ≤ 两个孩子；没有指针，缓存友好", cls="mu", size=9.5)
    f.text(500, 40, "push(4)：放到末尾 [7]，和父节点 [3]=12 比，小就交换，\n一路上浮到不比父节点小为止：O(log n)", cls="mu", size=9.5)
    f.text(500, 100, "pop()：取走堆顶 3，把末尾的 15 放到顶上，\n和两个孩子中较小的交换，一路下沉：O(log n)", cls="mu", size=9.5)
    f.text(500, 160, "heapify：从最后一个非叶子节点倒着下沉，O(n)，\n比逐个 push 的 O(n log n) 快", cls="mu", size=9.5)
    f.text(500, 215, "用在哪：Top-K（维护大小为 K 的堆）、合并 K 路有序流、\n按截止时间调度（推理引擎按预计完成时间取请求就是一个堆）", cls="mu", size=9.5)
    return f


@figure("cs", "sm-anatomy")
def sm_anatomy():
    f = Fig(700, 300, "拆开一个 SM（Hopper）：4 个分区各有自己的 warp 调度器、寄存器文件、CUDA Core 和 Tensor Core，共享 L1 / 共享内存")
    f.rect(30, 40, 640, 220, "gray", rx=10, sw=1.2)
    f.text(350, 58, "SM", cls="tx", size=12, weight="600")
    for q in range(4):
        x = 45 + q * 157
        f.rect(x, 72, 145, 120, "blue", rx=7, sw=1)
        f.text(x + 72, 88, f"分区 {q}", cls="tx", size=10, weight="600")
        f.rect(x + 8, 98, 129, 18, "orange", rx=3, text="warp 调度器 + 分发", size=8.5)
        f.rect(x + 8, 120, 129, 18, "purple", rx=3, text="寄存器文件 64 KB", size=8.5)
        f.rect(x + 8, 142, 60, 18, "green", rx=3, text="32 个 FP32", size=8)
        f.rect(x + 76, 142, 61, 18, "green", rx=3, text="Tensor Core", size=8)
        f.rect(x + 8, 164, 129, 18, "gray", rx=3, text="LD/ST、SFU", size=8.5)
    f.rect(45, 204, 625, 22, "orange", rx=5, text="L1 数据缓存 / 共享内存：256 KB，可配置划分", size=9.5)
    f.rect(45, 232, 625, 20, "gray", rx=5, text="指令缓存、常量缓存、纹理单元", size=9)
    f.text(350, 278, "每个调度器每周期发一条 warp 指令：4 个分区同时发 4 个 warp；一个 SM 最多驻留 64 个 warp（2048 线程），靠在它们之间切换来掩盖访存延迟", cls="mu", size=9.5)
    return f


@figure("cs", "process-vs-thread")
def process_vs_thread():
    f = Fig(700, 275, "进程和线程各自拥有什么：线程共享地址空间，只有自己的栈和寄存器；进程之间什么都不共享，要靠 IPC")
    for k, (x0, title, cls) in enumerate(((40, "进程 A", "blue"), (380, "进程 B", "orange"))):
        f.rect(x0, 40, 280, 180, cls, rx=10, sw=1.2)
        f.text(x0 + 140, 58, title, cls="tx", size=12, weight="600")
        f.rect(x0 + 14, 72, 252, 22, "gray", rx=4, text="地址空间：代码、全局变量、堆、文件描述符、显存上下文", size=8.5)
        for t in range(3):
            tx = x0 + 14 + t * 84
            f.rect(tx, 104, 76, 100, "green", rx=6, sw=1)
            f.text(tx + 38, 120, f"线程 {t}", cls="tx", size=9.5, weight="600")
            f.rect(tx + 8, 130, 60, 18, "purple", rx=3, text="栈", size=8.5)
            f.rect(tx + 8, 152, 60, 18, "purple", rx=3, text="寄存器 / PC", size=8)
            f.rect(tx + 8, 174, 60, 18, "purple", rx=3, text="TLS", size=8.5)
    f.arrow(320, 130, 380, 130, sw=1.3, both=True)
    f.text(350, 236, "进程之间什么都不共享，靠 IPC 交换数据：管道、socket、共享内存、CUDA IPC（共享显存句柄）", cls="tx", size=10)
    f.text(350, 258, "线程切换只换栈和寄存器（几微秒内）；进程切换还要换页表、冲掉 TLB。fork 复制整个地址空间，CUDA 上下文不能跨 fork，所以 GPU 程序的子进程要用 spawn", cls="mu", size=9.5)
    return f


@figure("cs", "write-path")
def write_path():
    f = Fig(700, 270, "一次 write 的旅程：用户缓冲区 → 页缓存 → 块层 → 设备；O_DIRECT 绕过页缓存，mmap 把文件直接映射进地址空间")
    steps = [("用户缓冲区\n（进程内存）", "gray"), ("系统调用 write\n拷贝进内核", "blue"), ("页缓存\n（脏页，延迟回写）", "orange"), ("块层：合并、排队\nI/O 调度", "purple"), ("设备驱动 → NVMe\nDMA 写盘", "green")]
    for i, (name, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 40, 120, 56, cls, rx=7, text=name, size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 120, 68, x + 136, 68, sw=1.3)
    f.rect(20, 130, 120, 44, "gray", rx=7, text="用户缓冲区\n（对齐的）", size=9.5)
    f.arrow(140, 152, 428, 152, cls="orange-l", hcls="orange-s", sw=1.6, label="O_DIRECT：跳过页缓存，直接 DMA（数据库、模型文件加载常用）", ly=14, lsize=9)
    f.arrow(428, 152, 564, 100, cls="orange-l", hcls="orange-s", sw=1.6)
    f.text(350, 200, "write 返回只代表数据进了页缓存；fsync 才等它真正落盘。读也经过页缓存：第二次读同一个文件快得多，是因为还在内存里", cls="tx", size=10)
    f.text(350, 224, "mmap：把文件页映射进地址空间，读写就是访问内存，缺页时内核从页缓存填——safetensors 加载权重用的就是它，零拷贝、按需读", cls="mu", size=9.5)
    f.text(350, 248, "模型加载慢，先查哪一层：是盘的带宽（NVMe 几 GB/s）、是页缓存没命中、还是 Python 在一个个小文件地 open", cls="mu", size=9.5)
    return f


@figure("cs", "pinned-dma")
def pinned_dma():
    f = Fig(700, 250, "可分页内存要先拷到锁页的中转区再 DMA；锁页内存直接 DMA；GPUDirect 让网卡 / NVMe 直接读写显存")
    f.text(130, 24, "可分页内存（默认的 malloc）", cls="tx", size=11, weight="600")
    f.rect(30, 40, 90, 40, "gray", rx=6, text="用户内存\n（可换出）", size=9)
    f.arrow(120, 60, 150, 60, sw=1.2, label="CPU 拷贝", ly=-9, lsize=8.5)
    f.rect(150, 40, 90, 40, "orange", rx=6, text="锁页中转区", size=9)
    f.arrow(240, 60, 270, 60, sw=1.2, label="DMA", ly=-9, lsize=8.5)
    f.rect(270, 40, 70, 40, "blue", rx=6, text="显存", size=9.5)
    f.text(185, 100, "两步、CPU 参与、cudaMemcpyAsync 退化成同步；带宽掉一半以上", cls="mu", size=9)

    f.text(130, 140, "锁页内存（cudaHostAlloc / pin_memory）", cls="tx", size=11, weight="600")
    f.rect(30, 156, 150, 40, "orange", rx=6, text="锁页内存\n（不会被换出）", size=9)
    f.arrow(180, 176, 270, 176, sw=1.4, label="DMA，PCIe 全速，异步", ly=-9, lsize=8.5)
    f.rect(270, 156, 70, 40, "blue", rx=6, text="显存", size=9.5)
    f.text(185, 216, "DataLoader 的 pin_memory=True、权重预取、KV 卸载都靠它", cls="mu", size=9)

    f.text(520, 24, "GPUDirect", cls="tx", size=11, weight="600")
    f.rect(420, 40, 80, 40, "green", rx=6, text="网卡 / NVMe", size=9)
    f.arrow(500, 60, 590, 60, sw=1.4, label="直接 DMA", ly=-9, lsize=8.5)
    f.rect(590, 40, 80, 40, "blue", rx=6, text="显存", size=9.5)
    f.text(545, 100, "不经过主机内存：PD 分离传 KV、\n从存储直接加载权重", cls="mu", size=9)
    f.text(520, 140, "NUMA：内存也分远近", cls="tx", size=11, weight="600")
    f.rect(420, 156, 110, 40, "gray", rx=6, text="CPU0 + 本地内存\n+ GPU0-3", size=8.5)
    f.rect(560, 156, 110, 40, "gray", rx=6, text="CPU1 + 本地内存\n+ GPU4-7", size=8.5)
    f.arrow(530, 176, 560, 176, sw=1.2, both=True)
    f.text(545, 216, "跨节点访问慢几成：进程要绑在\n自己 GPU 所在的 NUMA 节点上", cls="mu", size=9)
    return f


@figure("cs", "namespaces-cgroups")
def namespaces_cgroups():
    f = Fig(700, 240, "容器 = namespace（看得到什么）+ cgroup（能用多少）+ 挂进来的设备；没有虚拟机那层，内核是同一个")
    f.rect(30, 40, 300, 150, "blue", rx=10, sw=1.2)
    f.text(180, 58, "namespace：隔离视图", cls="tx", size=11.5, weight="600")
    for i, name in enumerate(("pid：只看到自己的进程", "net：自己的网卡、端口", "mnt：自己的文件系统树", "ipc / uts / user")):
        f.rect(45, 70 + i * 28, 270, 22, "gray", rx=4, text=name, size=9.5)
    f.rect(370, 40, 300, 150, "orange", rx=10, sw=1.2)
    f.text(520, 58, "cgroup：限制资源", cls="tx", size=11.5, weight="600")
    for i, name in enumerate(("cpu：配额 + 周期 → 超了就节流", "memory：上限，超了 OOM kill", "io / pids", "devices：允许哪些设备文件")):
        f.rect(385, 70 + i * 28, 270, 22, "gray", rx=4, text=name, size=9.5)
    f.text(350, 212, "GPU 进容器：把 /dev/nvidia* 和驱动库挂进去（nvidia-container-toolkit），RDMA 网卡同理；/dev/shm 默认只有 64 MB，多进程共享张量要放大", cls="mu", size=9.5)
    f.text(350, 232, "常见的坑：容器里 os.cpu_count() 看到的是宿主机的核数，线程池开太大被 cgroup 节流；内存上限不含页缓存却含 /dev/shm", cls="mu", size=9.5)
    return f


@figure("cs", "ring-buffer")
def ring_buffer():
    f = Fig(700, 230, "共享内存环形缓冲区：生产者写 head、消费者读 tail，一块内存反复用，不拷贝、不走内核")
    n = 12
    cx, cy, r = 160, 115, 70
    for i in range(n):
        a0 = -math.pi / 2 + i * 2 * math.pi / n
        a1 = a0 + 2 * math.pi / n
        filled = 3 <= i < 8
        x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
        x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
        xi0, yi0 = cx + 40 * math.cos(a0), cy + 40 * math.sin(a0)
        xi1, yi1 = cx + 40 * math.cos(a1), cy + 40 * math.sin(a1)
        f.path(f"M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f} L {xi1:.1f} {yi1:.1f} A 40 40 0 0 0 {xi0:.1f} {yi0:.1f} Z", cls="orange" if filled else "gray", sw=0.8)
    f.text(cx, cy, "环", cls="tx", size=11)
    f.text(cx + 95, cy - 55, "head（生产者写）", cls="mu", size=9.5, anchor="start")
    f.text(cx + 70, cy + 70, "tail（消费者读）", cls="mu", size=9.5, anchor="start")
    f.text(480, 50, "满 = (head + 1) % n == tail；空 = head == tail", cls="tx", size=10, family="mono")
    f.text(480, 80, "单生产者单消费者只要原子读写两个下标，\n不需要锁；多生产者要 CAS 抢位置", cls="mu", size=9.5)
    f.text(480, 125, "推理引擎里：调度进程把 batch 元数据写进共享内存，\nGPU 工作进程直接读；大张量走 CUDA IPC 共享显存句柄", cls="mu", size=9.5)
    f.text(480, 170, "管道 / socket 每条消息要两次内核拷贝 + 一次唤醒，\n几十 µs；共享内存一次写一次读，亚微秒", cls="mu", size=9.5)
    f.text(350, 215, "ZMQ 是消息总线（跨机器也能用），共享内存是同机的快速路；两者在 vLLM / SGLang 里都在用", cls="mu", size=9.5)
    return f


@figure("cs", "sse-stream")
def sse_stream():
    f = Fig(700, 240, "流式输出：每生成一个 token 就作为一个 SSE 事件写出去——中间任何一层的缓冲都会把\"逐字出现\"变成\"一次全出\"")
    steps = [("推理引擎\n每步一个 token", "blue"), ("API 服务\n序列化成 data: {...}", "green"), ("反向代理\nnginx / 网关", "orange"), ("客户端\n按事件边界解析", "gray")]
    for i, (name, cls) in enumerate(steps):
        x = 30 + i * 170
        f.rect(x, 50, 140, 56, cls, rx=7, text=name, size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 140, 78, x + 170, 78, sw=1.3, label="chunk", ly=-9, lsize=8.5)
    f.text(350, 130, "三个常见的缓冲点：框架的响应缓冲（要用流式响应、每条 flush）、代理的 proxy_buffering（要关）、客户端一次性 read 整个 body", cls="tx", size=10)
    f.text(350, 154, "HTTP/1.1 用分块传输（chunked），HTTP/2 用帧，gRPC 用服务端流；SSE 是纯文本协议，\"data: ...\\n\\n\" 一行一个事件，最省事", cls="mu", size=9.5)
    f.text(350, 178, "工程细节：首 token 延迟从客户端看是\"连接 + 排队 + prefill\"；断连要能取消请求、释放 KV；心跳事件防止中间设备超时", cls="mu", size=9.5)
    f.text(350, 202, "吞吐和延迟的矛盾：每个 token 一个 chunk 开销大，攒几个再发延迟高；多数引擎按步发、不攒", cls="mu", size=9.5)
    return f


@figure("cs", "raft-replication")
def raft_replication():
    f = Fig(700, 280, "Raft：多数派选出领导者，领导者把日志复制到多数派才算提交；法定人数的读写交集保证不会读到旧值")
    f.rect(280, 40, 140, 40, "orange", rx=7, text="领导者（任期 3）", size=10.5)
    for i, (x, state) in enumerate(((60, "跟随者"), (170, "跟随者"), (530, "跟随者"), (640, "跟随者（落后）"))):
        f.rect(x - 45, 120, 90, 36, "blue" if i < 3 else "gray", rx=6, text=state, size=9.5)
        f.arrow(350, 80, x, 120, sw=1.1, label="AppendEntries\n（心跳 + 日志）" if i == 0 else None, lx=-110, ly=-6, lsize=8.5)
    for i, (x, log) in enumerate(((60, "1 2 3 4"), (170, "1 2 3 4"), (350, "1 2 3 4"), (530, "1 2 3 4"), (640, "1 2 3"))):
        f.text(x, 176, log, cls="tx", size=10, family="mono")
    f.text(350, 100, "日志 [1 2 3 4]，提交到 4", cls="mu", size=9.5)
    f.text(350, 206, "写：领导者追加日志 → 发给所有人 → 收到多数派（3/5）确认就提交并回复客户端；一个落后的跟随者不影响可用性", cls="tx", size=10)
    f.text(350, 230, "选举：跟随者超时没收到心跳就发起投票，拿到多数派票的成为新领导者，任期号加一；旧领导者看到更大的任期就退位——这就是防脑裂", cls="mu", size=9.5)
    f.text(350, 254, "推理集群里：路由表、实例注册、PD 配比这些元数据放 etcd（Raft），数据面（KV 传输、请求）不走共识——共识只给小而关键的状态", cls="mu", size=9.5)
    return f


@figure("cs", "use-method")
def use_method():
    f = Fig(700, 200, "USE 方法：对每一种资源问三个问题——利用率（Utilization）、饱和度（Saturation）、错误（Errors）")
    cols = ["资源", "利用率", "饱和度", "错误"]
    rows = [("CPU", "top / mpstat 的 %user", "运行队列长度（load）", "—"), ("内存", "free、页缓存占比", "swap、回收活动", "OOM kill"), ("磁盘", "iostat %util", "队列深度、await", "I/O 错误"), ("网络", "带宽占比", "丢包、重传", "错误计数"), ("GPU", "nvidia-smi 的 util（粗）", "等待的 kernel、队列", "Xid 错误")]
    for j, c in enumerate(cols):
        f.rect(30 + j * 165, 40, 160, 24, "blue", rx=4, text=c, size=10.5)
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            f.rect(30 + j * 165, 66 + i * 24, 160, 22, "gray" if j else "orange", rx=3, text=c, size=9, sw=0.6)
    f.text(350, 194, "先问\"哪种资源饱和了\"再找工具；GPU 利用率 100% 不等于算满了（它只表示有 kernel 在跑），要看 SM 占用和屋顶线", cls="mu", size=9.5)
    return f


@figure("cs", "radix-tree")
def radix_tree_cs():
    return radix_tree()


@figure("cs", "memory-hierarchy")
def memory_hierarchy_cs():
    return memory_hierarchy()


@figure("cs", "server-topology")
def server_topology_cs():
    return server_topology()


@figure("cpp", "object-lifetime")
def object_lifetime():
    f = Fig(700, 250, "RAII：资源的生命周期绑在栈对象上，离开作用域（正常返回或异常展开）都会调用析构，释放顺序和构造相反")
    f.rect(30, 40, 300, 170, "gray", rx=10, sw=1, dash="5 4")
    f.text(180, 58, "函数作用域", cls="tx", size=11.5, weight="600")
    items = [("std::ifstream f(...)", "构造：打开文件", "green"), ("std::lock_guard g(m)", "构造：加锁", "blue"), ("auto buf = make_unique<...>", "构造：分配显存 / 内存", "orange")]
    for i, (name, what, cls) in enumerate(items):
        f.rect(45, 72 + i * 42, 160, 30, cls, rx=5, text=name, size=8.5)
        f.text(265, 87 + i * 42, what, cls="mu", size=9.5)
    f.text(180, 200, "return 或 throw ……", cls="mu", size=10)
    f.arrow(330, 125, 380, 125, sw=1.4, label="离开作用域", ly=-10, lsize=9.5)
    f.rect(380, 40, 290, 170, "gray", rx=10, sw=1, dash="5 4")
    f.text(525, 58, "析构：和构造相反的顺序", cls="tx", size=11.5, weight="600")
    for i, (name, what, cls) in enumerate(reversed(items)):
        f.rect(395, 72 + i * 42, 160, 30, cls, rx=5, text=name.split("(")[0].replace("auto buf = ", "") + " 析构", size=8.5)
        f.text(615, 87 + i * 42, what.replace("构造：", "").replace("打开", "关闭").replace("加锁", "解锁").replace("分配", "释放"), cls="mu", size=9.5)
    f.text(525, 200, "异常展开时同样执行：不会漏掉", cls="mu", size=10)
    f.text(350, 236, "所以不写 close / unlock / free：每种资源包成一个类，拷贝要么禁掉、要么写清楚（零法则 / 五法则），作用域守卫处理\"离开时做什么\"的临时需求", cls="mu", size=9.5)
    return f


@figure("cpp", "ownership-kinds")
def ownership_kinds():
    f = Fig(700, 250, "三种所有权：unique_ptr 独占（只能移动）；shared_ptr 共享（控制块计数）；weak_ptr 观察（不计数，打破循环）")
    f.text(120, 24, "unique_ptr<T>", cls="tx", size=12, weight="600")
    f.rect(40, 40, 90, 34, "blue", rx=6, text="owner", size=10)
    f.arrow(130, 57, 190, 57, sw=1.4)
    f.rect(190, 40, 60, 34, "gray", rx=6, text="T", size=10)
    f.text(145, 100, "只有一个指针指向它；\n拷贝被禁止，std::move 转交；\n离开作用域就 delete；零开销", cls="mu", size=9.5)
    f.text(440, 24, "shared_ptr<T> + weak_ptr<T>", cls="tx", size=12, weight="600")
    f.rect(330, 40, 70, 34, "orange", rx=6, text="A 持有", size=9.5)
    f.rect(330, 90, 70, 34, "orange", rx=6, text="B 持有", size=9.5)
    f.rect(450, 56, 110, 50, "purple", rx=6, text="控制块\nstrong = 2，weak = 1", size=9)
    f.arrow(400, 57, 450, 70, sw=1.3)
    f.arrow(400, 107, 450, 92, sw=1.3)
    f.arrow(560, 81, 600, 81, sw=1.3)
    f.rect(600, 64, 60, 34, "gray", rx=6, text="T", size=10)
    f.rect(330, 140, 70, 34, "green", rx=6, text="weak 观察", size=9)
    f.arrow(400, 157, 450, 106, sw=1.2, dash="4 3")
    f.text(500, 150, "strong 归零才 delete T；\nweak 只能 lock() 后临时升级", cls="mu", size=9.5)
    f.text(350, 206, "循环引用：A 和 B 互相 shared_ptr，strong 永远不归零——把其中一条边改成 weak_ptr（父 → 子 shared，子 → 父 weak）", cls="tx", size=10)
    f.text(350, 230, "函数参数：只是用一下就传 T& / const T& / T*；要拿走所有权才传 unique_ptr 按值；要共享所有权才传 shared_ptr（每次拷贝都是一次原子计数）", cls="mu", size=9.5)
    return f


@figure("cpp", "move-vs-copy")
def move_vs_copy():
    f = Fig(700, 230, "拷贝要分配新 buffer 并逐个复制；移动只是把指针接过来，再把源对象置空——源对象之后只保证\"可析构、可赋值\"")
    for k, (x0, title, cls) in enumerate(((40, "拷贝 b = a", "orange"), (380, "移动 b = std::move(a)", "blue"))):
        f.text(x0 + 140, 24, title, cls="tx", size=12, weight="600")
        f.rect(x0, 44, 80, 34, "gray", rx=6, text="a：ptr, len", size=9.5)
        f.rect(x0, 110, 80, 34, "gray", rx=6, text="b：ptr, len", size=9.5)
        f.rect(x0 + 160, 44, 120, 34, cls, rx=6, text="buffer（堆上，很大）", size=9)
        f.arrow(x0 + 80, 61, x0 + 160, 61, sw=1.3)
        if k == 0:
            f.rect(x0 + 160, 110, 120, 34, cls, rx=6, text="新 buffer（复制一份）", size=9)
            f.arrow(x0 + 80, 127, x0 + 160, 127, sw=1.3)
            f.text(x0 + 140, 170, "O(n) 的内存分配 + 复制", cls="mu", size=10)
        else:
            f.arrow(x0 + 80, 127, x0 + 160, 78, sw=1.3)
            f.text(x0 + 40, 92, "a.ptr = nullptr", cls="mu", size=9, family="mono")
            f.text(x0 + 140, 170, "O(1)：搬几个指针，源对象置空", cls="mu", size=10)
    f.text(350, 205, "右值引用 T&& 就是\"允许被掏空的对象\"的标记：临时对象天然是右值，具名变量要显式 std::move；完美转发 forward<T> 保留实参本来的左右值性", cls="mu", size=9.5)
    return f


@figure("cpp", "dispatch-compile-time")
def dispatch_compile_time():
    f = Fig(700, 230, "运行时参数变成编译期常量：按值 switch 一次，之后每个分支里的模板实例都把它当常量，循环展开、分支消失")
    f.rect(30, 70, 120, 44, "gray", rx=7, text="运行时的 head_dim\n（64 / 128 / 256）", size=9.5)
    f.arrow(150, 92, 200, 92, sw=1.3, label="switch", ly=-9, lsize=9.5)
    f.rect(200, 40, 120, 30, "blue", rx=6, text="kernel<64>", size=10)
    f.rect(200, 77, 120, 30, "blue", rx=6, text="kernel<128>", size=10)
    f.rect(200, 114, 120, 30, "blue", rx=6, text="kernel<256>", size=10)
    for y in (55, 92, 129):
        f.arrow(320, y, 370, y, sw=1.1)
    f.rect(370, 40, 300, 104, "orange", rx=8, text="每个实例里 head_dim 是 constexpr：\n数组大小、循环次数、寄存器分配都在编译期定下来\nif constexpr 砍掉不适用的分支", size=9.5)
    f.text(350, 170, "代价：每个取值一份代码（编译时间、二进制体积），取值集合要有限——这就是推理 kernel 只支持几个 head_dim 的原因", cls="tx", size=10)
    f.text(350, 194, "模板 vs 虚函数：模板在编译期派发、能内联、零开销但代码膨胀；虚函数在运行时派发、一个实现、多一次间接调用", cls="mu", size=9.5)
    f.text(350, 216, "concepts 给模板参数写\"接口\"，错误信息从几百行变成一句\"不满足 X\"", cls="mu", size=9.5)
    return f


@figure("cpp", "build-pipeline")
def build_pipeline():
    f = Fig(700, 210, "从源码到可执行文件：每个 .cpp 单独预处理、编译、汇编成目标文件，最后链接——头文件里的东西会被复制进每个翻译单元")
    steps = [("a.cpp + 头文件", "gray", "预处理：展开 #include、宏"), ("翻译单元", "blue", "编译：语法树 → IR → 汇编"), ("a.o", "orange", "汇编：机器码 + 符号表"), ("可执行 / .so", "green", "链接：解析符号、合并段")]
    for i, (name, cls, desc) in enumerate(steps):
        x = 30 + i * 170
        f.rect(x, 50, 130, 40, cls, rx=7, text=name, size=10.5)
        f.text(x + 65, 112, desc, cls="mu", size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 130, 70, x + 170, 70, sw=1.3)
    f.text(350, 150, "单一定义规则：函数在多个翻译单元里各有一份定义就链接报错；头文件里的函数要 inline（模板和 constexpr 隐含 inline）", cls="tx", size=10)
    f.text(350, 174, "名字修饰：C++ 把参数类型编进符号名以支持重载；extern \"C\" 关掉它，才能被 C / Python 的 ctypes 找到", cls="mu", size=9.5)
    f.text(350, 196, "未定义行为不是\"会崩\"而是\"编译器可以假设它不发生\"：有符号溢出、悬垂引用、严格别名——开 sanitizer 当默认", cls="mu", size=9.5)
    return f


@figure("cpp", "vector-vs-list")
def vector_vs_list():
    f = Fig(700, 220, "vector 的元素连续，一次缓存行装进好几个；list 的节点散落在堆上，每走一步都是一次缓存未命中——遍历慢一个数量级")
    f.text(170, 24, "vector<int>", cls="tx", size=12, weight="600")
    for i in range(12):
        f.rect(40 + i * 24, 40, 22, 22, "blue", rx=2, text=str(i), size=8.5, sw=0.6)
    f.rect(38, 36, 16 * 12, 30, "orange", rx=4, sw=1.2)
    f.text(170, 86, "一个 64 字节的缓存行装 16 个 int；预取器看懂顺序访问，提前把下一行拉来", cls="mu", size=9.5)
    f.text(520, 24, "list<int>", cls="tx", size=12, weight="600")
    pts = [(400, 50), (520, 40), (470, 90), (620, 60), (560, 100), (650, 100)]
    for i, (x, y) in enumerate(pts):
        f.rect(x - 24, y - 10, 48, 20, "gray", rx=3, text=f"{i}  →", size=8.5, sw=0.6)
        if i < len(pts) - 1:
            (x2, y2) = pts[i + 1]
            f.arrow(x + 24, y, x2 - 24, y2, sw=1, opacity=0.6)
    f.text(520, 134, "每个节点单独 new，地址随机；每走一步等一次内存（~100 ns）", cls="mu", size=9.5)
    f.text(350, 170, "默认用 vector：末尾插入摊还 O(1)，中间插入 O(n) 但常数极小，遍历和二分都快；list 只在\"持有迭代器且频繁中间插删\"时才赢", cls="tx", size=10)
    f.text(350, 194, "同理：unordered_map 的桶是链表，节点分散；小表用排序过的 vector + 二分，或开放寻址的哈希（absl::flat_hash_map）更快", cls="mu", size=9.5)
    return f


@figure("cpp", "arena-bump")
def arena_bump():
    f = Fig(700, 200, "arena：一大块内存，一个指针往前推就是分配，不单独释放，用完整块一起扔——分配 O(1)、没有碎片、没有锁")
    f.rect(40, 50, 620, 40, "gray", rx=6, sw=1.2)
    widths = [90, 140, 60, 110]
    x = 42
    for i, w in enumerate(widths):
        f.rect(x, 52, w - 3, 36, ("blue", "green", "orange", "purple")[i], rx=4, text=f"对象 {i}", size=9.5, sw=0.8)
        x += w
    f.arrow(x + 2, 110, x + 2, 92, sw=1.4, label="bump 指针：下一次分配从这里开始", lx=140, ly=14, lsize=9.5)
    f.text(560, 72, "空闲", cls="mu", size=10)
    f.text(350, 150, "一次请求的生命周期内（解析 → 调度 → 采样）所有小对象都从 arena 里拿，请求结束整块归还；对齐靠把指针向上取整", cls="tx", size=10)
    f.text(350, 176, "块分配器把内存切成固定大小的块、空闲链表管理——分页 KV Cache 就是它；缓存分配器释放了也不还给系统，PyTorch 的显存池就是这种", cls="mu", size=9.5)
    return f


@figure("cpp", "deadlock-order")
def deadlock_order():
    f = Fig(700, 220, "死锁的标准姿势：两个线程以相反的顺序拿两把锁，各持一把、等另一把——按固定的全局顺序加锁，或者 std::scoped_lock 一次锁两把")
    for k, (x0, name, first, second, cls) in enumerate(((40, "线程 A", "锁 1", "锁 2", "blue"), (380, "线程 B", "锁 2", "锁 1", "orange"))):
        f.text(x0 + 140, 24, name, cls="tx", size=12, weight="600")
        f.rect(x0, 40, 120, 30, cls, rx=6, text=f"① 拿到 {first}", size=10)
        f.arrow(x0 + 60, 70, x0 + 60, 94, sw=1.2)
        f.rect(x0, 94, 120, 30, "gray", rx=6, text=f"② 等 {second} ……", size=10)
        f.rect(x0 + 160, 40, 120, 30, "red", rx=6, text=f"{second} 被对方持有", size=9.5)
    f.arrow(160, 109, 380, 55, cls="red-l", hcls="red-s", sw=1.4, dash="5 3")
    f.arrow(500, 109, 160, 55, cls="red-l", hcls="red-s", sw=1.4, dash="5 3")
    f.text(350, 150, "四个条件缺一不可：互斥、持有并等待、不可抢占、循环等待——打破\"循环等待\"最容易：所有线程按同一个顺序加锁", cls="tx", size=10)
    f.text(350, 174, "std::scoped_lock(m1, m2) 用避免死锁的算法同时锁两把；条件变量的 wait 要放在循环里检查谓词（虚假唤醒、多消费者）", cls="mu", size=9.5)
    f.text(350, 196, "锁住的区域越小越好：拷贝出数据再处理，别在持锁时做 I/O 或等别的锁；读多写少用 shared_mutex", cls="mu", size=9.5)
    return f


@figure("cpp", "thread-pool")
def thread_pool():
    f = Fig(700, 220, "线程池：任务进队列，固定数量的工作线程循环取任务——条件变量唤醒、停止标志退出、future 取回结果")
    f.rect(30, 70, 110, 40, "gray", rx=7, text="submit(task)\n→ future", size=9.5)
    f.arrow(140, 90, 190, 90, sw=1.3)
    f.rect(190, 60, 180, 60, "orange", rx=7, sw=1.2)
    f.text(280, 76, "任务队列（互斥锁保护）", cls="tx", size=10)
    for i in range(5):
        f.rect(200 + i * 33, 90, 28, 22, "gray", rx=3, text=f"t{i}", size=8.5, sw=0.6)
    f.arrow(370, 90, 420, 90, sw=1.3, label="notify_one", ly=-9, lsize=9)
    for i in range(3):
        f.rect(420, 44 + i * 36, 110, 28, "blue", rx=5, text=f"worker {i}：取 → 执行", size=9)
    f.text(600, 60, "while (!stop)\n  wait(有任务 || stop)\n  pop → run", cls="mu", size=9, family="mono")
    f.text(350, 150, "坑：析构时要先置 stop 再 notify_all 再 join，否则工作线程永远等在条件变量上；任务抛的异常要存进 future，别让线程死掉", cls="tx", size=10)
    f.text(350, 174, "任务里再 submit 并等待会把池子等死（所有 worker 都在等）；线程数按 CPU 核数定，I/O 多的任务可以多一些", cls="mu", size=9.5)
    f.text(350, 196, "推理引擎里：分词、反分词、HTTP 处理在线程池里，GPU 工作在单独的进程里（CUDA 上下文不喜欢多线程乱发 kernel）", cls="mu", size=9.5)
    return f


@figure("cpp", "pybind-boundary")
def pybind_boundary():
    f = Fig(700, 230, "pybind11 的边界：Python 对象和 C++ 对象之间要转换；大数组不拷贝只传指针；耗时的 C++ 代码要释放 GIL")
    f.rect(30, 50, 180, 110, "orange", rx=9, sw=1.2)
    f.text(120, 68, "Python", cls="tx", size=12, weight="600")
    f.rect(45, 80, 150, 24, "gray", rx=4, text="torch.Tensor / numpy", size=9.5)
    f.rect(45, 110, 150, 24, "gray", rx=4, text="int / str / list", size=9.5)
    f.rect(490, 50, 180, 110, "blue", rx=9, sw=1.2)
    f.text(580, 68, "C++", cls="tx", size=12, weight="600")
    f.rect(505, 80, 150, 24, "gray", rx=4, text="float* + shape（零拷贝）", size=9.5)
    f.rect(505, 110, 150, 24, "gray", rx=4, text="int / std::string / vector", size=9.5)
    f.rect(260, 60, 180, 90, "green", rx=8, text="pybind11 包装层\n类型转换（小对象拷贝）\n引用计数、异常翻译\ngil_scoped_release", size=9.5)
    f.arrow(210, 92, 260, 92, sw=1.3)
    f.arrow(440, 92, 490, 92, sw=1.3)
    f.arrow(490, 122, 440, 122, sw=1.3)
    f.arrow(260, 122, 210, 122, sw=1.3)
    f.text(350, 184, "边界上的三件事：谁拥有内存（返回值策略）、异常怎么过去（C++ 异常翻译成 Python 异常）、GIL 什么时候放（进 C++ 长计算前释放，回 Python 前拿回）", cls="tx", size=10)
    f.text(350, 208, "PyTorch 扩展走同一条路：拿 tensor.data_ptr()，在 C++ / CUDA 里算，返回新 tensor；一次调用几微秒的固定开销，所以别在 C++ 里只做一个加法", cls="mu", size=9.5)
    return f


@figure("cpp", "paged-kv")
def paged_kv_cpp():
    return paged_kv()


@figure("cpp", "ring-buffer")
def ring_buffer_cpp():
    return ring_buffer()


@figure("python", "mro-diamond")
def mro_diamond():
    f = Fig(700, 240, "菱形继承的方法解析顺序（C3 线性化）：D → B → C → A → object，每个类只出现一次，子类永远排在父类前面")
    pos = {"A": (350, 50), "B": (250, 110), "C": (450, 110), "D": (350, 170)}
    for a, b in (("B", "A"), ("C", "A"), ("D", "B"), ("D", "C")):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        f.arrow(x1, y1 - 16, x2, y2 + 16, sw=1.2)
    for name, (x, y) in pos.items():
        f.circle(x, y, 18, "orange" if name == "D" else "blue", text=name, size=12)
    f.text(120, 100, "class B(A): ...\nclass C(A): ...\nclass D(B, C): ...", cls="tx", size=10.5, family="mono")
    f.text(580, 90, "D.__mro__ =\n(D, B, C, A, object)", cls="tx", size=10.5, family="mono")
    f.text(580, 140, "super() 不是\"父类\"，是\"MRO 里的下一个\"：\nB 里的 super().f() 会调到 C.f()，再到 A.f()", cls="mu", size=9.5)
    f.text(350, 214, "所以协作式多继承要求每个类都调 super().__init__(**kwargs) 并把参数往后传；mixin 放在左边、基类放在右边", cls="mu", size=9.5)
    return f


@figure("python", "decorator-wrap")
def decorator_wrap():
    f = Fig(700, 230, "装饰器就是 f = deco(f)：调用 f 时先进包装函数，包装函数再调原函数；叠加时离函数最近的先包、最外层的先执行")
    f.text(130, 24, "@timer\n@retry\ndef f(): ...", cls="tx", size=10.5, family="mono")
    f.text(130, 72, "等价于 f = timer(retry(f))", cls="mu", size=9.5)
    f.rect(300, 40, 360, 130, "blue", rx=10, sw=1.2)
    f.text(480, 58, "timer 的包装函数（最外层，先执行）", cls="tx", size=10)
    f.rect(320, 70, 320, 85, "orange", rx=8, sw=1.2)
    f.text(480, 88, "retry 的包装函数", cls="tx", size=10)
    f.rect(340, 100, 280, 42, "green", rx=6, text="原函数 f\n（functools.wraps 把 __name__、__doc__ 搬到外层）", size=9)
    f.arrow(230, 105, 300, 105, sw=1.4, label="调用 f(...)", ly=-10, lsize=9.5)
    f.text(350, 195, "带参数的装饰器多一层：@retry(times=3) 先调用 retry(times=3) 得到真正的装饰器，再去包函数；用类实现时 __call__ 就是包装函数", cls="mu", size=9.5)
    f.text(350, 216, "装饰器只在定义时跑一次，包装函数在每次调用时跑——别在包装函数里做重活", cls="mu", size=9.5)
    return f


@figure("python", "legb-scope")
def legb_scope():
    f = Fig(700, 230, "名字查找的顺序 LEGB：局部 → 外层函数 → 全局 → 内置；闭包把外层变量装进 cell，函数离开定义处后仍能访问")
    layers = [("Builtins：len、print", "gray", 640), ("Global：模块里的名字", "purple", 540), ("Enclosing：外层函数的局部变量（闭包 cell）", "orange", 440), ("Local：当前函数的局部变量", "blue", 340)]
    for i, (name, cls, w) in enumerate(layers):
        f.rect(350 - w / 2, 36 + i * 30, w, 24, cls, rx=6, text=name, size=10)
    f.arrow(350, 165, 350, 150, sw=1.2)
    f.text(350, 178, "查找从最里层开始，找不到往外一层；赋值默认创建局部名字（要改外层的用 nonlocal，改全局的用 global）", cls="tx", size=10)
    f.text(350, 202, "闭包：内层函数引用了外层变量，Python 把这些变量放进 cell，函数对象的 __closure__ 指着它们——装饰器、回调、工厂函数都靠这个", cls="mu", size=9.5)
    f.text(350, 222, "经典的坑：循环里定义的 lambda 都引用同一个 cell，循环结束后全是最后一个值；用默认参数 i=i 固定下来", cls="mu", size=9.5)
    return f


@figure("python", "optimize-order")
def optimize_order():
    f = Fig(700, 200, "优化的顺序：先测量再动手——找到真正的热点，换算法，向量化到 NumPy / PyTorch，最后才是 C 扩展或换解释器")
    steps = [("测量\ntimeit / cProfile", "gray"), ("定位热点\n80% 时间在 20% 代码", "blue"), ("换算法 / 数据结构\nO(n²) → O(n log n)", "orange"), ("向量化\nNumPy / PyTorch", "green"), ("C / Rust 扩展\n或多进程", "purple")]
    for i, (name, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 50, 120, 56, cls, rx=7, text=name, size=9.5)
        if i < len(steps) - 1:
            f.arrow(x + 120, 78, x + 136, 78, sw=1.3)
    f.text(350, 140, "每一步都要回到第一步重新测：优化没测过的代码等于猜。GIL 让多线程只对 I/O 有用，CPU 密集要多进程或在 C 扩展里释放 GIL", cls="tx", size=10)
    f.text(350, 166, "推理服务里的典型热点：分词和反分词、请求的 JSON 解析、Python 侧的调度循环——所以引擎把它们挪到 Rust / C++ 或者批量化", cls="mu", size=9.5)
    f.text(350, 188, "tracemalloc 看内存：泄漏往往是缓存无限增长、闭包抓住大对象、日志里存了引用", cls="mu", size=9.5)
    return f


@figure("media", "long-video-paths")
def long_video_paths():
    f = Fig(700, 300, "长视频的三条路：分段拼接靠条件帧续写；滑动窗口在重叠区混合；自回归把前面块的 K、V 缓存起来，块内双向、块间因果")
    rows = [("① 分段 + 条件拼接", "orange", [(0, 5), (5, 10), (10, 15)], "前一段末尾几帧当下一段的条件；成本线性，靠条件帧，容易漂移"),
            ("② 滑动窗口 / 重叠去噪", "blue", [(0, 6), (4, 10), (8, 14)], "窗口沿时间滑动，重叠区混合两边的结果；多付重叠比例的计算，更平滑"),
            ("③ 自回归（块因果）", "green", [(0, 5), (5, 10), (10, 15)], "块内双向、块间因果，缓存前面块的 K、V；要专门训练，训练时就对抗漂移")]
    for r, (name, cls, segs, note) in enumerate(rows):
        y = 40 + r * 84
        f.text(30, y + 14, name, cls="tx", size=11, weight="600", anchor="start")
        for i, (a, b) in enumerate(segs):
            x0, x1 = 240 + a * 28, 240 + b * 28
            f.rect(x0, y + (6 if r == 1 and i % 2 else 0), x1 - x0 - 2, 26, cls, rx=4, text=f"段 {i + 1}", size=9.5, sw=0.9)
            if r == 0 and i > 0:
                f.rect(x0 - 10, y + 2, 8, 22, "red", rx=2, sw=0.6)
            if r == 2 and i > 0:
                f.arrow(x0 - 28, y + 40, x0 + 20, y + 28, cls="green-l", hcls="green-s", sw=1.2)
        f.text(30, y + 48, note + ("（红：条件帧）" if r == 0 else "（绿箭头：读前面块的 KV）" if r == 2 else ""), cls="mu", size=9.5, anchor="start")
    f.text(350, 286, "一次生成的注意力是平方的，分段把它变成线性；真正的代价是漂移——误差沿着段累积，颜色、身份、运动慢慢跑偏", cls="mu", size=9.5)
    return f


@figure("media", "three-stage-pipeline")
def three_stage_pipeline():
    f = Fig(700, 260, "三个阶段三种放法：文本编码攒 batch 并缓存，去噪每卡一个请求，VAE 解码放到另一条 stream 和下一个请求的去噪重叠")
    f.text(100, 24, "串行", cls="tx", size=11.5, weight="600", anchor="start")
    x = 100
    for i in range(2):
        for name, w, cls in (("文本", 14, "green"), ("去噪", 150, "blue"), ("VAE", 40, "orange")):
            f.rect(x, 36, w - 2, 24, cls, rx=3, text=name if w > 30 else "", size=9.5)
            x += w
    f.text(x + 10, 50, "每请求 = 文本 + 去噪 + VAE", cls="mu", size=9.5, anchor="start")
    f.text(100, 100, "流水", cls="tx", size=11.5, weight="600", anchor="start")
    f.text(60, 126, "文本编码器\n（批处理 + 缓存）", cls="mu", size=9, anchor="start")
    f.rect(180, 114, 12, 22, "green", rx=3, sw=0.8)
    f.rect(194, 114, 12, 22, "green", rx=3, sw=0.8)
    f.text(60, 160, "去噪（主力卡）", cls="mu", size=9, anchor="start")
    f.rect(210, 148, 148, 24, "blue", rx=3, text="请求 1 去噪", size=9.5)
    f.rect(360, 148, 148, 24, "blue", rx=3, text="请求 2 去噪", size=9.5)
    f.rect(510, 148, 148, 24, "blue", rx=3, text="请求 3 去噪", size=9.5)
    f.text(60, 196, "VAE 解码\n（另一条 stream / 卡）", cls="mu", size=9, anchor="start")
    f.rect(360, 184, 38, 24, "orange", rx=3, text="1", size=9.5)
    f.rect(510, 184, 38, 24, "orange", rx=3, text="2", size=9.5)
    f.text(350, 236, "稳态下每个请求只占 max(去噪, VAE) 的时间：SDXL 从 3.25 s 到 2.9 s（省一成），少步模型（去噪 0.15 s、VAE 0.3 s）下 VAE 反而成了瓶颈，要分块或分卡", cls="mu", size=9.5)
    return f


@figure("media", "lora-bypass-controlnet")
def lora_bypass_controlnet():
    f = Fig(700, 270, "LoRA 旁路 vs 融合；ControlNet 是去噪网络的一个副本编码器，每一步多跑半个网络，把条件特征加进主干的跳连")
    f.text(170, 24, "LoRA：旁路还是融合", cls="tx", size=12, weight="600")
    f.rect(40, 50, 60, 30, "gray", rx=5, text="x", size=10)
    f.arrow(100, 65, 150, 65, sw=1.2)
    f.rect(150, 50, 90, 30, "blue", rx=5, text="W（基座）", size=9.5)
    f.rect(150, 96, 90, 30, "orange", rx=5, text="A·B（秩 r）", size=9.5)
    f.arrow(100, 65, 150, 111, sw=1.1)
    f.circle(275, 65, 11, "green", text="+", size=12)
    f.arrow(240, 65, 264, 65, sw=1.1)
    f.arrow(240, 111, 268, 74, sw=1.1)
    f.arrow(286, 65, 320, 65, sw=1.2)
    f.rect(320, 50, 30, 30, "gray", rx=5, text="y", size=10)
    f.text(30, 150, "旁路：多两个小矩阵乘（1%～2% FLOP），切换零成本，\n多个 LoRA 可以同一 batch 混用\n融合：W′ = W + A·B 一次性算进权重，推理零开销，\n但换 LoRA 要重新算，量化和编译图都要重来", cls="mu", size=9, anchor="start")
    f.line(370, 30, 370, 240, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(535, 24, "ControlNet：每步多半个网络", cls="tx", size=12, weight="600")
    f.rect(390, 50, 110, 110, "blue", rx=8, sw=1.2)
    f.text(445, 68, "去噪 UNet", cls="tx", size=10, weight="600")
    for i, name in enumerate(("编码器", "中间", "解码器")):
        f.rect(400, 80 + i * 26, 90, 20, "blue", rx=3, text=name, size=9, sw=0.6)
    f.rect(540, 50, 110, 70, "orange", rx=8, sw=1.2)
    f.text(595, 68, "ControlNet", cls="tx", size=10, weight="600")
    f.rect(550, 80, 90, 20, "orange", rx=3, text="编码器副本", size=9, sw=0.6)
    f.rect(550, 102, 90, 14, "gray", rx=3, text="零卷积", size=8, sw=0.6)
    f.text(595, 140, "输入：边缘 / 深度 / 姿态图", cls="mu", size=9)
    f.arrow(550, 90, 495, 132, cls="orange-l", hcls="orange-s", sw=1.4, label="加到跳连", lx=18, ly=10, lsize=8.5)
    f.text(535, 180, "权重约为 UNet 的一半（SD 1.5：1.2 GB），每步再跑一次编码器：\n算力 +40%～50%；多个 ControlNet 叠加就多跑多份", cls="mu", size=9)
    f.text(350, 252, "放置：LoRA 几十 MB 全部常驻；ControlNet 几 GB 按 LRU 常驻几个、其余放锁页内存按需搬，或按类型分实例", cls="mu", size=9.5)
    return f


@figure("media", "distill-trajectory")
def distill_trajectory():
    f = Fig(700, 270, "少步蒸馏：老师沿 ODE 轨迹走几十小步，学生学会从任意一点一步（或几步）跨到终点——步数不再是求解器的精度问题")
    pts = [(60, 180), (130, 150), (200, 128), (270, 112), (340, 100), (410, 92), (480, 86), (550, 82), (620, 80)]
    for i in range(len(pts) - 1):
        f.arrow(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], cls="blue-l", hcls="blue-s", sw=1.4)
    for x, y in pts:
        f.circle(x, y, 4, "blue-s", sw=0)
    f.text(100, 190, "噪声 x_T", cls="mu", size=10, anchor="start")
    f.text(620, 62, "数据 x_0", cls="mu", size=10)
    f.path(f"M 60 180 C 200 60 450 40 620 80", cls="orange-l", sw=2.4, dash="7 4")
    f.head(620, 80, -0.3, cls="orange-s", size=8)
    f.path(f"M 200 128 C 350 70 500 60 620 80", cls="green-l", sw=1.6, dash="4 3")
    f.text(30, 210, "蓝：老师，几十次网络调用，一步一步沿轨迹走　　橙：学生，从同一个起点一步到达同一个终点（一致性 / 分布匹配 / 对抗蒸馏）", cls="tx", size=10, anchor="start")
    f.text(30, 232, "绿：一致性要求轨迹上任意一点出发都到同一个终点；4～8 步版本就是把轨迹切成几段各跨一次", cls="mu", size=9.5, anchor="start")
    f.text(30, 254, "代价：多样性和可控性下降、对 LoRA / ControlNet 兼容变差、不能随意换调度器；引导蒸馏把 CFG 也烘进去，一步只算一次前向", cls="mu", size=9.5, anchor="start")
    return f


@figure("serving", "rl-rollout-tail")
def rl_rollout_tail():
    f = Fig(700, 270, "RL 的一步：推理池生成一批回答 → 算奖励 → 训练 → 权重同步；一批里最长的那几条回答决定了整步的时间（长尾）")
    steps = [("rollout\n推理引擎采样", "blue"), ("奖励\n验证器 / 奖励模型", "orange"), ("训练\n策略梯度一步", "green"), ("权重同步\n训练器 → 推理池", "purple")]
    for i, (name, cls) in enumerate(steps):
        x = 30 + i * 165
        f.rect(x, 40, 140, 50, cls, rx=7, text=name, size=10)
        if i < len(steps) - 1:
            f.arrow(x + 140, 65, x + 165, 65, sw=1.3)
    f.path("M 650 90 C 650 120 60 120 100 90", cls="ln", sw=1.2, dash="4 3")
    f.text(100, 130, "一批 512 条回答的长度", cls="tx", size=10, weight="600", anchor="start")
    lens = [0.15, 0.3, 0.2, 0.45, 0.25, 0.35, 0.18, 0.28, 0.22, 0.4, 0.3, 1.0, 0.26, 0.33]
    for i, l in enumerate(lens):
        f.rect(100, 142 + i * 6.5, l * 420, 5, "red" if l == 1.0 else "blue", rx=1, sw=0)
    f.text(100, 244, "红色那条最长：其他槽位早就空了，整步却要等它写完，池子的利用率掉到三四成", cls="tx", size=10, anchor="start")
    f.text(350, 264, "对策：一步异步 / 全异步流水、超长截断、按长度排程；训练与推理的概率不一致要用重要性采样修正或 batch 无关 kernel", cls="mu", size=9.5)
    return f


@figure("serving", "rl-pipelines")
def rl_pipelines():
    f = Fig(700, 250, "三种 RL 流水：同步等整批；一步异步用上一版权重生成下一批；全异步槽位一空就开新样本，训练端凑够一批就更新")
    rows = [("同步", [("生成", 0, 180, "blue"), ("训练", 180, 50, "green"), ("生成", 230, 180, "blue"), ("训练", 410, 50, "green")], "多数时间在等长尾"),
            ("一步异步", [("生成 i+1", 0, 180, "blue"), ("生成 i+2", 180, 180, "blue"), ("生成 i+3", 360, 100, "blue")], "样本落后一个版本"),
            ("全异步", [("槽位不断开新样本……", 0, 460, "blue")], "凑够一批就训")]
    for r, (name, segs, note) in enumerate(rows):
        y = 40 + r * 62
        f.text(80, y + 14, name, cls="tx", size=11, weight="600", anchor="end")
        for label, x0, w, cls in segs:
            f.rect(100 + x0, y, w - 2, 26, cls, rx=3, text=label, size=9)
        if r == 1:
            for i, x0 in enumerate((180, 360)):
                f.rect(100 + x0 - 60, y + 30, 50, 14, "green", rx=2, text=f"训 {i + 1}", size=8, sw=0.6)
        if r == 2:
            for i, x0 in enumerate((90, 200, 300, 400)):
                f.rect(100 + x0, y + 30, 45, 14, "green", rx=2, text="训", size=8, sw=0.6)
        f.text(575, y + 14, note, cls="mu", size=9, anchor="start")
    f.text(350, 228, "异步的代价是样本陈旧（off-policy）：要靠重要性采样或限制落后版本数；权重同步从秒级（NCCL 广播）到分钟级（走存储）决定能异步到什么程度", cls="mu", size=9.5)
    return f


@figure("serving", "deepep-two-hop")
def deepep_two_hop():
    f = Fig(700, 260, "DeepEP 高吞吐模式的两跳 dispatch：token 先经 RDMA 发到目标节点上与自己同位置的卡（每个节点只发一份），再经 NVLink 转发给真正持有专家的卡")
    for n, x0 in ((0, 40), (1, 400)):
        f.rect(x0, 50, 260, 150, "gray", rx=10, sw=1, dash="5 4")
        f.text(x0 + 130, 68, f"节点 {n}", cls="tx", size=11, weight="600")
        for g in range(4):
            x = x0 + 15 + g * 62
            f.rect(x, 84, 52, 40, "blue", rx=5, text=f"GPU {g}\n专家 {n * 16 + g * 4}…", size=8)
        for g in range(4):
            x = x0 + 15 + g * 62
            f.rect(x, 140, 52, 40, "blue", rx=5, text=f"GPU {g + 4}", size=8.5)
    f.arrow(107, 104, 467, 104, cls="orange-l", hcls="orange-s", sw=2.2)
    f.text(350, 36, "① RDMA：同轨，每个目标节点只发一份（一个 token 最多去 4 个节点）", cls="tx", size=9.5)
    f.arrow(467, 124, 591, 140, cls="green-l", hcls="green-s", sw=1.8, label="② NVLink 转发", lx=40, ly=0, lsize=9)
    f.text(350, 222, "机间带宽（50 GB/s）比 NVLink（450 GB/s）贵 9 倍，所以去重发给节点、再在节点内分发；节点受限路由保证每个 token 最多跨 4 个节点", cls="tx", size=9.5)
    f.text(350, 246, "低延迟模式（decode）：固定槽位、纯 RDMA、无 CPU 同步，牺牲带宽换几十微秒的延迟；kernel 内通信靠 NVSHMEM 的 put / signal", cls="mu", size=9.5)
    return f


@figure("serving", "vllm-processes")
def vllm_processes():
    f = Fig(700, 240, "vLLM 的进程结构：API server 进程做分词 / 反分词，EngineCore 进程只管调度与 KV，Worker 进程（每卡一个）跑模型")
    boxes = [(30, "API server 进程", "FastAPI 路由\nAsyncLLM\nInputProcessor（分词）\nOutputProcessor（反分词）", "green"),
             (270, "EngineCore 进程", "输入线程 → input_queue\n主线程：busy loop\nScheduler + KVCacheManager\n输出线程 ← output_queue", "blue"),
             (510, "Worker 进程 × N", "GPUModelRunner\n模型 + 注意力后端\nSampler", "orange")]
    for x, title, body, cls in boxes:
        f.rect(x, 40, 160, 120, cls, rx=9, sw=1.2)
        f.text(x + 80, 58, title, cls="tx", size=10.5, weight="600")
        f.text(x + 80, 108, body, cls="tx", size=9)
    f.arrow(190, 90, 270, 90, sw=1.4, label="ZMQ + msgpack", ly=-10, lsize=9)
    f.arrow(270, 120, 190, 120, sw=1.4)
    f.arrow(430, 90, 510, 90, sw=1.4, label="共享内存消息队列（广播）", ly=-10, lsize=9)
    f.arrow(510, 120, 430, 120, sw=1.4)
    f.text(350, 190, "把 CPU 重活（HTTP、分词）和调度分到不同进程，GPU 调度循环不被 Python 的 GIL 和 HTTP 处理卡住；单卡时 Worker 直接在 EngineCore 进程里", cls="mu", size=9.5)
    f.text(350, 214, "数据并行：每个 DP rank 一个 EngineCore，前端做负载均衡；读源码从 EngineCore.step() 和 Scheduler.schedule() 两条主线开始", cls="mu", size=9.5)
    return f


@figure("serving", "sglang-processes")
def sglang_processes():
    f = Fig(700, 240, "SGLang 的进程结构：TokenizerManager 在主进程，每个 TP rank 一个 Scheduler 进程（调度和模型执行在同一进程），反分词单独一个进程")
    boxes = [(30, "主进程", "HTTP 服务（FastAPI）\nTokenizerManager\n分词、对话模板、多模态预处理", "green"),
             (270, "Scheduler 进程 × TP", "event_loop_overlap / normal\n接收请求 → 组批 → run_batch\nTpModelWorker → ModelRunner", "blue"),
             (510, "DetokenizerManager 进程", "增量反分词\n停止字符串", "orange")]
    for x, title, body, cls in boxes:
        f.rect(x, 40, 160, 120, cls, rx=9, sw=1.2)
        f.text(x + 80, 58, title, cls="tx", size=10.5, weight="600")
        f.text(x + 80, 108, body, cls="tx", size=9)
    f.arrow(190, 90, 270, 90, sw=1.4, label="ZMQ", ly=-10, lsize=9)
    f.arrow(430, 90, 510, 90, sw=1.4, label="ZMQ", ly=-10, lsize=9)
    f.arrow(510, 130, 190, 130, sw=1.4, label="结果回到 TokenizerManager，再流式返回客户端", ly=12, lsize=9)
    f.text(350, 190, "和 vLLM 最大的差别：没有单独的 Engine 进程，每个 TP rank 的 Scheduler 各自调度出相同的批次；overlap 模式让 CPU 调度和 GPU 前向错开一步", cls="mu", size=9.5)
    f.text(350, 214, "读源码从 Scheduler.event_loop_overlap() 和 get_next_batch_to_run() 开始；KV 内存与基数树前缀缓存在 mem_cache 目录", cls="mu", size=9.5)
    return f


@figure("serving", "k8s-filter-score")
def k8s_filter_score():
    f = Fig(700, 250, "kube-scheduler 的两段式：从队列取一个 Pod → 过滤出可行节点 → 给可行节点打分 → 选最高分 → 绑定")
    f.rect(30, 70, 90, 40, "orange", rx=7, text="待调度 Pod\n要 4 张 GPU", size=9.5)
    f.arrow(120, 90, 160, 90, sw=1.3)
    nodes = [("节点 A：8 卡空闲", True, 92), ("节点 B：2 卡空闲", False, 0), ("节点 C：4 卡空闲", True, 70), ("节点 D：有污点", False, 0)]
    for i, (name, ok, score) in enumerate(nodes):
        y = 36 + i * 36
        f.rect(160, y, 130, 28, "blue" if ok else "gray", rx=5, text=name, size=9)
        if ok:
            f.arrow(290, y + 14, 340, y + 14, sw=1.1)
            f.rect(340, y, 90, 28, "green", rx=5, text=f"打分 {score}", size=9.5)
        else:
            f.text(312, y + 14, "✗", cls="mu", size=12)
    f.text(160, 190, "过滤（predicate）：资源够不够、亲和性、污点 / 容忍", cls="mu", size=9.5, anchor="start")
    f.text(160, 206, "打分（score）：最少 / 最多分配、镜像本地、拓扑分散", cls="mu", size=9.5, anchor="start")
    f.arrow(430, 50, 490, 90, sw=1.3)
    f.rect(490, 70, 90, 40, "orange", rx=7, text="绑定到 A\n写 nodeName", size=9.5)
    f.text(590, 150, "GPU 的特殊：整卡分配、\n要同一 NVSwitch 域、\n多 Pod 要 gang 调度", cls="mu", size=9, anchor="start")
    f.text(350, 236, "抢占：高优先级 Pod 放不下时驱逐低优先级的；队列：训练 / 批处理任务排队等一整组资源，别让它们和在线服务抢", cls="mu", size=9.5)
    return f


@figure("serving", "rolling-update")
def rolling_update():
    f = Fig(700, 250, "滚动发布：新旧两个 ReplicaSet 此消彼长，maxSurge 和 maxUnavailable 决定节奏；探针决定什么时候算就绪、什么时候算挂了")
    steps = [(3, 0), (3, 1), (2, 2), (1, 3), (0, 3)]
    for i, (old, new) in enumerate(steps):
        x = 40 + i * 128
        f.text(x + 50, 36, f"t{i}", cls="mu", size=10)
        for j in range(old):
            f.rect(x, 48 + j * 26, 46, 22, "gray", rx=3, text="旧", size=9)
        for j in range(new):
            f.rect(x + 54, 48 + j * 26, 46, 22, "green", rx=3, text="新", size=9)
        if i < len(steps) - 1:
            f.arrow(x + 104, 80, x + 128, 80, sw=1.1)
    f.text(350, 150, "maxSurge：最多比期望多几个（先起新的再杀旧的，要多占显存）；maxUnavailable：最多少几个（先杀旧的再起新的，容量掉）", cls="tx", size=10)
    f.text(350, 174, "三种探针：startup 给权重加载留足时间（否则被判成挂了反复重启）；readiness 决定接不接流量；liveness 决定要不要重启", cls="mu", size=9.5)
    f.text(350, 198, "优雅退出：收到 SIGTERM 先从 Service 摘掉、把在途请求生成完再退出，terminationGracePeriodSeconds 要比最长回答还长", cls="mu", size=9.5)
    f.text(350, 222, "扩缩容按排队请求数 / KV 使用率，不按 CPU；PodDisruptionBudget 挡住运维一次拿走太多副本", cls="mu", size=9.5)
    return f


@figure("serving", "api-processes")
def api_processes():
    f = Fig(700, 230, "OpenAI 兼容的流式服务：HTTP 进程收请求、分词、推流；引擎进程每步出一批 token；增量反分词把 token 变成不回退的文本片段")
    f.rect(30, 50, 150, 90, "green", rx=8, text="HTTP 进程\nFastAPI + asyncio\n分词、SSE 推流", size=9.5)
    f.rect(275, 50, 150, 90, "blue", rx=8, text="引擎进程\n调度 + 前向 + 采样\n每步每请求 1 个 token", size=9.5)
    f.rect(520, 50, 150, 90, "orange", rx=8, text="增量反分词\n攒到能确定的 UTF-8 边界\n再吐出文本片段", size=9.5)
    f.arrow(180, 80, 275, 80, sw=1.3, label="请求（token id）", ly=-10, lsize=9)
    f.arrow(425, 95, 520, 95, sw=1.3, label="token id 流", ly=-10, lsize=9)
    f.arrow(520, 120, 180, 120, sw=1.3, label="文本片段 → data: {...}", ly=12, lsize=9)
    f.text(350, 172, "为什么多进程：HTTP 解析、JSON、分词都在 GIL 下，和引擎的调度循环挤在一个进程里会让每一步都变慢；拆开后用 ZMQ / 共享内存传 token", cls="mu", size=9.5)
    f.text(350, 196, "停止条件：EOS、max_tokens、停止字符串（要在反分词后匹配，且可能跨 token）；取消要传回引擎释放 KV", cls="mu", size=9.5)
    f.text(350, 218, "批量采样：温度、top-k、top-p、惩罚都按请求不同，要向量化成一次 kernel；logprobs 要在采样前算", cls="mu", size=9.5)
    return f


@figure("serving", "new-model-steps")
def new_model_steps():
    f = Fig(700, 200, "接入一个新模型的五步：找差别 → 照参考实现写 → 逐层对齐 → 端到端指标 → 精度评测；每一步都有可量化的通过标准")
    steps = [("① 找差别", "和已支持的最近模型比 config、\n权重名、注意力 / MoE 的变体", "blue"), ("② 照参考实现写", "transformers 的实现是真相；\n按引擎的 Linear / Attention 抽象改", "green"), ("③ 逐层对齐", "同一输入，每一层输出\n和参考实现的最大误差 < 阈值", "orange"), ("④ 端到端", "贪心输出一致；\nTTFT / TPOT 对得上估算", "purple"), ("⑤ 精度评测", "几个基准的分数\n和官方对齐", "gray")]
    for i, (name, desc, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 40, 120, 36, cls, rx=7, text=name, size=10.5)
        f.text(x + 60, 110, desc, cls="mu", size=8.5)
    for i in range(4):
        f.arrow(140 + i * 136, 58, 156 + i * 136, 58, sw=1.2)
    f.text(350, 160, "权重名映射和 QKV 合并是最常出错的地方；逐层对齐要在 fp32 下做，bf16 的误差会把真正的 bug 掩盖掉", cls="tx", size=10)
    f.text(350, 184, "在框架里落地：注册模型类、写权重加载器、挑注意力后端、补 CUDA Graph 的形状、加一条 CI 的精度测试", cls="mu", size=9.5)
    return f


@figure("serving", "platform-layers")
def platform_layers():
    f = Fig(700, 230, "多硬件支持的分层：引擎逻辑不变，Platform 类把设备名、通信后端、注意力后端、图模式这些差异收进插件")
    layers = [("调度器、KV 管理、API：和硬件无关", "gray", 640), ("Platform 抽象：device_type、dispatch_key、集合通信、可见设备、注意力后端选择", "blue", 560), ("平台插件：CUDA / ROCm / XPU / 昇腾（vllm-ascend）", "orange", 480), ("kernel：FlashAttention / aiter / 昇腾注意力；MoE、量化、图模式", "green", 400)]
    for i, (name, cls, w) in enumerate(layers):
        f.rect(350 - w / 2, 36 + i * 36, w, 28, cls, rx=6, text=name, size=9.5)
    f.text(350, 190, "接入一种新硬件要做的：实现 Platform 的几十个钩子、提供注意力和 MoE kernel、通信后端（HCCL / RCCL / XCCL）、图模式或等价物", cls="tx", size=10)
    f.text(350, 214, "差别最大的不在表层属性，在 kernel 的覆盖面和性能：没有对应的融合 kernel 时，同样的引擎在新硬件上可能慢好几倍", cls="mu", size=9.5)
    return f


@figure("train", "cpu-verifiable")
def cpu_verifiable():
    f = Fig(700, 230, "没有多卡时能学什么：逻辑和数值正确性在 CPU 上用 gloo 多进程 100% 验证，性能数字一个都不能信")
    f.rect(30, 40, 400, 150, "green", rx=10, sw=1.2)
    f.text(230, 58, "CPU 上能验证（torchrun + gloo，4 个进程）", cls="tx", size=11, weight="600")
    items = ["集合通信的语义、环形 all-reduce 的通信量", "DDP 的分桶与重叠逻辑、ZeRO 的切分与 all-gather 时机", "TP / SP 的切法、流水线的 1F1B 气泡、Ulysses / Ring 注意力", "专家并行的 all-to-all 与负载统计；α-β 时间模型纯算"]
    for i, t in enumerate(items):
        f.rect(45, 70 + i * 28, 370, 22, "gray", rx=4, text=t, size=9)
    f.rect(460, 40, 210, 150, "red", rx=10, sw=1.2)
    f.text(565, 58, "必须真卡", cls="tx", size=11, weight="600")
    for i, t in enumerate(("实测带宽与拓扑", "重叠的真实收益、scaling 曲线", "NCCL 调参、多机 RDMA", "故障恢复、弹性训练")):
        f.rect(475, 70 + i * 28, 180, 22, "gray", rx=4, text=t, size=9)
    f.text(350, 212, "最重要的一招：每种并行都写一个单进程参考实现，多进程版本和它逐元素对齐；一张卡能多补的是 kernel 级的 profiling 和混合精度", cls="mu", size=9.5)
    return f


@figure("scratch", "grad-accum")
def grad_accum():
    f = Fig(700, 200, "梯度累积：把一个大 batch 切成几个 micro-batch，前向反向各跑一次、梯度累加，最后才 optimizer.step()——显存按 micro-batch 算，等价于大 batch")
    for i in range(4):
        x = 40 + i * 120
        f.rect(x, 50, 100, 26, "blue", rx=4, text=f"micro {i + 1} 前向", size=9)
        f.rect(x, 80, 100, 26, "orange", rx=4, text="反向，梯度 +=", size=9)
        if i < 3:
            f.arrow(x + 100, 78, x + 120, 78, sw=1.1)
    f.rect(530, 50, 140, 56, "green", rx=7, text="optimizer.step()\nzero_grad()", size=10)
    f.arrow(500, 78, 530, 78, sw=1.3)
    f.text(350, 134, "激活显存只有一个 micro-batch 的量；损失要除以累积步数，DDP 下只在最后一个 micro-batch 同步梯度（no_sync），否则通信翻几倍", cls="tx", size=10)
    f.text(350, 158, "等价的前提：BatchNorm 这类按 batch 统计的层会不一样（Transformer 没有）；学习率按全局 batch 定，不按 micro-batch", cls="mu", size=9.5)
    f.text(350, 182, "再往上：多张卡做 DDP 是把 micro-batch 摊到卡上，各卡算完 all-reduce——同样的等价关系", cls="mu", size=9.5)
    return f


@figure("python", "attribute-lookup")
def attribute_lookup():
    f = Fig(700, 230, "obj.attr 的查找顺序：先看类的 MRO 里有没有数据描述符（property）→ 实例字典 → 非数据描述符 / 类属性 → __getattr__ 兜底")
    steps = [("① type(obj) 的 MRO\n找到数据描述符？\n（有 __set__：property）", "purple", "调用它的 __get__"), ("② obj.__dict__\n实例自己的属性", "blue", "直接返回"), ("③ MRO 里的\n非数据描述符 / 类属性\n（函数 → 绑定成方法）", "orange", "函数的 __get__ 产生方法"), ("④ __getattr__\n兜底", "gray", "没有就 AttributeError")]
    for i, (name, cls, out) in enumerate(steps):
        x = 20 + i * 170
        f.rect(x, 44, 150, 64, cls, rx=7, text=name, size=8.5)
        f.text(x + 75, 126, out, cls="mu", size=9)
        if i < len(steps) - 1:
            f.arrow(x + 150, 76, x + 170, 76, sw=1.2, label="没找到", ly=-9, lsize=8)
    f.text(350, 166, "所以 property 不会被实例字典里的同名值遮住（它在第 ① 步），而方法可以被实例属性覆盖（它在第 ③ 步）", cls="tx", size=10)
    f.text(350, 190, "描述符就是\"访问属性时跑一段代码\"的协议：property、方法、classmethod、__slots__、ORM 字段都是描述符", cls="mu", size=9.5)
    f.text(350, 212, "__getattribute__ 拦截一切（慎用），__getattr__ 只在找不到时才调；元类改的是\"类本身怎么创建\"，和属性查找是两回事", cls="mu", size=9.5)
    return f


@figure("python", "exception-tree")
def exception_tree():
    f = Fig(700, 240, "异常的层级：except 匹配的是\"是不是这个类或其子类\"——捕获 Exception 会吞掉几乎一切，捕获 BaseException 连 Ctrl-C 都吞")
    pos = {"BaseException": (350, 40), "SystemExit": (120, 95), "KeyboardInterrupt": (260, 95), "Exception": (470, 95),
           "ValueError": (318, 150), "OSError": (428, 150), "KeyError": (538, 150), "RuntimeError": (648, 150), "FileNotFoundError": (428, 200), "UnicodeError": (318, 200)}
    edges = [("BaseException", "SystemExit"), ("BaseException", "KeyboardInterrupt"), ("BaseException", "Exception"), ("Exception", "ValueError"), ("Exception", "OSError"), ("Exception", "KeyError"), ("Exception", "RuntimeError"), ("OSError", "FileNotFoundError"), ("ValueError", "UnicodeError")]
    for a, b in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        f.line(x1, y1 + 12, x2, y2 - 12, cls="ln", sw=1.1)
    for name, (x, y) in pos.items():
        cls = "red" if name in ("BaseException", "SystemExit", "KeyboardInterrupt") else ("orange" if name == "Exception" else "blue")
        f.rect(x - 50, y - 12, 100, 24, cls, rx=5, text=name, size=8.5)
    f.text(120, 200, "自定义异常继承 Exception，\n按领域建一棵小树，\n调用方按需要的粒度 except", cls="mu", size=9, anchor="start")
    f.text(350, 230, "except 的顺序从具体到一般；finally 总会执行；上下文管理器的 __exit__ 收到异常三元组，返回 True 才吞掉", cls="mu", size=9.5)
    return f


@figure("python", "abc-vs-protocol")
def abc_vs_protocol():
    f = Fig(700, 220, "两种\"接口\"：ABC 靠继承（名义子类型，运行时检查），Protocol 靠形状（结构子类型，静态检查），鸭子类型是两者的根")
    f.text(180, 24, "ABC：显式继承", cls="tx", size=12, weight="600")
    f.rect(100, 40, 160, 30, "purple", rx=6, text="class Reader(ABC)", size=9.5, )
    f.rect(100, 90, 160, 30, "blue", rx=6, text="class FileReader(Reader)", size=9.5)
    f.arrow(180, 90, 180, 70, sw=1.2)
    f.text(180, 140, "继承；缺抽象方法时实例化就报错；isinstance 可用\n第三方的类不能被你声明成你的 ABC（除非 register）", cls="mu", size=9)
    f.line(350, 30, 350, 170, cls="ln", sw=1, dash="4 4", opacity=0.4)
    f.text(530, 24, "Protocol：只看形状", cls="tx", size=12, weight="600")
    f.rect(430, 40, 200, 30, "orange", rx=6, text="class SupportsRead(Protocol): read()", size=9)
    f.rect(430, 90, 200, 30, "blue", rx=6, text="任何有 read() 的类都算", size=9.5)
    f.arrow(530, 90, 530, 70, sw=1.2, dash="4 3")
    f.text(530, 140, "不用继承，mypy 按方法签名匹配；第三方类、内置类型天然满足\nruntime_checkable 才能 isinstance，且只查方法名", cls="mu", size=9)
    f.text(350, 190, "怎么选：给别人实现的框架接口用 ABC（强制、带默认实现）；描述\"我只需要它有这些方法\"的参数类型用 Protocol；标准库的 collections.abc 两者兼备", cls="mu", size=9.5)
    f.text(350, 210, "推理框架里：注意力后端、Platform 这类插件点多用 ABC；工具函数的参数类型多用 Protocol（SupportsIndex、Iterable）", cls="mu", size=9.5)
    return f


@figure("python", "ci-pipeline")
def ci_pipeline():
    f = Fig(700, 180, "从提交到发布的流水线：本地 pre-commit 挡掉格式和明显错误，CI 跑测试与类型检查，通过后打包发布")
    steps = [("本地提交", "pre-commit：\nruff 格式 + lint", "gray"), ("CI：检查", "ruff、mypy\n多 Python 版本矩阵", "blue"), ("CI：测试", "pytest + 覆盖率\n慢测试打标记", "green"), ("构建", "uv build → wheel\n版本号来自 tag", "orange"), ("发布", "推到 PyPI / 内部源\n或构建镜像", "purple")]
    for i, (name, desc, cls) in enumerate(steps):
        x = 20 + i * 136
        f.rect(x, 40, 120, 34, cls, rx=7, text=name, size=10.5)
        f.text(x + 60, 100, desc, cls="mu", size=9)
        if i < len(steps) - 1:
            f.arrow(x + 120, 57, x + 136, 57, sw=1.2)
    f.text(350, 150, "配置都在 pyproject.toml 一个文件里；密钥走 CI 的 secrets 不进仓库；锁文件（uv.lock）保证 CI 和本地装的是同一组版本", cls="mu", size=9.5)
    return f


@figure("cuda", "library-map")
def library_map():
    f = Fig(700, 240, "CUDA 库的全景：从手写 kernel 到调库，抽象层级越高越省事、越难榨干性能——高性能 GEMM 和注意力几乎都落在中间那一层")
    layers = [("调库：cuBLAS / cuDNN / cuFFT / NCCL", "成熟、稳定、形状通用；融合不了自己的算子", "gray", 640),
              ("模板库：CUTLASS / CuTe、CUB、Thrust、libcu++", "GEMM / 注意力 / 归约的积木，自己拼装、能融合 epilogue", "blue", 560),
              ("DSL：Triton、TileLang、CuTe DSL", "写 tile 级的逻辑，编译器排线程和访存；几十行一个融合 kernel", "orange", 480),
              ("手写 CUDA / PTX：mma.sync、cp.async、TMA", "极致性能，工作量最大；FlashAttention、DeepGEMM 这一层", "green", 400)]
    for i, (name, desc, cls, w) in enumerate(layers):
        f.rect(350 - w / 2, 36 + i * 40, w, 30, cls, rx=6, text=name, size=10)
        f.text(350, 36 + i * 40 + 38 - 2, "", cls="mu", size=1)
    for i, (name, desc, cls, w) in enumerate(layers):
        f.text(350 + w / 2 + 4, 51 + i * 40, "", cls="mu", size=1)
    f.text(350, 200, "接入 PyTorch 都走同一条路：torch.library 注册算子 → 自定义 autograd（训练才需要）", cls="mu", size=9.5)
    f.text(350, 216, "→ 注意 stream、dtype、连续性 → 用 torch.compile 的 custom op 让编译器认识它", cls="mu", size=9.5)
    f.text(350, 234, "怎么选：先 torch.compile，不够再 Triton，GEMM 类用 CUTLASS，只有热点中的热点才值得手写 PTX", cls="mu", size=9.5)
    return f


@figure("cpp", "project-layout")
def project_layout():
    f = Fig(700, 245, "一个 C++ 组件的工程结构：公开头文件、实现、测试、CMake 目标各在其位，依赖方向只能从外向内")
    f.rect(40, 40, 140, 150, "blue", rx=8, sw=1.2)
    f.text(110, 58, "include/kv/", cls="tx", size=10.5, weight="600")
    f.text(110, 100, "公开头文件\n只放接口\n被别的目标 #include", cls="mu", size=9)
    f.rect(220, 40, 140, 150, "orange", rx=8, sw=1.2)
    f.text(290, 58, "src/", cls="tx", size=10.5, weight="600")
    f.text(290, 100, "实现 .cpp\n私有头文件\n编译成库目标", cls="mu", size=9)
    f.rect(400, 40, 140, 150, "green", rx=8, sw=1.2)
    f.text(470, 58, "tests/", cls="tx", size=10.5, weight="600")
    f.text(470, 100, "GoogleTest / Catch2\n每个测试一个可执行文件\nctest 统一跑", cls="mu", size=9)
    f.rect(580, 40, 100, 150, "purple", rx=8, sw=1.2)
    f.text(630, 58, "python/", cls="tx", size=10.5, weight="600")
    f.text(630, 100, "pybind11 绑定\n→ .so 模块", cls="mu", size=9)
    f.arrow(360, 150, 220, 150, sw=1.2, both=False)
    f.arrow(400, 160, 180, 160, sw=1.2)
    f.arrow(580, 170, 180, 170, sw=1.2)
    f.text(350, 210, "CMakeLists.txt：add_library(kv) + target_include_directories(PUBLIC include) + target_link_libraries；sanitizer 作为一个 option", cls="mu", size=9.5)
    f.text(350, 230, "箭头 = 谁 #include 谁：实现、测试、绑定都只依赖公开头文件；编译时间靠 ccache、预编译头和拆分翻译单元", cls="mu", size=9.5)
    return f


@figure("media", "flash-attention")
def flash_attention_media():
    return flash_attention()


@figure("train", "rlhf-dpo-flow")
def rlhf_dpo_flow_train():
    return rlhf_dpo_flow()




# ====================================================================== SGLang 设计演进
@figure("sglang", "sgl-origins-timeline")
def sgl_origins_timeline():
    f = Fig(720, 190, "从空仓库到初始提交的四个月：2023-10-09 建仓，12-12 论文上 arXiv，2024-01-08 一次放出一万行代码，01-17 博客与 v0.1.5")
    x0, x1, y = 40, 680, 100
    f.line(x0, y, x1, y, sw=2)
    f.head(x1 + 2, y, 0)
    for i, m in enumerate(["2023-10", "11", "12", "2024-01", "02"]):
        x = x0 + 20 + i * 150
        f.line(x, y - 5, x, y + 5, sw=1.2)
        f.text(x, y + 20, m, cls="mu", size=11)
    # (圆点 x, 标签中心 x, 标签 y, 文字, 颜色)：一月的事件挤在一起，标签错开到左右两侧
    ev = [(60, 90, 46, "10-09 空仓库\n.gitignore、LICENSE、README", "gray"), (337, 337, 154, "12-12 论文 v1 上 arXiv\n2312.07104", "purple"),
          (470, 420, 46, "01-08 release initial code\n145 个文件、1.78 万行", "blue"), (520, 500, 154, "01-16 #7 修匹配 bug\nv0.1.3", "orange"),
          (525, 590, 46, "01-17 LMSYS 博客\nv0.1.5", "green"), (620, 645, 154, "02-05 jump-forward 博客\n#144", "orange")]
    for x, lx, ly, label, cls in ev:
        f.circle(x, y, 6, cls + "-s", sw=1)
        f.line(x, y + (-8 if ly < y else 8), lx, ly + (18 if ly < y else -18), sw=1)
        f.text(lx, ly, label, cls="tx", size=10)
    return f


@figure("sglang", "sgl-v0-processes")
def sgl_v0_processes():
    f = Fig(720, 300, "初始提交的进程与数据流：主进程（HTTP + 分词）→ 路由进程 → 模型进程（rpyc，调度与前向）→ 反分词进程 → 主进程，ZMQ 连成环")
    boxes = [(20, 40, 150, 110, "green", "主进程", "uvicorn / FastAPI\nTokenizerManager\n分词、图片预处理\nrid → 等待事件"),
             (215, 40, 150, 110, "gray", "路由进程", "RouterManager\n两个协程：收请求 /\n每步调用 model_client.step\nawait sleep(1ms)"),
             (410, 40, 150, 110, "blue", "模型进程 × TP", "ModelRpcServer（rpyc）\nexposed_step → forward_step\n调度 + ModelRunner 前向\nRadixCache、内存池"),
             (585, 40, 115, 110, "orange", "反分词进程", "Detokenizer\nManager\nbatch_decode\n裁停止串")]
    for x, y, w, h, cls, title, body in boxes:
        f.rect(x, y, w, h, cls, rx=9, sw=1.2)
        f.text(x + w / 2, y + 17, title, size=11, weight="600")
        f.text(x + w / 2, y + 66, body, size=9.3)
    f.arrow(170, 80, 215, 80, label="ZMQ PUSH", ly=-9, lsize=9)
    f.arrow(365, 80, 410, 80, label="rpyc", ly=-9, lsize=9)
    f.arrow(410, 110, 365, 110, label="out_pyobjs", ly=12, lsize=9)
    f.elbow([(290, 150), (290, 190), (642, 190), (642, 150)])
    f.text(466, 203, "BatchTokenIDOut（ZMQ）", cls="mu", size=9.5)
    f.elbow([(642, 150), (642, 225), (95, 225), (95, 150)])
    f.text(370, 238, "BatchStrOut（ZMQ）→ 主进程按 rid 唤醒等待的协程，流式返回客户端", cls="mu", size=9.5)
    f.text(360, 272, "--tp-size N 时有 N 个模型进程，各自用 NCCL 组成张量并行组；路由进程并发地对每个 rank 调用 step，只取 rank 0 的返回值", cls="mu", size=9.5)
    f.text(360, 288, "2024-07 #646 去掉 rpyc 之后，路由进程与模型进程合并成今天的 Scheduler 进程；三类角色与 ZMQ 环保留至今", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-v0-step")
def sgl_v0_step():
    f = Fig(720, 280, "初版调度的一步：先尝试组一个新的 extend batch，组不出来就对运行中的 batch 连做 10 步 decode；准入靠预估未来需求")
    f.rect(20, 30, 120, 44, "gray", text="收到的新请求\n→ forward_queue", size=10)
    f.arrow(140, 52, 185, 52)
    f.rect(185, 24, 180, 56, "blue", text="get_new_fill_batch\n前缀匹配 → 按策略排序\n→ 逐个判断能否接纳", size=10)
    f.arrow(365, 52, 415, 52, label="组出来了", ly=-9, lsize=9.5)
    f.rect(415, 30, 140, 44, "green", text="forward_fill_batch\nEXTEND 前向 + 采样", size=10)
    f.arrow(555, 52, 595, 52)
    f.rect(595, 30, 105, 44, "orange", text="并入\nrunning_batch", size=10)
    f.arrow(275, 80, 275, 118, label="组不出来", lx=38, ly=0, lsize=9.5)
    f.rect(185, 118, 180, 44, "green", text="forward_decode_batch × 10\n每步一个 token", size=10)
    f.arrow(365, 140, 415, 140)
    f.rect(415, 118, 140, 44, "gray", text="handle_finished_requests\n插回树、释放槽位", size=10)
    f.arrow(555, 140, 595, 140)
    f.rect(595, 118, 105, 44, "orange", text="发给\n反分词进程", size=10)
    f.rect(20, 190, 680, 70, "bx", rx=9, dash="4 3")
    f.text(360, 207, "准入判断（model_rpc.py 236–285）", size=10.5, weight="600")
    f.text(360, 228, "可用空间 = 空闲槽位 + 树上可淘汰的 token − Σ 运行中请求的（剩余 max_new_tokens × 0.4）", size=10.5, family="mono")
    f.text(360, 248, "新请求的 新增 token + max_new_tokens 放得下才接纳；接纳时锁住命中的树节点（inc_ref_counter），锁完发现放不下就立刻解锁放弃", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-radix-split")
def sgl_radix_split():
    f = Fig(720, 290, "匹配到边的中间：修复前继续往下递归，把别的上下文的 KV 当成命中；修复后在分叉点分裂并停止")

    def tree(x0, title, edges, hl, note):
        f.text(x0 + 150, 30, title, size=11, weight="600")
        f.circle(x0 + 150, 60, 10, "gray", text="根", size=10)
        for (x1, y1, x2, y2, label, cls) in edges:
            f.line(x0 + x1, y1, x0 + x2, y2, cls=cls, sw=2 if cls != "ln" else 1.4)
            f.text(x0 + (x1 + x2) / 2 + 14, (y1 + y2) / 2, label, size=10, family="mono", anchor="start")
            f.circle(x0 + x2, y2, 9, "bx", sw=1.2)
        f.text(x0 + 150, 232, hl, size=10.5, family="mono")
        f.text(x0 + 150, 256, note, cls="mu", size=9.5)

    tree(10, "修复前：树里有 Hello_L.A.! → world，查 Hello_world",
         [(150, 70, 150, 120, "Hello_L.A.!", "red-l"), (150, 129, 150, 180, "world", "red-l")],
         "命中 = Hello_ + world（11 个）", "分叉在边中间，却递归进了孩子：world 的 KV 来自另一条序列")
    tree(370, "修复后（#7，2024-01-16）",
         [(150, 70, 150, 110, "Hello_", "green-l"), (150, 119, 150, 160, "L.A.!", "ln"), (150, 169, 150, 210, "world", "ln")],
         "命中 = Hello_（6 个）", "边在分叉点切成两段，新中间节点承接前半段；匹配到此为止")
    f.text(360, 278, "修复的两行：prefix_len < len(c_key) 就分裂并停止；整条边命中时追加整条 child.value 再递归", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-jump-forward")
def sgl_jump_forward():
    f = Fig(720, 240, "压缩 FSM：一串只有一条出边的状态压成一条边，一次跳过整段确定的字符串；跳过之后要连前文一起重新分词")
    chars = ['"', 'a', 'g', 'e', '"', ':', ' ']
    x = 30
    for i, c in enumerate(chars):
        f.circle(x, 60, 11, "gray", text=f"s{i}", size=9)
        f.arrow(x + 11, 60, x + 54, 60, sw=1.2)
        f.text(x + 32, 48, c if c != " " else "␣", size=11, family="mono")
        x += 65
    f.circle(x, 60, 11, "blue", text="s7", size=9)
    for dy, lab in ((-22, "0-9"), (22, "-")):
        f.arrow(x + 11, 60, x + 60, 60 + dy, sw=1.2)
        f.text(x + 72, 60 + dy, lab, size=10, family="mono", anchor="start")
    f.text(30 + 3.5 * 65 - 32, 92, "每个状态只有一条出边、且只对应一个字符 → 可以压缩", cls="mu", size=10)
    f.circle(30, 150, 11, "gray", text="s0", size=9)
    f.arrow(41, 150, 480, 150, sw=2.4, cls="green-l")
    f.text(260, 136, '一条边：跳过 "age":␣ （7 个字符，一次前向都不用）', size=10.5, family="mono")
    f.circle(490, 150, 11, "blue", text="s7", size=9)
    f.text(590, 150, "s7 有多条出边：\n交还给模型采样", size=10, anchor="start")
    f.text(360, 196, "初版实现：可以跳的请求从 batch 里摘出来 → 已有 KV 插回基数树 → 前文 + 跳过的字符串重新分词 → 当作新请求重新准入（前缀从树上命中）", cls="mu", size=9.5)
    f.text(360, 216, "2025-03 #4032 删掉了调度器里的这条路径；xgrammar 等语法库在 C++ 里提供 find_jump_forward_string，SGLang 只保留接口", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-frontend-stack")
def sgl_frontend_stack():
    f = Fig(720, 310, "前端的四层：程序 → IR → 执行器 → 后端；追踪器从程序提取常量前缀，编译器把追踪得到的图按依赖顺序执行")
    layers = [(30, "程序", "@sgl.function\ns += system(...) + gen(\"a\") ; s.fork(2) ; select(choices)", "green"),
              (95, "IR（lang/ir.py）", "SglExprList：Constant、Gen、Select、Image、RoleBegin/End、Fork、Variable", "blue"),
              (160, "执行器（lang/interpreter.py）", "StreamExecutor：后台线程 + 队列；每个变量一个 Event；fork 复制执行器", "purple"),
              (225, "后端（backend/runtime_endpoint.py）", "/generate：max_new_tokens=0 预热、normalized_logprob 打分\n/concate_and_append_request：分支的 KV 拼回主干", "orange")]
    for y, title, body, cls in layers:
        h = 62 if y == 225 else 52
        f.rect(30, y, 470, h, cls, rx=8, sw=1.2)
        f.text(265, y + 16, title, size=10.5, weight="600")
        f.text(265, y + (42 if h == 62 else 37), body, size=9.3, family="mono")
        if y < 225:
            f.arrow(265, y + 52, 265, y + 65)
    f.rect(30, 292, 470, 16, "gray", rx=5, text="SRT：三类进程 + RadixCache（第二章）", size=9.5)
    f.rect(530, 30, 170, 110, "bx", rx=8, dash="4 3")
    f.text(615, 48, "追踪器（lang/tracer.py）", size=10, weight="600")
    f.text(615, 92, "用假参数跑一遍程序\n→ 原语链（图）\n→ 开头的常量文本 = 前缀\n→ pin_program 预热缓存", size=9.3)
    f.rect(530, 160, 170, 90, "bx", rx=8, dash="4 3")
    f.text(615, 178, "编译器（lang/compiler.py）", size=10, weight="600")
    f.text(615, 216, "追踪得到的图\n→ 拓扑排序\n→ 每个 fork 分支一个执行器", size=9.3)
    f.arrow(500, 56, 530, 56, dash="3 2")
    f.arrow(530, 205, 500, 186, dash="3 2")
    return f



@figure("sglang", "sgl-process-evolution")
def sgl_process_evolution():
    f = Fig(720, 330, "进程模型的三个阶段：初版 rpyc 远程调用；v0.2 去掉 rpyc、rank 0 住进控制进程并广播请求；v0.4 起每个 rank 一个对等的调度器进程")
    cols = [(10, "2024-01 初版（rpyc）", "gray"), (250, "2024-07 v0.2（#646 去 rpyc）", "blue"), (490, "2024-10 起（#1538 scheduler.py）", "green")]
    for x, title, _ in cols:
        f.text(x + 110, 22, title, size=10.5, weight="600")
    # 阶段一
    f.rect(10, 40, 220, 30, "green", text="主进程：HTTP + Tokenizer", size=9.5)
    f.arrow(120, 70, 120, 92, label="ZMQ", lx=24, ly=0, lsize=9)
    f.rect(10, 92, 220, 34, "gray", text="路由进程 RouterManager\n每步 rpyc 调用 step", size=9)
    for i in range(2):
        f.rect(10 + i * 115, 150, 105, 44, "blue", text=f"模型进程 rank {i}\nModelRpcServer（rpyc）\n调度 + 前向", size=8.6)
        f.arrow(63 + i * 115, 126, 63 + i * 115, 150, dash="3 2")
    f.rect(10, 214, 220, 28, "orange", text="反分词进程", size=9.5)
    f.arrow(120, 194, 120, 214)
    # 阶段二
    f.rect(250, 40, 220, 30, "green", text="主进程：HTTP + Tokenizer", size=9.5)
    f.arrow(360, 70, 360, 92, label="ZMQ", lx=24, ly=0, lsize=9)
    f.rect(250, 92, 220, 56, "blue", text="控制进程 ControllerSingle\n内含 rank 0 的 ModelTpServer\n每步：收请求 → 广播 → step", size=8.8)
    f.rect(250, 166, 105, 40, "blue", text="进程 rank 1\nrun_tp_server", size=8.8)
    f.rect(365, 166, 105, 40, "blue", text="进程 rank 2…\nrun_tp_server", size=8.8)
    f.arrow(300, 148, 300, 166, label="gloo 广播", lx=-30, ly=0, lsize=8.5)
    f.arrow(420, 148, 420, 166)
    f.rect(250, 226, 220, 28, "orange", text="反分词进程", size=9.5)
    f.arrow(360, 206, 360, 226)
    # 阶段三
    f.rect(490, 40, 220, 30, "green", text="主进程：HTTP + Tokenizer", size=9.5)
    f.arrow(600, 70, 600, 92, label="ZMQ", lx=24, ly=0, lsize=9)
    f.rect(490, 92, 220, 30, "gray", text="DP 控制器（dp_size > 1 时）", size=9, dash="4 3")
    for i in range(2):
        f.rect(490 + i * 115, 140, 105, 52, "green", text=f"Scheduler 进程 rank {i}\nevent_loop_overlap\nTpModelWorker", size=8.6)
        f.arrow(543 + i * 115, 122, 543 + i * 115, 140)
    f.text(600, 206, "各 rank 收到相同请求，各自跑同一份调度", cls="mu", size=8.8)
    f.rect(490, 226, 220, 28, "orange", text="反分词进程", size=9.5)
    f.arrow(600, 192, 600, 226)
    f.text(360, 280, "贯穿三个阶段的决定：调度与前向在同一进程；分词 → 调度 → 反分词 → 分词的 ZMQ 环；每个 TP rank 重复执行同一份调度、只广播输入", cls="mu", size=9.5)
    f.text(360, 300, "变化的是分发：rpyc 远程调用 → 控制进程内直接调用 + gloo 广播 → 对等的调度器进程 + DP 控制器（2025 年再加上流水线并行的 PP × TP）", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-incremental-decode")
def sgl_incremental_decode():
    f = Fig(720, 250, "增量反分词：surr_offset 之前已确认输出，[surr_offset, read_offset) 是陪着一起 decode 的周围 token，read_offset 之后是新 token")
    toks = ["The", "ĠPar", "is", "Ġcap", "ital", "Ġof", "ĠFr", "ance", "Ġis"]
    x0, y0, w = 40, 70, 68
    for i, t in enumerate(toks):
        cls = "gray" if i < 5 else ("blue" if i < 7 else "orange")
        f.rect(x0 + i * w, y0, w - 4, 34, cls, rx=5)
        f.text(x0 + i * w + (w - 4) / 2, y0 + 17, t, size=10.5, family="mono")
        f.text(x0 + i * w + (w - 4) / 2, y0 - 10, str(i), cls="mu", size=9)
    f.text(x0 + 5 * w - 2, y0 + 50, "surr_offset = 5", cls="mu", size=10, anchor="middle")
    f.line(x0 + 5 * w - 2, y0 + 36, x0 + 5 * w - 2, y0 + 42, sw=1.2)
    f.text(x0 + 7 * w - 2, y0 + 50, "read_offset = 7", cls="mu", size=10, anchor="middle")
    f.line(x0 + 7 * w - 2, y0 + 36, x0 + 7 * w - 2, y0 + 42, sw=1.2)
    f.text(x0 + 2.5 * w - 2, y0 + 72, "decoded_text（已确认输出）", cls="mu", size=9.5)
    f.text(x0 + 6 * w - 2, y0 + 72, "surr_ids（周围）", cls="mu", size=9.5)
    f.text(x0 + 8 * w - 2, y0 + 72, "新 token", cls="mu", size=9.5)
    f.text(360, 168, 'read_text = decode(ids[5:]) → " of France is"     surr_text = decode(ids[5:7]) → " of"', size=10.5, family="mono")
    f.text(360, 190, 'new_text = read_text[len(surr_text):] → " France is"   不以 � 结尾才确认：surr_offset ← 7，read_offset ← 9', size=10.5, family="mono")
    f.text(360, 226, "每步只 decode 窗口里的几个 token，开销与已生成长度无关；周围 token 让空格和多字节字符的边界和整体 decode 一致（#517，2024-06-12）", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-vllm-imports")
def sgl_vllm_imports():
    f = Fig(720, 260, "srt/ 里引用 vLLM 的 import 语句数：v0.2.0 达到峰值 231 条，之后一路降到 13 条；同期 srt/ 的 Python 文件数从 31 涨到近 2000")
    data = [("v0.1.5", 21, 31), ("v0.2.0", 231, 60), ("v0.3.0", 192, 76), ("v0.4.0", 132, 137), ("v0.4.6", 49, 266), ("v0.5.0rc0", 41, 437), ("29f6d408c0", 13, 1979)]
    x0, y0, bw, gap = 70, 200, 56, 36
    f.line(x0 - 10, y0, 700, y0, sw=1.2)
    for i, (tag, lines, files) in enumerate(data):
        x = x0 + i * (bw + gap)
        h = lines / 231 * 140
        f.rect(x, y0 - h, bw, h, "blue", rx=3, sw=1)
        f.text(x + bw / 2, y0 - h - 10, str(lines), size=10.5, weight="600")
        f.text(x + bw / 2, y0 + 16, tag, cls="mu", size=9.5, family="mono")
        f.text(x + bw / 2, y0 + 32, f"{files} 个文件", cls="mu", size=9)
    f.text(360, 30, "蓝柱：`from vllm` / `import vllm` 语句数（srt/ 下）；柱下：srt/ 的 .py 文件数", cls="mu", size=10)
    f.text(360, 248, "去依赖的三波：2024-11 → 2025-01 基础层（distributed、linear、rope）；2025-03 算子进 sgl-kernel；2025-07 → 11 量化的 10 步解耦", cls="mu", size=9.5)
    return f


@figure("sglang", "sgl-cuda-graph-pad")
def sgl_cuda_graph_pad():
    f = Fig(720, 250, "CUDA Graph 回放：启动时按固定的 batch 列表各捕获一张图，运行时把实际 batch 向上取整到最近的捕获大小，多出的槽位用假请求填充，回放后截掉")
    sizes = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    x0, y0 = 40, 60
    f.text(360, 30, "捕获的 batch 大小（v0.2.0：[1, 2, 4] + [8, 16, …, 128]）", cls="mu", size=10)
    for i, s in enumerate(sizes):
        cls = "orange" if s == 16 else "gray"
        f.rect(x0 + i * 54, y0, 48, 26, cls, rx=5, text=str(s), size=10.5)
    f.text(x0 + 11 * 54 + 8, y0 + 13, "… 128", cls="mu", size=10, anchor="start")
    f.text(360, 118, "实际 batch = 13 个请求", size=10.5, weight="600")
    for i in range(16):
        cls = "blue" if i < 13 else "bx"
        f.rect(60 + i * 28, 130, 24, 24, cls, rx=4, sw=1, dash=None if i < 13 else "3 2")
    f.text(60 + 16 * 28 + 10, 142, "← 3 个假请求：seq_len=1、out_cache_loc=0", cls="mu", size=9.5, anchor="start")
    f.arrow(360, 160, 360, 180)
    f.text(360, 194, "13 个请求的 input_ids / seq_lens / out_cache_loc 写进固定缓冲区前 13 格 → graphs[16].replay() → 输出截到前 13 行", size=9.8, family="mono")
    f.text(360, 226, "固定缓冲区 + 共享的 graph 内存池：所有图共用一份地址，decode 一步从上千次 kernel 发射变成一次 replay", cls="mu", size=9.5)
    return f



@figure("sglang", "sgl-batch-trio")
def sgl_batch_trio():
    f = Fig(720, 300, "三份批次与注意力后端：ScheduleBatch 在调度层，ModelWorkerBatch 跨线程交给 worker，ForwardBatch 在 GPU 上为一次前向准备；注意力后端只看 ForwardBatch")
    cols = [(20, "调度器（managers/scheduler.py）", "green"), (260, "TpModelWorker（managers/tp_worker.py）", "blue"), (500, "ModelRunner + 注意力后端", "purple")]
    for x, title, cls in cols:
        f.text(x + 100, 24, title, size=10, weight="600")
    f.rect(20, 44, 200, 96, "green", rx=8)
    f.text(120, 62, "ScheduleBatch", size=10.5, weight="600", family="mono")
    f.text(120, 104, "reqs: List[Req]\n前缀匹配结果、准入 / 撤回状态\nnew_token_ratio、树节点锁\n（CPU 侧 Python 对象）", size=9)
    f.arrow(220, 92, 260, 92)
    f.text(240, 154, "① get_model_worker_batch()", cls="mu", size=8.5)
    f.rect(260, 44, 200, 96, "blue", rx=8)
    f.text(360, 62, "ModelWorkerBatch", size=10.5, weight="600", family="mono")
    f.text(360, 104, "input_ids、seq_lens、槽位\nsampling_info、forward_mode\n只带 worker 需要的字段\n可放进队列交给另一个线程", size=9)
    f.arrow(460, 92, 500, 92)
    f.text(480, 154, "② ForwardBatch.init_new()", cls="mu", size=8.5)
    f.rect(500, 44, 200, 96, "purple", rx=8)
    f.text(600, 62, "ForwardBatch", size=10.5, weight="600", family="mono")
    f.text(600, 104, "GPU 张量：positions、out_cache_loc\nreq_to_token_pool、token_to_kv_pool\nattn_backend 的元数据\n（每次前向重建）", size=9)
    f.rect(500, 170, 200, 70, "orange", rx=8)
    f.text(600, 188, "AttentionBackend", size=10.5, weight="600", family="mono")
    f.text(600, 220, "init_forward_metadata(forward_batch)\ninit_*_cuda_graph(...)\nforward_decode / forward_extend", size=8.8, family="mono")
    f.arrow(600, 140, 600, 170)
    f.text(360, 200, "重叠调度：调度线程继续改 ScheduleBatch，\n前向线程只拿 ModelWorkerBatch", cls="mu", size=9.5)
    f.text(360, 266, "2024-09-29 → 30：#1538 调度代码进 scheduler.py，#1541 执行层不再看 ScheduleBatch，\n#1543 InputMetadata → ForwardBatch，#1544 引入 ModelWorkerBatch，#1547 注意力后端目录", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-mla-absorb")
def sgl_mla_absorb():
    f = Fig(720, 270, "MLA decode 的两条路径：解压路径把每个历史 token 的潜向量还原成 K、V；吸收路径把解压矩阵挪到 query 与输出两侧，注意力直接在潜空间算")
    f.text(180, 24, "解压路径（prefill 常用）", size=10.5, weight="600")
    f.rect(30, 44, 130, 34, "gray", rx=6, text="历史 L 个潜向量\n[L, 512 + 64]", size=9)
    f.arrow(160, 61, 200, 61, label="× W_UK, W_UV", ly=-10, lsize=8.5)
    f.rect(200, 36, 130, 50, "orange", rx=6, text="还原 K、V\n[L, 128 头, 128]\n每 token 128×512×128 乘加", size=8.6)
    f.arrow(265, 86, 265, 112)
    f.rect(200, 112, 130, 34, "blue", rx=6, text="普通多头注意力\nq [128 头, 128]", size=9)
    f.text(180, 170, "代价 ∝ 历史长度 L", cls="mu", size=9.5)
    f.text(540, 24, "吸收路径（decode）", size=10.5, weight="600")
    f.rect(400, 44, 120, 34, "blue", rx=6, text="q_nope\n[128 头, 128]", size=9)
    f.arrow(520, 61, 560, 61, label="× w_kc（bmm）", ly=-10, lsize=8.5)
    f.rect(560, 44, 130, 34, "purple", rx=6, text="潜空间的 q\n[128 头, 512]", size=9)
    f.arrow(625, 78, 625, 104)
    f.rect(560, 104, 130, 34, "gray", rx=6, text="对 L 个潜向量做注意力\n（相当于 MQA）", size=8.8)
    f.arrow(625, 138, 625, 164)
    f.rect(560, 164, 130, 34, "orange", rx=6, text="× w_vc（bmm）→ 输出\n[128 头, 128]", size=8.8)
    f.text(540, 222, "变换 query 的代价与 L 无关；历史只存潜向量，KV 缓存只有 576 维", cls="mu", size=9.5)
    f.text(360, 252, "#905（2024-08-05）：MLATokenToKVPool + 支持 kv_lora_rank 的 Triton kernel + 加载时拆出 w_kc / w_vc；#1138 分组 decode kernel；#1285 FP8 bmm", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-overlap-timeline")
def sgl_overlap_timeline():
    f = Fig(720, 320, "重叠调度的时间线：调度线程发射第 N 批后不等结果，先处理第 N−1 批的结果、再组第 N+1 批；第 N+1 批里还没采样出的 token 用负数占位，前向线程在 GPU 上解析")
    f.text(60, 60, "调度线程", size=10.5, weight="600", anchor="start")
    f.text(60, 150, "前向线程\n（GPU 流）", size=10.5, weight="600", anchor="start")
    segs = [(150, "处理 N−1 结果", "gray"), (250, "组 N+1", "green"), (330, "发射 N+1", "blue"), (400, "处理 N 结果", "gray"), (500, "组 N+2", "green"), (580, "发射 N+2", "blue")]
    for x, label, cls in segs:
        w = 95 if "处理" in label else (75 if "组" in label else 65)
        f.rect(x, 44, w, 32, cls, rx=5, text=label, size=9.5)
    f.rect(150, 134, 175, 32, "blue", rx=5, text="前向 N（GPU 计算）", size=9.5)
    f.rect(330, 134, 245, 32, "blue", rx=5, text="前向 N+1", size=9.5)
    f.rect(580, 134, 120, 32, "blue", rx=5, text="前向 N+2 …", size=9.5)
    f.arrow(395, 76, 395, 134, dash="3 2")
    f.text(406, 100, "input_queue", cls="mu", size=8.5, anchor="start")
    f.arrow(325, 166, 400, 76, dash="3 2")
    f.text(318, 112, "copy_done 事件", cls="mu", size=8.5, anchor="end")
    f.rect(150, 196, 550, 80, "bx", rx=8, dash="4 3")
    f.text(425, 214, "未来 token：组 N+1 时第 N 批的采样结果还没出来", size=10, weight="600")
    f.text(425, 246, "调度线程把 −(ct+1) … −(ct+bs) 写进 N+1 的 input_ids；前向线程算完 N 把真实 token 写进 future_token_ids_map，\n前向 N+1 之前用 where(ids < 0, map[−ids], ids) 替换", size=9, family="mono")
    f.text(360, 300, "GPU 在整个过程中没有空隙；代价是所有读 token 的逻辑都要能处理占位符（撤回、约束解码、logprob、混批各修了一遍）", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-dp-attention")
def sgl_dp_attention():
    f = Fig(720, 280, "MLA 下的两种并行：TP 时每个 rank 都要存一份完整的潜向量 KV；DP attention 让每个 rank 各管一组请求和自己的 KV，只在 MoE 层把 token 聚到一起")
    f.text(180, 24, "张量并行（TP=4）", size=10.5, weight="600")
    for i in range(4):
        x = 30 + i * 78
        f.rect(x, 40, 70, 60, "blue", rx=6)
        f.text(x + 35, 54, f"rank {i}", size=9.5, weight="600")
        f.text(x + 35, 80, "KV 全量\n（重复）", size=8.8)
    f.text(180, 120, "注意力按头切，但 MLA 只有一个潜向量：4 份一样的 KV", cls="mu", size=9.3)
    f.text(540, 24, "DP attention（DP=4）", size=10.5, weight="600")
    for i in range(4):
        x = 400 + i * 78
        f.rect(x, 40, 70, 60, "green", rx=6)
        f.text(x + 35, 54, f"rank {i}", size=9.5, weight="600")
        f.text(x + 35, 80, f"请求组 {i}\n自己的 KV", size=8.8)
    f.text(540, 120, "各 rank 独立调度和注意力；KV 不重复，batch 可以大 4 倍", cls="mu", size=9.3)
    f.rect(400, 150, 300, 36, "orange", rx=6, text="MoE / 稠密 FFN：all-gather 各 rank 的 token → 一起算 → 散回", size=9)
    for i in range(4):
        f.arrow(435 + i * 78, 100, 435 + i * 78, 150, dash="3 2")
    f.text(360, 214, "每步同步：all_gather 各 rank 的 token 数（gather 形状）；没活的 rank 用 IDLE batch 陪跑；all_reduce(MIN) 确认都在 decode 才回放 CUDA Graph", cls="mu", size=9.3)
    f.text(360, 236, "#1970（2024-11-16）Support DP MLA、#2061 CUDA Graph、#2096 更保守的准入；v0.4 博客：DeepSeek decode 吞吐 1.9×", cls="mu", size=9.3)
    f.text(360, 262, "EP（#2371，12-06）再把 MoE 的专家切到不同的 rank：DP attention + EP 成为 DeepSeek 部署的标准形态", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-kernel-layers")
def sgl_kernel_layers():
    f = Fig(720, 292, "sgl-kernel 在栈里的位置：Python 层调用 torch.ops 里注册的算子，C++ 侧按算子类型分目录，主力 GEMM / 注意力来自 CUTLASS、FlashInfer、DeepGEMM、FlashMLA 等外部库")
    f.rect(30, 30, 660, 36, "green", rx=8, text="srt/layers：linear、moe、quantization、sampler、attention 后端 …（Python）", size=10)
    f.arrow(360, 66, 360, 92, label="torch.ops.sgl_kernel.<op>（TORCH_LIBRARY 注册，#3130）", ly=0, lx=190, lsize=8.5)
    f.rect(30, 92, 660, 36, "blue", rx=8, text="sgl-kernel 的 Python 包装（elementwise / gemm / moe / sampling / speculative …）", size=10)
    f.arrow(360, 128, 360, 152)
    dirs = ["allreduce", "attention", "elementwise", "gemm", "moe", "speculative", "cpu"]
    for i, d in enumerate(dirs):
        f.rect(30 + i * 94, 152, 86, 30, "purple", rx=5, text=d, size=9.5, tcls="tx")
    f.text(360, 198, "csrc/ 按算子类型分目录（#4025 / #4027，2025-03）", cls="mu", size=9.3)
    f.arrow(360, 206, 360, 226)
    ext = ["CUTLASS", "FlashInfer", "DeepGEMM", "FlashMLA", "自写融合算子"]
    for i, e in enumerate(ext):
        f.rect(70 + i * 120, 226, 108, 28, "orange" if e != "自写融合算子" else "gray", rx=5, text=e, size=9.5)
    f.text(360, 270, "2024-11-30 建目录 → 12-01 PyPI + warp 归约示例 → 12-06 FP8 算子搬入 → 2025-01 TORCH_LIBRARY\n→ 2025-03 分目录、DeepGEMM、CMake → 2026-07 搬进 python/sglang/kernels/", cls="mu", size=9)
    return f



@figure("sglang", "sgl-eagle-worker")
def sgl_eagle_worker():
    f = Fig(720, 290, "EAGLEWorker 的一步 decode：草稿模型跑 num_steps 步展开成树，目标模型一次前向验证整棵树，接受最长路径后草稿模型再 extend 一次；调度器只看到“一步返回多个 token”")
    f.rect(20, 40, 150, 50, "green", rx=8, text="Scheduler\nforward_batch_generation(batch)", size=9)
    f.arrow(170, 65, 215, 65)
    f.rect(215, 28, 485, 180, "bx", rx=10, dash="4 3")
    f.text(457, 44, "EAGLEWorker(TpModelWorker)", size=10.5, weight="600", family="mono")
    f.rect(230, 62, 130, 48, "blue", rx=6, text="draft\n草稿模型 × num_steps\n每步 top-k → 树", size=8.8)
    f.arrow(360, 86, 395, 86)
    f.rect(395, 62, 140, 48, "orange", rx=6, text="verify\n目标模型一次前向\n树形掩码、取最长接受路径", size=8.8)
    f.arrow(535, 86, 570, 86)
    f.rect(570, 62, 120, 48, "purple", rx=6, text="draft extend\n用接受的 token\n更新草稿模型状态", size=8.8)
    f.rect(230, 128, 460, 30, "gray", rx=6, text="槽位：草稿前一次申请 batch × topk × num_steps 个并备份分配器状态，验证后只保留接受路径、其余回滚", size=8.8)
    f.rect(230, 166, 220, 30, "gray", rx=6, text="目标 TpModelWorker + 草稿 ModelRunner", size=8.8)
    f.rect(470, 166, 220, 30, "gray", rx=6, text="草稿模型自己的 CUDA Graph", size=8.8)
    f.arrow(215, 120, 170, 120, label="logits、接受的 token、接受数", ly=12, lsize=8.5)
    f.text(360, 234, "part 1 草稿模型文件（#2640）→ part 2 修 CUDA Graph 与 DP attention（#2684）→ part 3 调度器处理“一步多 token”（#2709）→ part 4 worker（#2150，2025-01-02）", cls="mu", size=9.3)
    f.text(360, 256, "同一接口后来接入 MTP / NextN（#3582）、n-gram、spec v2（草稿与验证纳入重叠调度）以及 2026 年的新方法：speculative/ 下 60 多个文件", cls="mu", size=9.3)
    f.text(360, 278, "撤回要连树上的槽位一起释放（#2711）；DP attention 的 gather 形状要按 token 数而不是请求数（#2684）", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-hicache-tiers")
def sgl_hicache_tiers():
    f = Fig(720, 306, "三层缓存：GPU 池里的 value、CPU 池里的 host_value、远端存储里按内容哈希索引的页；HiCacheController 的写线程与读线程异步搬运")
    f.rect(20, 40, 200, 110, "blue", rx=8)
    f.text(120, 58, "GPU：RadixCache 节点的 value", size=10, weight="600")
    f.text(120, 100, "命中 → 直接用\n显存满 → 叶子 LRU 淘汰\n备份过的节点只释放显存\n（节点留在树上，value 置空）", size=9)
    f.rect(260, 40, 200, 110, "green", rx=8)
    f.text(360, 58, "CPU：节点的 host_value", size=10, weight="600")
    f.text(360, 100, "与 GPU 池同布局的锁页内存\nwrite_through / selective / write_back\n命中 ≥ load_back_threshold 才搬回\n内存满 → evict_host", size=9)
    f.rect(500, 40, 200, 110, "orange", rx=8)
    f.text(600, 58, "存储：按内容哈希的页", size=10, weight="600")
    f.text(600, 100, "3FS、Mooncake Store、NIXL、文件……\n键 = 链式页哈希（含完整前缀）\n跨实例共享，prefill 可复用\n预取（storage_prefetch）", size=9)
    f.arrow(220, 80, 260, 80, label="write", ly=-9, lsize=8.5)
    f.arrow(260, 110, 220, 110, label="load_back", ly=12, lsize=8.5)
    f.arrow(460, 80, 500, 80, label="set", ly=-9, lsize=8.5)
    f.arrow(500, 110, 460, 110, label="get / prefetch", ly=12, lsize=8.5)
    f.rect(140, 180, 440, 60, "bx", rx=8, dash="4 3")
    f.text(360, 198, "HiCacheController（managers/cache_controller.py）", size=10, weight="600")
    f.text(360, 224, "写队列 + 写线程、读队列 + 读线程；CacheOperation 可合并 / 拆分；LayerDoneCounter 让前向逐层等待；拷贝走独立的 CUDA 流", size=9)
    f.text(360, 262, "#2771 CPU 池（2025-01-07）→ #2804 控制器（01-10）→ #2693 主 PR（02-23）→ #7704 存储层原型（07-18）\n→ #9053 节点哈希（08-11）→ #10190 淘汰策略插件化（09）", cls="mu", size=9.3)
    f.text(360, 290, "磨合：TP 一致性（#4082）、MLA 池（#4009）、页对齐（#4581）、DP attention（#7159）、PD 复用远端缓存（#8211 系列）", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-pd-flow")
def sgl_pd_flow():
    f = Fig(720, 320, "PD 分离下一个请求的路径：负载均衡器分配 bootstrap room 同时发两边；decode 侧先握手、预分配 KV；prefill 侧按 chunk 算完就用 RDMA 单边写发过去；传完 decode 侧做预构建 extend 进入普通循环")
    f.rect(20, 30, 110, 40, "gray", rx=7, text="客户端", size=10)
    f.arrow(130, 50, 180, 50)
    f.rect(180, 30, 150, 40, "purple", rx=7, text="负载均衡器\n选一对 P/D，分配 room id", size=8.8)
    f.elbow([(255, 70), (255, 100), (130, 100), (130, 120)])
    f.elbow([(305, 70), (305, 100), (560, 100), (560, 120)])
    f.rect(20, 120, 300, 130, "blue", rx=9)
    f.text(170, 138, "prefill 服务器（prefill.py）", size=10, weight="600")
    f.rect(35, 152, 270, 26, "bx", rx=5, text="PrefillBootstrapQueue：等 decode 侧握手完成", size=8.8)
    f.rect(35, 184, 270, 26, "bx", rx=5, text="普通调度：每个 chunk 算完 → send_kv_chunk", size=8.8)
    f.rect(35, 216, 270, 26, "bx", rx=5, text="inflight 队列：传输完成才释放 KV", size=8.8)
    f.rect(400, 120, 300, 130, "green", rx=9)
    f.text(550, 138, "decode 服务器（decode.py）", size=10, weight="600")
    f.rect(415, 152, 270, 26, "bx", rx=5, text="DecodePreallocQueue：握手、预分配整段 KV、告知地址", size=8.5)
    f.rect(415, 184, 270, 26, "bx", rx=5, text="DecodeTransferQueue：轮询 receiver 直到传完", size=8.8)
    f.rect(415, 216, 270, 26, "bx", rx=5, text="预构建 extend → 普通 decode 循环", size=8.8)
    f.arrow(320, 197, 400, 197, label="RDMA 写", ly=-9, lsize=8.5)
    f.arrow(400, 165, 320, 165, label="握手", ly=-9, lsize=8.5, dash="3 2")
    f.text(360, 213, "Mooncake / NIXL", cls="mu", size=8)
    f.text(360, 274, "传输接口 base/conn.py：BaseKVManager / BaseKVSender（init、send、poll）/ BaseKVReceiver（init、poll）/ BaseKVBootstrapServer——每个后端一个子目录", cls="mu", size=9)
    f.text(360, 294, "#4654（2025-03-21）1410 行 → #4880 Mooncake → #5328 后端抽象 → #5477 NIXL → #5608 / #5609 重叠调度 → #5435 DP attention + DeepEP（04-23）", cls="mu", size=9)
    f.text(360, 312, "博客（05-05）的三个动机：prefill 打断 decode、DP attention 失衡、DeepEP 两种模式不能共存", cls="mu", size=9)
    return f


@figure("sglang", "sgl-large-ep")
def sgl_large_ep():
    f = Fig(720, 300, "大规模 EP 的两套配置：prefill 用 normal dispatch + contiguous GEMM + 大 batch，decode 用 low-latency dispatch + masked GEMM + CUDA Graph；TBO 让两个 micro-batch 的通信与计算交错；EPLB 用冗余专家平衡负载")
    f.rect(20, 36, 330, 118, "blue", rx=9)
    f.text(185, 54, "prefill（博客：4 节点，EP32）", size=10, weight="600")
    f.text(185, 104, "DeepEP normal dispatch：按实际 token 数 all-to-all，动态形状\nDeepGEMM contiguous：Triton permute 后的连续布局\n每卡 16384 token 的大 batch；不能进 CUDA Graph\n先提交 GPU 计算、再做阻塞 CPU 的 dispatch", size=9)
    f.rect(370, 36, 330, 118, "green", rx=9)
    f.text(535, 54, "decode（博客：9 节点，EP72）", size=10, weight="600")
    f.text(535, 104, "DeepEP low-latency：固定大小的 RDMA 缓冲区\nDeepGEMM masked：固定形状 + 掩码\n每卡 128–256 序列；CUDA Graph 回放\nDeepEPMode.AUTO 按角色选模式（PD 分离是前提）", size=9)
    f.rect(20, 172, 330, 58, "orange", rx=8)
    f.text(185, 188, "TBO 双 batch 重叠（two_batch_overlap.py，#4068）", size=9.8, weight="600")
    f.text(185, 212, "batch 切两半；A 算 attention / MLP 时 B 做 dispatch / combine\nprefill +27–35%，decode +25.5%；峰值显存减半", size=8.8)
    f.rect(370, 172, 330, 58, "purple", rx=8)
    f.text(535, 188, "EPLB 专家负载均衡（eplb/，#6387 → #6469）", size=9.8, weight="600")
    f.text(535, 212, "统计每个专家的 token 数 → 冗余专家（256 → 288）→ 重排映射\n静态一次 / 动态周期重平衡；prefill 1.49×、decode 2.54×", size=8.8)
    f.text(360, 256, "2025-05-05 博客：96 张 H100 上的 DeepSeek-V3，每节点每秒 52.3k 输入 / 22.3k 输出 token，输出 $0.20 / 百万 token，比纯 TP 快最多 5 倍", cls="mu", size=9.3)
    f.text(360, 278, "三个库都来自 DeepSeek 开源周（DeepEP、DeepGEMM、EPLB 算法），SGLang 做的是调度与集成：三个月从 EP 支持（#3602）到博客", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-backend-matrix")
def sgl_backend_matrix():
    f = Fig(720, 300, "注意力后端按来源分组：外部 kernel 库、MLA 专用、各家硬件、特殊结构；同一个 AttentionBackend 接口，方法数随功能从 8 个长到 22 个")
    groups = [(20, "外部 kernel 库", "blue", ["flashinfer", "flashattention (FA3)", "triton", "torch_native"]),
              (195, "MLA 专用", "purple", ["flashmla", "cutlass_mla", "trtllm_mla", "flashinfer_mla"]),
              (370, "各家硬件", "orange", ["aiter (AMD)", "ascend (昇腾)", "intel_amx / xpu (CPU)", "tpu → sglang-jax"]),
              (545, "特殊结构", "green", ["nsa / dsa（稀疏索引）", "hybrid（线性 + 全注意力）", "dual_chunk（长上下文）", "double_sparsity"])]
    for x, title, cls, items in groups:
        f.rect(x, 36, 155, 150, cls, rx=9, sw=1.2)
        f.text(x + 77, 54, title, size=10, weight="600")
        for i, it in enumerate(items):
            f.rect(x + 10, 68 + i * 29, 135, 24, "bx", rx=5, text=it, size=8.8)
    f.text(360, 210, "2024-09 两个后端（FlashInfer、Triton）→ 2025-03 FA3、FlashMLA、页大小 > 1 → 2025-08 v0.5.0rc0 十几个 → 2026-10 三十多个后端文件；layers/attention/ 139 个文件", cls="mu", size=9.3)
    f.text(360, 232, "接口的生长：CUDA Graph 的捕获 / 回放元数据（2024-09）→ 投机解码的 draft / verify 元数据 → forward_mixed → 共享前缀读 → 可分段的图捕获 → 稀疏索引元数据", cls="mu", size=9.3)
    f.text(360, 262, "页大小：初版每页 1 个 token（树可任意切分）；#4356（2025-03-12）新分配器 + 页对齐的树，随后每个功能逐一适配（PD、FA3、EAGLE、撤回、HiCache）", cls="mu", size=9.3)
    f.text(360, 284, "多硬件：2024-09 第一个 AMD 提交 → 散落的 is_hip() 分支 → hardware_backend/（95 个文件）与 platforms/ 集中平台逻辑", cls="mu", size=9.3)
    return f



@figure("sglang", "sgl-entry-layers")
def sgl_entry_layers():
    f = Fig(720, 300, "入口层服务的四类客户：HTTP 客户端走 http_server 与 OpenAI / Anthropic 兼容层，网关走 gRPC 直连调度器，离线脚本与 RL 框架直接用 Engine；Engine 是唯一的子进程拉起点")
    clients = [(20, "OpenAI SDK / curl", "green"), (195, "Rust 网关", "purple"), (370, "离线脚本", "gray"), (545, "RL 框架（veRL、slime）", "orange")]
    for x, name, cls in clients:
        f.rect(x, 30, 155, 30, cls, rx=7, text=name, size=9.5)
    f.rect(20, 92, 155, 56, "green", rx=7, text="http_server.py\nopenai/serving_*、anthropic/\n86 个路由（REF）", size=8.8)
    f.rect(195, 92, 155, 56, "purple", rx=7, text="grpc_server（#10283）\n分词与模板在网关侧\nproto/ 定义", size=8.8)
    f.rect(370, 92, 155, 56, "gray", rx=7, text="Engine.generate()\nengine.py / EngineBase", size=8.8)
    f.rect(545, 92, 155, 56, "orange", rx=7, text="Engine 嵌入训练进程（SPMD）\nupdate_weights_* / release_memory", size=8.5)
    for x, _, _ in clients:
        f.arrow(x + 77, 60, x + 77, 92)
    f.rect(20, 176, 680, 34, "blue", rx=8, text="Engine（entrypoints/engine.py）：唯一的子进程拉起点 → TokenizerManager → 调度器进程 × (PP × TP) → 反分词进程", size=9.5)
    for x in (97, 447, 622):
        f.arrow(x, 148, x, 176)
    f.elbow([(272, 148), (272, 162), (120, 162), (120, 176)], dash="3 2")
    f.text(272, 228, "gRPC 入口绕过 Python 的 HTTP 层，但调度器进程仍由 Engine 拉起", cls="mu", size=9)
    f.text(360, 258, "#2996（2025-01-19）Engine 与 HTTP 分离 → #7167（06-16）OpenAI 层重写 4424 行 → function_call/ 按模型族解析 → #10283（09-11）gRPC 服务器", cls="mu", size=9.3)
    f.text(360, 280, "路由数：3（v0.1.5）→ 7（v0.2.0）→ 29（v0.4.0）→ 41（v0.4.6）→ 49（v0.5.0rc0）→ 86（2026-10）", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-gateway-layers")
def sgl_gateway_layers():
    f = Fig(720, 304, "网关在系统里的位置：客户端 → sgl-model-gateway（协议转换、解析、策略选实例、服务发现、可观测性）→ 多个引擎实例（HTTP 或 gRPC）→ 实例内的 DP 控制器与调度器")
    f.rect(20, 30, 680, 26, "gray", rx=6, text="客户端：OpenAI / Anthropic / 原生协议", size=9.5)
    f.arrow(360, 56, 360, 76)
    f.rect(20, 76, 680, 96, "purple", rx=9)
    f.text(360, 94, "sgl-model-gateway（Rust，约 9.5 万行，独立发版 gateway-v*）", size=10, weight="600")
    cols = [("routers/\nHTTP、PD、gRPC、OpenAI", 40), ("policies/\n轮询、随机、缓存感知、\npower-of-two、PD 组合", 180), ("解析器\n工具调用、推理段\n（Rust 实现）", 320), ("service_discovery\nKubernetes 标签\n动态加减实例", 460), ("observability / wasm\n指标、追踪\n插件层", 600)]
    for text, x in cols:
        f.rect(x - 10, 108, 130, 56, "bx", rx=6, text=text, size=8.5)
    for x in (120, 360, 600):
        f.arrow(x, 172, x, 198)
    f.text(240, 186, "HTTP 或 gRPC（#10283）", cls="mu", size=8.5)
    for i, x in enumerate((20, 260, 500)):
        f.rect(x, 198, 200, 40, "green" if i < 2 else "blue", rx=7, text=f"引擎实例 {i}\nDP 控制器 → 调度器 × (PP × TP)" if i < 2 else "prefill / decode 实例\n（PD 感知路由）", size=8.8)
    f.text(360, 258, "2024-10 rust/（轮询、随机）→ 11 近似基数树的缓存感知 → 12 改名 sgl-router、0.1.0 动态扩缩\n→ 2025-07 #7987 策略与路由分离 → 08 PD 路由、解析器 → 12 改名 sgl-model-gateway", cls="mu", size=9)
    f.text(360, 288, "rust/ 下另有六个按需编译的 crate（基数树核心、模板渲染、多模态、gRPC、服务层）嵌进 Python 包：Rust 的第二种身份", cls="mu", size=9)
    return f


@figure("sglang", "sgl-rl-loop")
def sgl_rl_loop():
    f = Fig(720, 296, "RL 训练闭环里的推理引擎：生成 rollout → 让出显存 → 训练一步 → 更新权重 → 恢复显存 → 再生成；三种更新方式对应三种部署形态")
    steps = [(20, "generate\nrollout（batch）", "green"), (160, "release_memory\n_occupation", "orange"), (300, "训练一步\n（FSDP / Megatron）", "gray"), (440, "update_weights\n_from_tensor / distributed", "blue"), (580, "resume_memory\n_occupation", "orange")]
    for x, text, cls in steps:
        f.rect(x, 40, 120, 48, cls, rx=7, text=text, size=8.8)
    for x in (140, 280, 420, 560):
        f.arrow(x, 64, x + 20, 64)
    f.elbow([(640, 88), (640, 112), (80, 112), (80, 88)])
    f.text(360, 104, "每个训练步循环一次", cls="mu", size=8.5)
    f.rect(20, 136, 220, 70, "bx", rx=7, dash="4 3", text="共置（SPMD）：引擎在训练进程里\n（verl_engine.py，2025-03 → 06）\n权重用 update_weights_from_tensor（分桶）", size=8.8)
    f.rect(250, 136, 220, 70, "bx", rx=7, dash="4 3", text="分离：训练卡 ↔ 推理卡\nupdate_weights_from_distributed\n（同一 torch.distributed 组广播）", size=8.8)
    f.rect(480, 136, 220, 70, "bx", rx=7, dash="4 3", text="服务化：HTTP 远程 rollout（#4848）\n或从磁盘 / checkpoint engine\n重新加载（#1157、#11755）", size=8.8)
    f.text(360, 232, "显存释放要与 CUDA Graph 兼容（#2630）：只归还物理页、虚拟地址不变，恢复后图继续回放", cls="mu", size=9.3)
    f.text(360, 252, "2024-08 #1157 不重启换权重 → 12 #2279 from_distributed、#2631 from_tensor → 2025-01 #2630 释放 / 恢复\n→ 03 #3852 SPMD + veRL（06 移除包装类）→ 07 weight_sync/ → 2026 weight_cache/", cls="mu", size=9)
    f.text(360, 282, "slime、AREAL、veRL 在 2025 Q3 路线图的合作名单里；weight_sync/ 的分桶传输来自 RL 引擎、抽回了本体", cls="mu", size=9)
    return f


@figure("sglang", "sgl-two-runtimes")
def sgl_two_runtimes():
    f = Fig(720, 300, "一个仓库、两个运行时：srt/ 的 LLM 运行时与 multimodal_gen/ 的扩散运行时各有调度、缓存与模型层，共享 kernel、分布式、平台层与接口风格")
    f.rect(20, 36, 330, 150, "blue", rx=9)
    f.text(185, 54, "srt/：LLM 运行时", size=10.5, weight="600")
    f.text(185, 112, "生成单位：token，循环到结束\n连续批处理，按 token 预算\nKV 缓存 + 基数树前缀缓存\nTP / EP / DP attention / PD / 投机解码\nmultimodal/：视觉语言模型的预处理器（95 个文件）\ndllm/：扩散式文本生成，复用这套运行时", size=8.8)
    f.rect(370, 36, 330, 150, "orange", rx=9)
    f.text(535, 54, "multimodal_gen/：扩散运行时（SGLang Diffusion）", size=10.5, weight="600")
    f.text(535, 112, "生成单位：一张图 / 一段视频，固定步数\n按图组 batch；潜变量固定大小，无 KV\n特征缓存（跨步）、条件缓存（跨请求）\n序列并行、CFG 并行、流水线按阶段\npipelines / scheduler_client / layers / models / loader\n#12484（2025-11-06）249 个文件、6.4 万行并入", size=8.8)
    f.rect(20, 206, 680, 34, "green", rx=8, text="共享：sgl-kernel（kernels/）、distributed/、platforms/ 与 hardware_backend/、CUDA Graph 的做法、OpenAI 风格接口、launch_server 与 CLI", size=9.3)
    f.arrow(185, 186, 185, 206); f.arrow(535, 186, 535, 206)
    f.text(360, 262, "为什么另起炉灶：两种负载在生成单位、批处理、状态、缓存复用、并行方式、瓶颈上全部不同；复用边界放在下层更现实", cls="mu", size=9.3)
    f.text(360, 284, "CI 按目录触发（#12940）；两边都在接 Rust 的多模态预处理（rust/sglang-mm）", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-codebase-2026")
def sgl_codebase_2026():
    f = Fig(720, 330, "srt/ 的主要目录按文件数排列，颜色是目录的出生时期：2024 年的骨架仍是最大的几个，2025 下半年之后的目录多与可靠性、可观测与平台化有关")
    data = [("layers", 400, "blue"), ("models", 287, "blue"), ("mem_cache", 156, "blue"), ("multimodal", 95, "green"), ("hardware_backend", 95, "orange"),
            ("debug_utils", 91, "orange"), ("configs", 79, "blue"), ("arg_groups", 79, "purple"), ("entrypoints", 66, "green"), ("utils", 65, "blue"),
            ("speculative", 62, "green"), ("model_executor", 58, "blue"), ("managers", 53, "blue"), ("kv_canary", 50, "purple"), ("function_call", 47, "green"),
            ("lora", 46, "green"), ("disaggregation", 36, "green"), ("distributed", 30, "green"), ("compilation", 13, "orange"), ("elastic_ep", 3, "orange")]
    x0, y0, bw, gap = 30, 220, 30, 4
    for i, (name, n, cls) in enumerate(data):
        x = x0 + i * (bw + gap)
        h = n / 400 * 160
        f.rect(x, y0 - h, bw, h, cls, rx=3, sw=1)
        f.text(x + bw / 2, y0 - h - 8, str(n), size=8.5)
        f.el.append(f'<text x="{x + bw / 2:.1f}" y="{y0 + 6}" class="mu" font-size="8.5" text-anchor="end" transform="rotate(-55 {x + bw / 2:.1f} {y0 + 6})">{name}</text>')
    legend = [("2024 骨架", "blue"), ("2025 上半年规模化", "green"), ("2025 下半年可靠性", "orange"), ("2026 平台化", "purple")]
    for i, (t, cls) in enumerate(legend):
        f.rect(30 + i * 170, 24, 14, 14, cls, rx=3, sw=1)
        f.text(52 + i * 170, 31, t, cls="mu", size=9.5, anchor="start")
    f.text(360, 300, "基准提交：srt/ 1979 个 .py 文件；2024 / 2025 / 2026 的提交 1607 / 6766 / 10831，作者 189 / 796 / 1214；v0.5 系列 50 个 tag，2026 年约每两三周一个", cls="mu", size=9.3)
    f.text(360, 320, "另有 kernels/（2026-07 从 sgl-kernel 搬入）、rust_extensions/（按需编译的 Rust）、scheduler_components/（拆分 scheduler.py）", cls="mu", size=9.3)
    return f



@figure("sglang", "sgl-ten-decisions")
def sgl_ten_decisions():
    f = Fig(720, 340, "十个设计决定按做出的时期分三组：初始提交定下的（树与池、预估准入、同进程）、2024 下半年性能工程期的（重复调度、分层、CPU 隐藏、注册表）、2025 年之后反复验证的方法论（先借后还、先简单后统一、不碰主干）")
    x0, x1, y = 40, 690, 50
    f.line(x0, y, x1, y, sw=2); f.head(x1 + 2, y, 0)
    for lbl, frac in [("2024-01", 0.0), ("2024-07", 0.17), ("2025-01", 0.34), ("2025-07", 0.51), ("2026-01", 0.68), ("2026-10", 0.93)]:
        x = x0 + frac * (x1 - x0)
        f.line(x, y - 5, x, y + 5, sw=1.2); f.text(x, y + 18, lbl, cls="mu", size=9.5)
    cols = [("blue", 0.0, 0.0, "初始提交（2024-01）就定下", ["① 树与池分两级", "② 预估准入、估错撤回", "③ 调度与前向同进程"]),
            ("green", 0.16, 0.24, "2024 下半年：性能工程期", ["④ 每个 rank 重复调度 (#646)", "⑤ 分目录、批次三层 (#807/#1543)", "⑥ CPU 藏到 GPU 后面 (#612/#2067)", "⑦ 接口 + 注册表 (#1381/#1547)"]),
            ("orange", 0.30, 0.42, "2025 起：反复验证的方法论", ["⑧ 先借后还：sgl-kernel (#2261)", "⑨ 先简单后统一：页大小 (#4356)", "⑩ worker / mixin 不碰主干 (#2150/#4654)"])]
    for i, (cls, fa, fb, head, items) in enumerate(cols):
        cx = 20 + i * 230; mid = cx + 110
        xa, xb = x0 + fa * (x1 - x0), x0 + fb * (x1 - x0)
        if fb > fa:
            f.line(xa, y, xb, y, cls + "-l", sw=5)
        f.circle(xa, y, 5, cls + "-s", sw=1)
        if fb > fa:
            f.circle(xb, y, 5, cls + "-s", sw=1)
        f.line((xa + xb) / 2, y + 7, mid, 84, sw=0.9, dash="3 2")
        f.rect(cx, 84, 220, 24, cls + "-s", rx=6, sw=1, text=head, size=9.4)
        for j, t in enumerate(items):
            f.rect(cx, 116 + j * 34, 220, 26, cls, rx=5, sw=1, text=t, size=9)
    f.text(360, 282, "蓝：初始提交就定下、没动过；绿：2024 下半年的性能工程期；橙：2025 年起反复验证的方法论", cls="mu", size=9.3)
    f.text(360, 300, "被换掉的决定（rpyc、nonzero 分配、自研 jump-forward、vLLM 层、VerlEngine）不在图上", cls="mu", size=9.3)
    f.text(360, 318, "依赖：① 撑起后来的所有池；③ + ④ 决定了 ⑥ 只能靠线程级重叠；⑤ 之后才有 ⑦ 和 ⑩；⑧ 之后才有大规模 EP 的 kernel 归宿", cls="mu", size=9.3)
    return f


@figure("sglang", "sgl-archaeology")
def sgl_archaeology():
    f = Fig(720, 260, "本书的考古流程：固定基准 → 按月提交量找活跃期 → 模块 × 季度热力表定位目录 → 第一次出现（目录、文件、关键词、-S）→ 读当时的版本 → 读 PR、路线图与博客补“为什么”")
    steps = [("固定基准\nREF=29f6d408c0\nmerge-base 验证", "gray"), ("数提交\nlog --format=%ad\n按月 / 年 / 作者", "blue"), ("热力表\nlog --name-only\n模块 × 季度", "blue"),
             ("第一次出现\nlog -- dir / --follow\ngrep 标题 / -S", "green"), ("读当时的版本\nshow 提交:路径\nshow --stat -M", "green"), ("读人写的\nPR、路线图 issue\n博客、文件头", "orange")]
    for i, (text, cls) in enumerate(steps):
        x = 20 + i * 116
        f.rect(x, 40, 104, 70, cls, rx=8, text=text, size=8.6)
        if i < 5:
            f.arrow(x + 104, 75, x + 116, 75)
    f.rect(20, 136, 680, 44, "bx", rx=8, dash="4 3", text="核对：check_code.py 在克隆上重跑每个脚本、按 title=\"路径 @ 提交 L起-止\" 重新截取引用的代码，逐行比对——换一个 REF，数字更新；换一个仓库，方法不变", size=9)
    f.text(360, 206, "坑：tag 不一定在 main 上（SGLang 的发布分支）；--diff-filter=A 看不到改名进入的文件（用 -M / --follow）；PR 号的时间和合入时间可差数月", cls="mu", size=9.3)
    f.text(360, 228, "squash 合并时提交数 ≈ PR 数；rebase 合并的仓库要 --first-parent", cls="mu", size=9.3)
    return f



@figure("llm", "bpe-lookup")
def bpe_lookup():
    f = Fig(720, 470, "分词与嵌入的全过程：文字 → UTF-8 字节 → 预切分 → 按合并规则表合并 → token id → 取嵌入矩阵的第 id 行。以 Qwen3-0.6B 处理“我想学推理 hello”为例，id 与规则排名都取自它的 tokenizer.json")
    steps = [("我想学推理 hello", "gray"), ("UTF-8 字节\n21 个", "blue"), ("预切分\n我想学推理 | ␣hello", "purple"),
             ("BPE 合并\n我想 学 推理 ␣hello", "orange"), ("token id\n104100 47764\n113272 23811", "green"), ("嵌入矩阵\n取第 id 行\n→ 4 × 1024", "red")]
    for i, (t, cls) in enumerate(steps):
        x = 10 + i * 119
        f.rect(x, 14, 104, 62, cls, rx=8, text=t, size=9.6)
        if i < 5:
            f.arrow(x + 104, 45, x + 119, 45, sw=1.2)

    f.text(225, 100, "放大“BPE 合并”：␣hello 怎么变成一个 token（Ġ 是空格的替身字符）", cls="tx", size=10.5, weight="600")
    rows = [("开始：每个字节查词表", [("Ġ", 220), ("h", 71), ("e", 68), ("l", 75), ("l", 75), ("o", 78)], -1),
            ("排名 45：e + l", [("Ġ", 220), ("h", 71), ("el", 301), ("l", 75), ("o", 78)], 2),
            ("排名 49：Ġ + h", [("Ġh", 305), ("el", 301), ("l", 75), ("o", 78)], 0),
            ("排名 129：l + o", [("Ġh", 305), ("el", 301), ("lo", 385)], 2),
            ("排名 4535：el + lo", [("Ġh", 305), ("ello", 4791)], 1),
            ("排名 23555：Ġh + ello", [("Ġhello", 23811)], 0)]
    for r, (label, chips, new) in enumerate(rows):
        y = 124 + r * 38
        f.text(132, y, label, cls="mu", size=9.6, anchor="end")
        x = 140
        for k, (piece, tid) in enumerate(chips):
            w = max(40, 14 + 9 * len(piece))
            cls = "green" if r == len(rows) - 1 else ("orange" if k == new else "bx")
            f.rect(x, y - 15, w, 30, cls, rx=6, sw=1.2)
            f.text(x + w / 2, y - 5, piece, cls="tx", size=10.5, weight="600", family="mono")
            f.text(x + w / 2, y + 8, str(tid), cls="mu", size=8.4, family="mono")
            x += w + 6

    tables = [("词表：片段 → id（151643 条）\n'Ġ' → 220，'Ġhello' → 23811", "blue"),
              ("合并规则表：(左 id, 右 id) → (排名, 新 id)\n(68, 75) → (45, 301)，即 e + l → el", "orange"),
              ("反向词表：id → 片段（解码时用）\n23811 → 'Ġhello'", "green")]
    for i, (t, cls) in enumerate(tables):
        f.rect(455, 108 + i * 60, 255, 50, cls, rx=8, text=t, size=9.2)
    f.rect(455, 288, 255, 52, "bx", rx=8, dash="4 3", text="三张都是哈希表。合并时用小顶堆按排名取下一对，\n双向链表拼接两块；合并完的结果按原文缓存", size=9)

    f.text(360, 362, "最后一步：按 id 取嵌入矩阵的一行（151936 × 1024，BF16，每行 2048 字节）", cls="tx", size=10.5, weight="600")
    f.rect(20, 376, 680, 28, "bx", rx=4, sw=1.1)
    for x, t in ((20, "第 0 行"), (92, "第 1 行"), (164, "第 2 行")):
        f.rect(x, 376, 72, 28, "bx", rx=4, sw=1.1, text=t, size=9)
    f.text(290, 390, "…", cls="mu", size=12)
    f.rect(380, 376, 120, 28, "red", rx=4, sw=1.6, text="第 23811 行", size=9.5)
    f.text(600, 390, "…", cls="mu", size=12)
    f.arrow(380, 418, 380, 406, cls="red-l", hcls="red-s", sw=1.2)
    f.text(360, 432, "地址 = 起始地址 + id × 1024 × 2 字节；第 23811 行从第 48764928 字节开始，连续读出 2048 字节", cls="mu", size=9.6)
    f.text(360, 452, "整个查表只做一次乘法来算地址，其余都是读内存；“one-hot 向量 × 矩阵”只是数学上等价的写法", cls="mu", size=9.6)
    return f



# ====================================================================== SGLang-Omni 源码导读
@figure("omni", "omni-position")
def omni_position():
    f = Fig(720, 320, "SGLang 与 SGLang-Omni 的分工：omni 管多阶段流水线，自回归的 stage 借用 SGLang 的调度和执行")
    f.text(133, 20, "SGLang：一个进程组服务一个自回归模型", size=11, weight="600")
    steps = [("TokenizerManager（分词）", "gray"), ("Scheduler（调度、KV、Radix）", "blue"),
             ("TpModelWorker / ModelRunner", "orange"), ("Detokenizer（反分词）", "gray")]
    for i, (t, c) in enumerate(steps):
        f.rect(28, 38 + i * 58, 210, 38, c, text=t, size=10.5)
        if i:
            f.arrow(133, 38 + i * 58 - 20, 133, 38 + i * 58)
    f.text(133, 282, "输出：文本 token", cls="mu", size=10)
    f.text(497, 20, "SGLang-Omni：多个 stage 组成的流水线", size=11, weight="600")
    f.rect(318, 36, 358, 32, "purple", text="HTTP API → Client → Coordinator（请求生命周期、多终点合并、abort）", size=9.6)
    stages = [("编码器\nSimpleScheduler", "green"), ("thinker\nOmniScheduler", "blue"), ("talker\nOmniScheduler", "blue"),
              ("声码器\n流式调度器", "green")]
    xs = [300, 396, 492, 588]
    for x, (t, c) in zip(xs, stages):
        f.rect(x, 104, 88, 50, c, text=t, size=9.6)
    for a, b in zip(xs, xs[1:]):
        f.arrow(a + 88, 129, b, 129, dash="5 3" if a >= 396 else None)
    f.arrow(344, 68, 344, 104)
    f.arrow(632, 104, 632, 68)
    f.text(540, 168, "虚线：边生成边推的流式通道", cls="mu", size=9.4)
    f.rect(396, 194, 184, 40, "orange", text="借 SGLang：调度、KV 池、\nModelRunner、CUDA Graph", size=9.6)
    f.arrow(440, 194, 440, 156, dash="4 3", opacity=0.7)
    f.arrow(536, 194, 536, 156, dash="4 3", opacity=0.7)
    f.elbow([(396, 214), (300, 214), (300, 115), (238, 115)], dash="4 3")
    f.elbow([(396, 226), (270, 226), (270, 173), (238, 173)], dash="4 3")
    f.text(332, 244, "组合 · 继承", cls="mu", size=9.4)
    f.text(497, 270, "omni 自己管：拓扑、stage 生命周期、stage 间传输、OpenAI 兼容 API", size=10)
    f.text(497, 290, "输出：文本 + 音频（多个终点）；也有模型完全不用 SGLang", cls="mu", size=9.6)
    return f


@figure("omni", "omni-layers")
def omni_layers():
    f = Fig(720, 300, "sglang_omni/ 的分层：按一条请求经过的顺序排列，右侧是各层的 Python 行数和本书对应的章节")
    rows = [("API", "serve/ · client/ · cli/", "约 1.8 万行", "第三章", "purple"),
            ("编排", "pipeline/ · proto/", "约 0.9 万行", "第三、四章", "blue"),
            ("调度", "scheduling/ · model_runner/ · vendor/", "约 2.2 万行", "第四、五章", "blue"),
            ("通信", "comm/ · relay/", "约 0.7 万行", "第六章", "green"),
            ("配置与部署", "config/ · mps/ · platforms/", "约 0.8 万行", "第七章", "green"),
            ("模型", "models/（25 个目录）", "约 14.8 万行", "第八、九章", "orange")]
    for i, (name, dirs, lines, ch, c) in enumerate(rows):
        y = 18 + i * 44
        f.rect(20, y, 110, 34, c, text=name, size=11, weight="600")
        f.rect(140, y, 330, 34, "bx", text=dirs, size=10.5)
        f.text(540, y + 17, lines, size=10.5)
        f.text(650, y + 17, ch, cls="mu", size=10)
        if i:
            f.arrow(75, y - 10, 75, y)
    f.text(360, 290, "横向工具：utils/ · preprocessing/ · profiler/ · diagnostics/；运行时核心（编排 + 调度 + 通信 + 配置）约 3.9 万行", cls="mu", size=9.6)
    return f


@figure("omni", "omni-control-plane")
def omni_control_plane():
    f = Fig(720, 290, "Coordinator 只看两头：PUSH 提交给入口 stage，PULL 收终点的完成和流，PUB 广播 abort；stage 之间直接传")
    f.rect(250, 20, 220, 46, "purple", text="Coordinator\nrequests · futures · stream 队列", size=10.5)
    names = [("preprocessing", 30), ("encoder", 205), ("thinker", 380), ("decode（终点）", 555)]
    for n, x in names:
        f.rect(x, 170, 135, 44, "blue" if n == "thinker" else ("green" if "终点" in n else "gray"), text=n, size=10.5)
    for (_, a), (_, b) in zip(names, names[1:]):
        f.arrow(a + 135, 192, b, 192)
    f.text(272, 232, "DataReadyMessage（PUSH）+ relay 数据面", cls="mu", size=9.6)
    f.elbow([(250, 50), (97, 50), (97, 170)], cls="purple-l", hcls="purple-s")
    f.text(150, 40, "SubmitMessage（PUSH）", cls="mu", size=9.6)
    f.elbow([(622, 170), (622, 50), (470, 50)], cls="green-l", hcls="green-s")
    f.text(580, 40, "Complete / Stream（PULL）", cls="mu", size=9.6)
    for _, x in names:
        f.arrow(360, 66, x + 67, 168, dash="3 4", opacity=0.45)
    f.text(205, 128, "AbortMessage\n（PUB / SUB 广播）", cls="mu", size=9.6)
    f.text(360, 262, "全部是 ipc:// 的 Unix 域套接字，放在一个临时运行目录里；Coordinator 看不到中间 stage 之间的数据", cls="mu", size=9.6)
    return f


@figure("omni", "omni-stage-threads")
def omni_stage_threads():
    f = Fig(720, 270, "Stage 的两个线程：asyncio 事件循环做 IO，调度器线程做计算，两者之间只有 inbox / outbox 两个队列")
    f.rect(110, 26, 500, 196, "bx", rx=12, dash="5 4")
    f.text(360, 42, "一个 OS 进程（可以住多个 stage，共享事件循环）", cls="mu", size=9.8)
    f.rect(130, 58, 190, 140, "blue", rx=10)
    f.text(225, 76, "asyncio 事件循环：Stage", size=10.8, weight="600")
    f.text(225, 132, "收 ZMQ 控制消息\n读写 relay、回 ACK\n扇入攒齐（AggregatedInput）\n路由结果、转发流式块", size=9.8)
    f.rect(400, 58, 190, 140, "orange", rx=10)
    f.text(495, 76, "调度器线程：scheduler.start()", size=10.8, weight="600")
    f.text(495, 132, "从 inbox 取消息\n算一个函数 / 组一批\n发射一次 GPU 前向\n结果放进 outbox", size=9.8)
    f.arrow(320, 100, 400, 100, label="inbox", ly=-8, lsize=9.6)
    f.arrow(400, 160, 320, 160, label="outbox", ly=14, lsize=9.6)
    f.arrow(30, 128, 130, 128)
    f.text(60, 116, "上游 /\nCoordinator", cls="mu", size=9.2)
    f.arrow(590, 128, 690, 128)
    f.text(655, 116, "下游 /\nCoordinator", cls="mu", size=9.2)
    f.text(360, 242, "inbox / outbox 都是线程安全的 queue.Queue；outbox 用 run_in_executor 阻塞读，不卡事件循环", cls="mu", size=9.6)
    f.text(360, 260, "Stage 不按调度器类型分支：SimpleScheduler、OmniScheduler、流式调度器呈现同一个接口", cls="mu", size=9.6)
    return f


@figure("omni", "omni-ar-stage")
def omni_ar_stage():
    f = Fig(720, 346, "一个自回归 stage 的内部分层：左边是 omni 的类，右边是它们从 SGLang 借来的东西，以及借的方式")
    rows = [("Stage（IO 壳）", "inbox / outbox", None),
            ("OmniScheduler", "收请求 → 建 Req → 选批 → 执行 → 出结果", "SGLang Scheduler 的方法\n组合：__getattr__ 借用"),
            ("omni 的 ModelRunner", "钩子：多模态注入、反馈式多码本、采样", None),
            ("ModelWorker", "替代 TpModelWorker：分布式、模型配置", None),
            ("SGLModelRunner", "加载权重、KV 池、CUDA Graph、前向", "SGLang ModelRunner\n继承"),
            ("模型（talker、thinker……）", "网络结构", "SGLang 并行层 + ModelRegistry\n调用 · 登记")]
    for i, (name, desc, sgl) in enumerate(rows):
        y = 14 + i * 50
        f.rect(20, y, 190, 38, "blue", text=name, size=10.6, weight="600")
        f.text(330, y + 19, desc, size=9.8)
        if sgl:
            f.rect(500, y - 2, 204, 42, "orange", text=sgl, size=9.6)
            f.arrow(452, y + 19, 500, y + 19, dash="4 3")
        if i:
            f.arrow(115, y - 12, 115, y)
    f.text(360, 326, "SGLangGenerationEngineBuilder.build() 是一个模板方法：按固定顺序把这些层搭起来，模型子类只覆盖自己不同的几步", cls="mu", size=9.6)
    return f


@figure("omni", "omni-transport-ladder")
def omni_transport_ladder():
    f = Fig(720, 310, "数据面按每条边选择传输方式：从上往下判断，第一个满足的条件决定这条边怎么传")
    f.rect(16, 30, 196, 70, "purple", text="控制面：ZMQ\nSubmit / DataReady / Ack\nComplete / Stream / Abort", size=9.8)
    f.rect(16, 120, 196, 70, "green", text="数据面：relay\n张量拼成一个 uint8 缓冲区\n发送方拥有，接收方 ACK 后释放", size=9.8)
    f.text(114, 214, "小于 16 KB 的 CPU 流式块：\n直接内联进控制消息", cls="mu", size=9.6)
    qs = [("目标在另一台机器？", "Mooncake（RDMA）"),
          ("目标在同一个进程？", "直接传 Python 对象（不进通信层）"),
          ("张量在 GPU，且双方都是 GPU stage？", "同一张卡：PyTorch CUDA IPC 句柄\n不同的卡：CUDA IPC 显存池"),
          ("其他（CPU 张量……）", "SHM 共享内存")]
    for i, (q, a) in enumerate(qs):
        y = 22 + i * 66
        f.rect(250, y, 220, 44, "bx", text=q, size=10)
        f.arrow(470, y + 22, 510, y + 22, label="是", ly=-7, lsize=9)
        f.rect(510, y, 196, 44, "blue" if i != 1 else "gray", text=a, size=9.6)
        if i < len(qs) - 1:
            f.arrow(360, y + 44, 360, y + 66, label="否", lx=10, ly=0, lsize=9)
    f.text(478, 296, "CPU 平台上，大的 CPU 流式块发给非 GPU 的 stage 会报错（第六章的实验）", cls="mu", size=9.4)
    return f


@figure("omni", "omni-qwen3-tts")
def omni_qwen3_tts():
    f = Fig(720, 230, "Qwen3-TTS：三个 stage 住在同一个进程里，TTS 引擎每一步生成一帧 16 个码并流式推给声码器")
    f.rect(14, 26, 692, 150, "bx", rx=12, dash="5 4")
    f.text(360, 42, "process = pipeline（同一个进程；声码器和 TTS 引擎在 GPU0）", cls="mu", size=9.8)
    f.rect(30, 62, 180, 96, "gray", text="preprocessing\nThreadedSimpleScheduler\n8 个工作线程\n文本、音色、参考音频编码", size=9.6)
    f.rect(260, 62, 200, 96, "blue", text="tts_engine\nOmniScheduler（嵌 SGLang）\ntalker 主干出第 1 个码本\ncode predictor 补齐 16 个", size=9.6)
    f.rect(510, 62, 180, 96, "green", text="vocoder（终点）\n流式声码器\nchunk：1、2、4、8 帧\nCUDA Graph、高优先级流", size=9.6)
    f.arrow(210, 110, 260, 110, label="payload", ly=-8, lsize=9)
    f.arrow(460, 98, 510, 98, label="码流", ly=-8, lsize=9, dash="5 3")
    f.arrow(460, 128, 510, 128, label="payload", ly=14, lsize=9)
    f.text(360, 196, "首包延迟 ≈ 预处理 + prefill + 第一段需要的 talker 步数 × 每步时间 + 第一段解码 + 传输", size=10)
    f.text(360, 216, "每帧 80 ms 音频（12.5 Hz）；第一次发送的码前面拼上参考音频的码，作为声码器的左上下文", cls="mu", size=9.6)
    return f


@figure("omni", "omni-qwen3-omni")
def omni_qwen3_omni():
    f = Fig(720, 330, "Qwen3-Omni 语音流水线：七个 stage、两个终点；实线传完整 payload，虚线是流式通道")
    f.rect(16, 130, 120, 50, "gray", text="preprocessing\nCPU", size=10)
    f.rect(180, 50, 130, 44, "green", text="image_encoder\nGPU0", size=10)
    f.rect(180, 120, 130, 44, "green", text="audio_encoder\nGPU0", size=10)
    f.rect(370, 60, 150, 70, "blue", text="thinker（GPU0）\nOmniScheduler · MoE\nasync decode", size=9.8)
    f.rect(370, 210, 150, 70, "blue", text="talker_ar（GPU1）\n反馈式多码本\npartial start", size=9.8)
    f.rect(580, 60, 126, 54, "green", text="decode\n文本（终点）", size=10)
    f.rect(580, 210, 126, 54, "green", text="code2wav（GPU0）\n音频（终点）", size=9.8)
    f.arrow(136, 145, 180, 75)
    f.arrow(136, 152, 180, 142)
    f.arrow(310, 72, 370, 85)
    f.arrow(310, 142, 370, 110)
    f.elbow([(76, 130), (76, 30), (445, 30), (445, 60)])
    f.elbow([(76, 180), (76, 300), (445, 300), (445, 280)])
    f.arrow(310, 152, 370, 225)
    f.arrow(520, 87, 580, 87, dash="5 3")
    f.arrow(445, 130, 445, 210, dash="5 3", label="token + 隐状态", lx=-48, ly=0, lsize=9)
    f.arrow(520, 237, 580, 237, dash="5 3")
    f.text(250, 22, "预处理的结果直达 thinker 和 talker（参与扇入）", cls="mu", size=9.2)
    f.text(625, 160, "要不要语音由同一个判断决定：\n路由、扇入、流结束、终点一起变", cls="mu", size=9.4)
    f.text(250, 186, "编码结果同时送 thinker 和 talker", cls="mu", size=9.2)
    f.text(360, 320, "单卡变体（speech-colocated）：五个 GPU stage 都在 GPU0，code2wav 住进 talker 的进程", cls="mu", size=9.4)
    return f


@figure("omni", "omni-router")
def omni_router():
    f = Fig(720, 270, "Rust router 里一条请求的路径：同质池直接流式转发，异构池先在有界预算里分类，再准入、选择、转发")
    f.rect(14, 100, 76, 46, "gray", text="客户端", size=10.5)
    f.rect(120, 100, 96, 46, "bx", text="有界监听器\nHTTP / WS 路由", size=9.4)
    f.rect(250, 40, 150, 46, "blue", text="直接路径\n同质池：不读请求体", size=9.4)
    f.rect(250, 160, 150, 46, "orange", text="分类路径\n异构池：读一次、分类", size=9.4)
    f.rect(436, 100, 116, 46, "purple", text="准入（信号量）\n选择 worker", size=9.4)
    f.rect(586, 100, 120, 46, "green", text="转发（背压）", size=10)
    f.arrow(90, 123, 120, 123)
    f.arrow(216, 116, 250, 70)
    f.arrow(216, 130, 250, 180)
    f.arrow(400, 66, 436, 112)
    f.arrow(400, 182, 436, 134)
    f.arrow(552, 123, 586, 123)
    for i, w in enumerate(("w1", "w2", "w3")):
        f.rect(600 + i * 36, 186, 30, 26, "bx", text=w, size=9)
        f.arrow(646, 146, 615 + i * 36, 186, opacity=0.6)
    f.rect(436, 200, 116, 34, "gray", text="健康检查（串行探测）", size=9)
    f.arrow(494, 200, 494, 146, dash="4 3")
    f.text(360, 254, "连接失败：502 upstream_protocol_error + 立即探测；当前不重试（RFC #1623 的待办项）", cls="mu", size=9.6)
    return f



@figure("omni", "omni-roadmap")
def omni_roadmap():
    f = Fig(720, 430, "SGLang-Omni 源码导读的学习路线图：四段阅读主线、每段要跑的实验和要达到的检查点，以及从第 2 天开始并行的贡献轨道")
    f.text(70, 20, "先修", cls="mu", size=10)
    f.rect(16, 30, 108, 58, "gray", text="SGLang 设计演进\n推理系统·源码导读\n（SGLang 部分）", size=9.2)
    stages = [("① 全景", "第 1 天上午", "定位 · 仓库地图", "purple"),
              ("② 一条请求的旅程", "第 1～4 天", "Coordinator · Stage\n嵌入 SGLang · 通信 · 配置", "blue"),
              ("③ 案例精读", "第 5～6 天", "Qwen3-TTS · Qwen3-Omni\nRust router", "orange"),
              ("④ 贡献实战", "第 7 天起", "测试与 CI\n从读代码到提 MR", "green")]
    xs = [150, 292, 448, 590]
    ws = [128, 142, 128, 116]
    f.text(420, 20, "阅读主线", cls="mu", size=10)
    for (title, when, chs, c), x, w in zip(stages, xs, ws):
        f.rect(x, 30, w, 58, c, rx=9)
        f.text(x + w / 2, 43, title, size=10.5, weight="600")
        f.text(x + w / 2, 70, chs, size=9.3)
        f.text(x + w / 2, 100, when, cls="mu", size=9.2)
    f.arrow(124, 59, 150, 59)
    for x, w, nx in zip(xs, ws, xs[1:]):
        f.arrow(x + w, 59, nx, 59)
    f.text(70, 173, "动手实验\n（都在 CPU 上跑）", cls="mu", size=10)
    labs = ["pins / imports\nno-sglang-models\npace",
            "ch3 terminals · abort\nch4 fanin · batch · stream\nch5 compose · methods\nch6 trace · ch7 topology",
            "ch8 chunks\nch9 routing\nch10 fleet · tests",
            "ch11 unit · lint\nscan_issues"]
    for lab, x, w in zip(labs, xs, ws):
        f.rect(x, 138, w, 70, "bx", rx=8, text=lab, size=9.2)
        f.arrow(x + w / 2, 88 + 18, x + w / 2, 138, dash="3 3", opacity=0.6)
    f.text(70, 266, "检查点\n（做不到就回头重读）", cls="mu", size=9.6)
    checks = ["M1 讲清 omni 与\nSGLang 的边界",
              "M2 在 CPU 上搭出带\n扇入和流式的流水线",
              "M3 画出任意模型的\nstage 图与首包公式",
              "M4 本地跑通单测、\nrouter 测试与 pre-commit"]
    for chk, x, w in zip(checks, xs, ws):
        cx = x + w / 2
        f.poly([(cx, 244), (cx + 9, 253), (cx, 262), (cx - 9, 253)], cls="orange-s")
        f.text(cx, 284, chk, size=9.4)
        f.arrow(cx, 208, cx, 244, dash="3 3", opacity=0.6)
    track = [("每天 15 分钟\n看新 issue 与 PR", 150), ("第 4 天\n选一个切入点", 292), ("第 6 天\n复现、读相关代码", 448),
             ("第 7 天\n认领评论 / issue", 590)]
    for i, (t, x) in enumerate(track):
        w = ws[i]
        f.rect(x, 332, w, 44, "green" if i == 3 else "gray", rx=8, text=t, size=9.3)
        if i:
            f.arrow(track[i - 1][1] + ws[i - 1], 354, x, 354)
    f.rect(590, 392, 116, 30, "green", rx=8, text="目标：一个合入的 PR", size=9.4, weight="600")
    f.arrow(648, 376, 648, 392)
    f.text(70, 354, "贡献轨道\n（第 2 天起并行）", cls="mu", size=9.6)
    f.text(300, 407, "之后每周 3～5 小时推进 PR：跟进 review、补测试、写验证数字", cls="mu", size=9.4)
    return f


if __name__ == "__main__":
    main(sys.argv[1:])
