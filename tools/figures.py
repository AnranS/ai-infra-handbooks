"""生成各手册里的示意图（SVG）。每张图是一个函数，画在一个小画布上，输出到对应手册的 docs/assets/figures/。

图里的文字、线条用 currentColor，形状的填充用 CSS 类（blue、green、orange、purple、red、gray、bx），
由 hooks/figures.py 内嵌进页面后，跟随站点的亮色 / 暗色主题。页面里这样引用：

    ![图：说明文字](../assets/figures/名字.svg){.aig-svg}

用法：python3 tools/figures.py          # 重新生成全部
      python3 tools/figures.py kv-cache # 只生成名字里含 kv-cache 的
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
FIGS: dict[str, tuple[str, callable]] = {}


def figure(book: str, name: str):
    def deco(fn):
        FIGS[name] = (book, fn)
        return fn
    return deco


class Fig:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title, self.el = w, h, title, []

    # ------------------------------------------------------------------ 基本图元
    def text(self, x, y, s, cls="tx", size=13, anchor="middle", weight=None, family=None):
        lines = str(s).split("\n")
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


@figure("llm", "roofline")
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
@figure("llm", "dot-product")
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


@figure("llm", "matmul-views")
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


@figure("llm", "singular-values")
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


@figure("llm", "entropy-kl")
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


@figure("llm", "backprop-graph")
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


@figure("llm", "queue-latency")
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
    for name, (book, fn) in FIGS.items():
        if argv and not any(a in name for a in argv):
            continue
        out = ROOT / book / "docs" / "assets" / "figures" / f"{name}.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fn().svg(), encoding="utf-8")
        print(f"{book}/{name}.svg")



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


@figure("llm", "float-numberline")
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


if __name__ == "__main__":
    main(sys.argv[1:])
