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


if __name__ == "__main__":
    main(sys.argv[1:])
