"""给各手册画"小剧场"漫画（SVG）：几格分镜、两个固定角色、对话气泡和一点示意图。

和 tools/figures.py 用同一套画布（Fig）和 CSS 类，所以漫画也跟随亮 / 暗主题、不依赖任何图像模型。
输出到 <book>/docs/assets/comics/<name>.svg，页面里这样引用：

    ![漫画：说明文字](../assets/comics/名字.svg){.aig-svg}

角色：
    小推 —— 学推理的人，火柴人，戴眼镜
    阿卡 —— 显卡君，一张带风扇的显卡，会出汗也会耍酷

用法：python3 tools/comics.py            # 重新生成全部
      python3 tools/comics.py kv-cache   # 只生成名字里含 kv-cache 的
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import Fig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COMICS: dict[str, tuple[str, callable]] = {}


def comic(book: str, name: str):
    def deco(fn):
        COMICS[name] = (book, fn)
        return fn
    return deco


def wrap(text: str, width: int) -> list[str]:
    """按显示宽度折行：中文算 1，ASCII 算 0.55；遇到 '\\n' 强制换行"""
    lines = []
    for para in text.split("\n"):
        cur, w = "", 0.0
        for ch in para:
            cw = 0.55 if ord(ch) < 128 else 1.0
            if w + cw > width and cur:
                lines.append(cur)
                cur, w = "", 0.0
            cur += ch
            w += cw
        lines.append(cur)
    return lines


class Panel:
    """一格：左上角 (x0, y0)，大小 w × h。所有坐标都是格内的相对坐标。"""

    def __init__(self, f: Fig, x0: float, y0: float, w: float, h: float, idx: int):
        self.f, self.x0, self.y0, self.w, self.h, self.idx = f, x0, y0, w, h, idx
        f.rect(x0, y0, w, h, "bx", rx=8, sw=1.2)

    # ------------------------------------------------------------------ 基本件
    def X(self, x):
        return self.x0 + x

    def Y(self, y):
        return self.y0 + y

    def caption(self, s: str):
        """左上角的旁白条"""
        f = self.f
        tw = sum(0.55 if ord(c) < 128 else 1.0 for c in s) * 10 + 14
        f.rect(self.X(0), self.Y(0), tw, 18, "gray", rx=6, sw=0)
        f.text(self.X(7), self.Y(9), s, cls="mu", size=9.5, anchor="start")

    def bubble(self, x, y, text, width=13, size=10.5, tail=None, cls="bx"):
        """对话气泡：左上角 (x, y)，按 width 个汉字折行；tail 是说话人嘴的位置（格内坐标）"""
        f = self.f
        lines = wrap(text, width)
        lh = size * 1.42
        w = max(sum(0.55 if ord(c) < 128 else 1.0 for c in ln) for ln in lines) * size + 16
        h = len(lines) * lh + 10
        if tail is not None:
            tx, ty = tail
            # 尾巴：从气泡边缘最近的点指向嘴
            bx = min(max(tx, x + 14), x + w - 14)
            by = y + h if ty > y + h / 2 else y
            d = 7
            f.poly([(self.X(bx - d), self.Y(by)), (self.X(bx + d), self.Y(by)), (self.X(tx), self.Y(ty))], cls="bubble-tail")
        f.rect(self.X(x), self.Y(y), w, h, cls, rx=8, sw=1.1)
        for i, ln in enumerate(lines):
            f.text(self.X(x + 8), self.Y(y + 5 + lh * (i + 0.5)), ln, cls="tx", size=size, anchor="start")
        return w, h

    def note(self, x, y, s, size=9.5, anchor="middle"):
        self.f.text(self.X(x), self.Y(y), s, cls="mu", size=size, anchor=anchor)

    # ------------------------------------------------------------------ 角色：小推（火柴人）
    def person(self, x, y, mood="ask", facing=1):
        """脚底中心在 (x, y)。mood: ask / think / idea / happy / worry"""
        f = self.f
        hx, hy = x, y - 46
        f.circle(self.X(hx), self.Y(hy), 11, "bx", sw=1.3)
        # 头发（刘海）
        f.path(f"M {self.X(hx - 11):.1f} {self.Y(hy - 2):.1f} Q {self.X(hx - 4):.1f} {self.Y(hy - 16):.1f} {self.X(hx + 6):.1f} {self.Y(hy - 11):.1f}", cls="ln", sw=1.6)
        # 眼镜
        for dx in (-4.5, 3.5):
            f.circle(self.X(hx + dx * facing), self.Y(hy - 1), 3.2, "ln", sw=1)
        f.line(self.X(hx - 1.3), self.Y(hy - 1), self.X(hx + 0.3), self.Y(hy - 1), cls="ln", sw=1)
        f.circle(self.X(hx - 4.5 * facing), self.Y(hy - 1), 1.1, "ah", sw=0)
        f.circle(self.X(hx + 3.5 * facing), self.Y(hy - 1), 1.1, "ah", sw=0)
        mouth_y = hy + 5
        if mood in ("ask", "worry"):
            f.circle(self.X(hx + 1 * facing), self.Y(mouth_y), 2.2, "ln", sw=1.1)
        elif mood == "think":
            f.line(self.X(hx - 3), self.Y(mouth_y), self.X(hx + 3), self.Y(mouth_y), cls="ln", sw=1.1)
        else:
            f.path(f"M {self.X(hx - 4):.1f} {self.Y(mouth_y - 1):.1f} Q {self.X(hx):.1f} {self.Y(mouth_y + 4):.1f} {self.X(hx + 4):.1f} {self.Y(mouth_y - 1):.1f}", cls="ln", sw=1.2)
        # 身体
        f.rect(self.X(x - 9), self.Y(y - 34), 18, 22, "blue", rx=5, sw=1.1)
        # 腿
        f.line(self.X(x - 4), self.Y(y - 12), self.X(x - 6), self.Y(y), cls="ln", sw=1.6)
        f.line(self.X(x + 4), self.Y(y - 12), self.X(x + 6), self.Y(y), cls="ln", sw=1.6)
        # 手臂
        if mood == "ask":
            f.line(self.X(x - 9 * facing), self.Y(y - 30), self.X(x - 20 * facing), self.Y(y - 42), cls="ln", sw=1.6)   # 一只手抬起
            f.line(self.X(x + 9 * facing), self.Y(y - 30), self.X(x + 16 * facing), self.Y(y - 18), cls="ln", sw=1.6)
        elif mood == "think":
            f.line(self.X(x + 9 * facing), self.Y(y - 30), self.X(x + 14 * facing), self.Y(y - 40), cls="ln", sw=1.6)   # 手托下巴
            f.line(self.X(x + 14 * facing), self.Y(y - 40), self.X(x + 6 * facing), self.Y(y - 40), cls="ln", sw=1.6)
            f.line(self.X(x - 9 * facing), self.Y(y - 30), self.X(x - 16 * facing), self.Y(y - 18), cls="ln", sw=1.6)
            f.text(self.X(x + 22 * facing), self.Y(y - 58), "…", cls="mu", size=14)
        elif mood == "happy" or mood == "idea":
            f.line(self.X(x - 9), self.Y(y - 30), self.X(x - 20), self.Y(y - 44), cls="ln", sw=1.6)
            f.line(self.X(x + 9), self.Y(y - 30), self.X(x + 20), self.Y(y - 44), cls="ln", sw=1.6)
        else:
            f.line(self.X(x - 9), self.Y(y - 30), self.X(x - 16), self.Y(y - 18), cls="ln", sw=1.6)
            f.line(self.X(x + 9), self.Y(y - 30), self.X(x + 16), self.Y(y - 18), cls="ln", sw=1.6)
        if mood == "idea":   # 头顶的灯泡
            bx, by = hx + 16, hy - 20
            f.circle(self.X(bx), self.Y(by), 6, "orange", sw=1)
            f.rect(self.X(bx - 3), self.Y(by + 5), 6, 4, "gray", rx=1, sw=0.6)
            for a in (-60, -30, 0, 30, 60):
                r = math.radians(a - 90)
                f.line(self.X(bx + 8 * math.cos(r)), self.Y(by + 8 * math.sin(r)), self.X(bx + 12 * math.cos(r)), self.Y(by + 12 * math.sin(r)), cls="ln", sw=1)
        if mood == "worry":
            self.sweat(hx + 13, hy - 8)
        return hx, hy + 7   # 嘴的位置，给气泡尾巴用

    # ------------------------------------------------------------------ 角色：阿卡（显卡君）
    def gpu(self, x, y, mood="smile", facing=1):
        """显卡的中心在 (x, y)。mood: smile / sweat / explain / cool / tired"""
        f = self.f
        w, h = 68, 42
        f.rect(self.X(x - w / 2), self.Y(y - h / 2), w, h, "green", rx=7, sw=1.3)
        # 金手指
        for i in range(6):
            f.rect(self.X(x - 22 + i * 8), self.Y(y + h / 2), 5, 5, "orange", rx=1, sw=0.5)
        # 风扇（右半边）
        fx, fy = x + 17 * facing, y
        f.circle(self.X(fx), self.Y(fy), 13, "bx", sw=1)
        for a in (0, 120, 240):
            r = math.radians(a)
            r2 = math.radians(a + 70)
            f.path(f"M {self.X(fx):.1f} {self.Y(fy):.1f} Q {self.X(fx + 14 * math.cos(r)):.1f} {self.Y(fy + 14 * math.sin(r)):.1f} {self.X(fx + 10 * math.cos(r2)):.1f} {self.Y(fy + 10 * math.sin(r2)):.1f} Z", cls="gray", sw=0.6)
        f.circle(self.X(fx), self.Y(fy), 2.5, "ah", sw=0)
        # 脸（左半边）
        ex = x - 15 * facing
        if mood == "cool":
            f.rect(self.X(ex - 11), self.Y(y - 9), 9, 6, "ah", rx=2, sw=0)
            f.rect(self.X(ex + 2), self.Y(y - 9), 9, 6, "ah", rx=2, sw=0)
            f.line(self.X(ex - 2), self.Y(y - 6), self.X(ex + 2), self.Y(y - 6), cls="ln", sw=1)
        else:
            f.circle(self.X(ex - 6), self.Y(y - 6), 1.8, "ah", sw=0)
            f.circle(self.X(ex + 6), self.Y(y - 6), 1.8, "ah", sw=0)
        my = y + 6
        if mood in ("sweat", "tired"):
            f.path(f"M {self.X(ex - 6):.1f} {self.Y(my + 3):.1f} Q {self.X(ex):.1f} {self.Y(my - 3):.1f} {self.X(ex + 6):.1f} {self.Y(my + 3):.1f}", cls="ln", sw=1.2)
            self.sweat(x - w / 2 - 2, y - h / 2 + 4)
        elif mood == "explain":
            f.circle(self.X(ex), self.Y(my + 1), 3, "ln", sw=1.1)
        else:
            f.path(f"M {self.X(ex - 6):.1f} {self.Y(my - 1):.1f} Q {self.X(ex):.1f} {self.Y(my + 5):.1f} {self.X(ex + 6):.1f} {self.Y(my - 1):.1f}", cls="ln", sw=1.2)
        # 小手
        if mood == "explain":
            f.line(self.X(x - w / 2 * facing), self.Y(y - 4), self.X(x - (w / 2 + 16) * facing), self.Y(y - 18), cls="ln", sw=1.6)
        elif mood == "happy":
            f.line(self.X(x - w / 2), self.Y(y - 4), self.X(x - w / 2 - 12), self.Y(y - 18), cls="ln", sw=1.6)
            f.line(self.X(x + w / 2), self.Y(y - 4), self.X(x + w / 2 + 12), self.Y(y - 18), cls="ln", sw=1.6)
        return ex, my + 6

    def sweat(self, x, y):
        f = self.f
        f.path(f"M {self.X(x):.1f} {self.Y(y - 5):.1f} Q {self.X(x + 4):.1f} {self.Y(y + 1):.1f} {self.X(x):.1f} {self.Y(y + 4):.1f} Q {self.X(x - 4):.1f} {self.Y(y + 1):.1f} {self.X(x):.1f} {self.Y(y - 5):.1f} Z", cls="blue", sw=0.8)

    # ------------------------------------------------------------------ 道具
    def token_stack(self, x, y, n=5, label=None, cls="gray"):
        """一摞 token 方块，底部中心在 (x, y)"""
        f = self.f
        for i in range(n):
            f.rect(self.X(x - 14), self.Y(y - 11 - i * 11), 28, 10, cls, rx=2, sw=0.7)
        if label:
            f.text(self.X(x), self.Y(y - 11 * n - 10), label, cls="mu", size=9)

    def kv_blocks(self, x, y, n=8, new=None, label=None):
        """一排 KV 块，左上角 (x, y)；new 是新追加的那一块（橙色）"""
        f = self.f
        for i in range(n):
            cls = "orange" if i == new else "blue"
            f.rect(self.X(x + i * 15), self.Y(y), 13, 13, cls, rx=2, sw=0.7)
        if label:
            f.text(self.X(x + n * 15 / 2), self.Y(y + 24), label, cls="mu", size=9)


class Strip:
    def __init__(self, title: str, cols: int, rows: int, pw: int = 236, ph: int = 212, gap: int = 10):
        self.cols, self.rows, self.pw, self.ph, self.gap = cols, rows, pw, ph, gap
        self.f = Fig(cols * pw + (cols + 1) * gap, rows * ph + (rows + 1) * gap, title)

    def panel(self, i: int) -> Panel:
        r, c = divmod(i, self.cols)
        return Panel(self.f, self.gap + c * (self.pw + self.gap), self.gap + r * (self.ph + self.gap), self.pw, self.ph, i)


# ====================================================================== 大模型原理
@comic("llm", "kv-cache")
def kv_cache_strip():
    s = Strip("小剧场：为什么要有 KV Cache", 3, 2)

    p = s.panel(0)
    p.caption("① 生成第 100 个字的时候")
    mouth = p.person(52, 196, "ask")
    p.bubble(14, 26, "模型生成第 100 个字，为什么又把前面 99 个字全都算一遍？", width=13, tail=(mouth[0] + 4, mouth[1]))
    g = p.gpu(176, 172, "sweat", facing=-1)
    p.token_stack(120, 206, n=6, label="前 99 个字", cls="gray")
    p.bubble(140, 96, "每一步都从头跑一遍前向……", width=9, size=9.5, tail=(g[0], g[1] - 2))

    p = s.panel(1)
    p.caption("② 注意力在算什么")
    g = p.gpu(60, 150, "explain")
    p.bubble(14, 26, "当前这个字的 Q，要和前面每一个字的 K 做点积，再按权重把 V 加起来。", width=14, tail=(g[0], g[1] - 2))
    # 示意：Q₁₀₀ 指向一排 K
    p.f.rect(p.X(118), p.Y(118), 26, 16, "orange", rx=3, sw=0.8, text="Q₁₀₀", size=8.5)
    for i, lab in enumerate(("K₁", "K₂", "…", "K₉₉", "K₁₀₀")):
        x = 118 + i * 24
        p.f.rect(p.X(x), p.Y(176), 22, 16, "blue", rx=3, sw=0.8, text=lab, size=8)
        p.f.arrow(p.X(131), p.Y(134), p.X(x + 11), p.Y(176), sw=0.8, opacity=0.6)
    p.note(178, 206, "前面的每一个都要点一次", size=8.5)

    p = s.panel(2)
    p.caption("③ 等等")
    mouth = p.person(60, 196, "idea")
    p.bubble(14, 26, "可是前 99 个字的 K、V，上一步不是已经算过了吗？因果注意力里它们根本不会变！", width=14, tail=(mouth[0] + 4, mouth[1]))
    p.kv_blocks(104, 150, n=8, label="上一步算出来的 K、V")
    p.note(160, 192, "新加一个字，它们一个都不动", size=8.5)

    p = s.panel(3)
    p.caption("④ KV Cache")
    g = p.gpu(66, 152, "happy")
    p.bubble(14, 26, "那就存起来！每一步只算新字的 K、V，追加到缓存里，前面的直接拿来用。", width=14, tail=(g[0], g[1] - 2))
    p.f.rect(p.X(112), p.Y(118), 118, 64, "gray", rx=6, sw=1, dash="4 3")
    p.f.text(p.X(171), p.Y(128), "KV Cache", cls="mu", size=9.5)
    p.kv_blocks(118, 140, n=7, new=6)
    p.note(171, 170, "n² / 2 → n：平方变线性", size=8.5)
    p.note(171, 196, "历史的 Q 用不到，所以不存", size=8.5)

    p = s.panel(4)
    p.caption("⑤ 于是推理分成了两段")
    g = p.gpu(60, 170, "cool")
    p.bubble(14, 26, "提示词一次算完叫 prefill，之后一个一个吐字叫 decode——两段的脾气完全不同。", width=14, tail=(g[0], g[1] - 2))
    p.f.rect(p.X(110), p.Y(116), 118, 18, "orange", rx=3, sw=0.8, text="prefill：一次算几百个字", size=8)
    p.note(169, 146, "算力瓶颈", size=8.5)
    for i in range(5):
        p.f.rect(p.X(110 + i * 24), p.Y(160), 20, 16, "blue", rx=3, sw=0.8, text="1", size=8)
        if i < 4:
            p.f.arrow(p.X(130 + i * 24), p.Y(168), p.X(134 + i * 24), p.Y(168), sw=0.8)
    p.note(172, 190, "decode：每步 1 个字，读权重和缓存", size=8)
    p.note(169, 203, "访存瓶颈", size=8.5)

    p = s.panel(5)
    p.caption("⑥ 代价")
    mouth = p.person(44, 196, "think")
    p.bubble(14, 26, "那缓存占多少显存？", width=13, tail=(mouth[0] + 4, mouth[1]))
    g = p.gpu(176, 156, "sweat", facing=-1)
    p.bubble(78, 62, "Qwen3-0.6B 每个字 112 KB，LLaMA-3-8B 128 KB；32 个 4K 上下文的请求就是 16 GB，和权重一样大……", width=14, tail=(g[0], g[1] - 2))
    p.note(118, 204, "怎么管好这块显存，就是本章后半段和整本推理系统手册的事", size=8)
    return s.f


def main(argv):
    pat = argv[0] if argv else ""
    for name, (book, fn) in COMICS.items():
        if pat and pat not in name:
            continue
        out = ROOT / book / "docs" / "assets" / "comics" / f"{name}.svg"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fn().svg(), encoding="utf-8")
        print(f"{book}/{name}.svg")


if __name__ == "__main__":
    main(sys.argv[1:])
