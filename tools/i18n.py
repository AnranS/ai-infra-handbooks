"""英文版：构建与翻译辅助。

英文译文放在 <book>/docs-en/，目录结构和 docs/ 一样，只放译好的页面和英文版的资源（比如英文示意图）；
书名、导航分组名等放在 <book>/i18n-en.yml。构建英文版时不改动中文源文件：

  python3 tools/i18n.py build <book> [--site _site]
      1. merge：把 docs/ 复制到 <book>/.i18n-en/docs/，再用 docs-en/ 覆盖；还没译的页面在 front matter 里
         标 untranslated: true（模板据此在页首提示"尚未翻译"），英文站因此总是完整的
      2. config：由 <book>/mkdocs.yml 生成 <book>/mkdocs.en.yml：英文站名、/en/<book>/ 地址、theme.language: en、
         extra.lang: en、导航标题换成英文（页面取英文页的一级标题，分组取 i18n-en.yml）
      3. mkdocs build --strict → <site>/en/<book>/

翻译一页的流程（代码块原样保留，只翻文字和代码注释）：

  python3 tools/i18n.py extract <book> <page.md>
      把页面里的代码块换成 ⟦CODE n⟧ 占位，写到 <book>/.i18n-en/work/<page>.src.md 供翻译；
      同时把代码块里的中文注释列到 <page>.comments.json（键是原文，值留空待填英文）
  python3 tools/i18n.py assemble <book> <page.md>
      读 <book>/.i18n-en/work/<page>.en.md（译文，保留占位）和填好的 comments.json，把代码块放回去、
      注释换成英文；给每个二级及以下标题补上与中文版相同的锚点 {#…}（中文版的链接和锚点在英文版里照样能用），
      写到 <book>/docs-en/<page.md>
  python3 tools/i18n.py status [book ...]
      各书的翻译进度
  python3 tools/i18n.py check [book ...]
      已译页面和中文版逐一对照：代码块（除了译成英文的注释）逐字相同，标题的级别与锚点一致；build.sh 每次都跑
  python3 tools/i18n.py sync <book> <page.md>
      中文页只改了代码时，把最新的代码块搬进英文页，已译的注释沿用

示意图（用 ```text 画的结构图、流程图，不是程序输出）可以整块翻译：在 <page>.diagrams.json 里写
{"代码块编号": "译好的整个代码块（含围栏）"}，assemble 会在它前面加一行 <!-- i18n:diagram <哈希> -->，
哈希对应中文原块；check 只核对中文原块有没有改过。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ["python", "cpp", "cs", "math", "llm", "cuda", "train", "serving", "minisgl", "media", "sglang"]
SITE_URL = "https://anrans.github.io/ai-infra-handbooks/en/"
FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")
HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
CJK = re.compile(r"[㐀-鿿＀-￯　-〿]")
PLACEHOLDER = "⟦CODE {}⟧"


# ---------------------------------------------------------------------------- 页面与导航
def nav_block(text: str) -> tuple[int, int]:
    """mkdocs.yml 里 nav: 这一段的起止位置（nav 是每本书配置的最后一段）"""
    start = text.index("\nnav:\n") + 1
    m = re.search(r"\n(?=\S)", text[start + 5:])
    return start, (start + 5 + m.start() + 1) if m else len(text)


def load_nav(book: str) -> list:
    import yaml
    text = (ROOT / book / "mkdocs.yml").read_text(encoding="utf-8")
    a, b = nav_block(text)
    return yaml.safe_load(text[a:b])["nav"]


def nav_pages(nav) -> list[str]:
    out = []
    for item in nav:
        if isinstance(item, str):
            out.append(item)
        else:
            (_, v), = item.items()
            out += nav_pages(v) if isinstance(v, list) else [v]
    return out


def h1(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING.match(line)
        if m and len(m.group(1)) == 1:
            return re.sub(r"\s*\{#[^}]*\}$", "", m.group(2)).strip()
    return None


def book_meta(book: str) -> dict:
    import yaml
    f = ROOT / book / "i18n-en.yml"
    return yaml.safe_load(f.read_text(encoding="utf-8")) if f.exists() else {}


def en_nav(book: str, nav, meta: dict):
    """导航换成英文：页面标题取英文页的一级标题（首页等取 i18n-en.yml 里的译名），分组名取 i18n-en.yml"""
    names = meta.get("nav") or {}
    out = []
    for item in nav:
        if isinstance(item, str):
            out.append(item)
            continue
        (title, v), = item.items()
        if isinstance(v, list):
            out.append({names.get(title, title): en_nav(book, v, meta)})
        else:
            t = names.get(title) if Path(v).name == "index.md" else None
            out.append({t or h1(ROOT / book / "docs-en" / v) or names.get(title, title): v})
    return out


# ---------------------------------------------------------------------------- 构建
def merge(book: str) -> list[str]:
    """docs/ + docs-en/ → .i18n-en/docs/；返回还没翻译的页面"""
    src, en, out = ROOT / book / "docs", ROOT / book / "docs-en", ROOT / book / ".i18n-en" / "docs"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    if en.exists():
        shutil.copytree(en, out, dirs_exist_ok=True)
    untranslated = []
    for md in sorted(src.rglob("*.md")):
        rel = md.relative_to(src)
        if (en / rel).exists():
            continue
        untranslated.append(rel.as_posix())
        target = out / rel
        text = target.read_text(encoding="utf-8")
        if text.startswith("---\n"):
            text = "---\nuntranslated: true\n" + text[4:]
        else:
            text = "---\nuntranslated: true\n---\n\n" + text
        target.write_text(text, encoding="utf-8")
    return untranslated



def esc_anchor(sid: str) -> str:
    """锚点里的 __ 要转义成 \\_\\_：Markdown 先做行内强调，`{#__repr__-与-__str__}` 里的 __repr__ 会变成
    <strong>，attr_list 就匹配不到、锚点直接当文字显示了。转义之后渲染出的 id 仍然是原来的样子。
    代价是 mkdocs 自己的锚点校验会看到还没还原的占位符而误报，所以英文版关掉它，由 tools/check_links.py
    在构建完成后按真实的 HTML id 统一检查。"""
    return sid.replace("_", chr(92) + "_") if "__" in sid else sid

def write_config(book: str) -> Path:
    import yaml
    meta = book_meta(book)
    text = (ROOT / book / "mkdocs.yml").read_text(encoding="utf-8")
    a, b = nav_block(text)
    nav = en_nav(book, yaml.safe_load(text[a:b])["nav"], meta)
    head = text[:a]

    def put(key: str, value: str | None):
        nonlocal head
        if value is None:
            return
        line = f"{key}: {json.dumps(value, ensure_ascii=False)}"
        head, n = re.subn(rf"^{key}:.*$", lambda _: line, head, count=1, flags=re.M)
        assert n == 1, (book, key)

    put("site_name", meta.get("site_name"))
    put("site_description", meta.get("site_description"))
    put("copyright", meta.get("copyright"))
    put("site_url", f"{SITE_URL}{book}/")
    put("edit_uri", f"edit/main/{book}/docs-en/")
    head = re.sub(rf"^site_url:.*$", lambda m: m.group(0) + "\ndocs_dir: .i18n-en/docs", head, count=1, flags=re.M)
    head = re.sub(r"^  anchors: warn$", "  anchors: info", head, count=1, flags=re.M)   # 见 esc_anchor
    head, n = re.subn(r"^  language: zh$", "  language: en", head, count=1, flags=re.M)
    assert n == 1, (book, "theme.language")
    head, n = re.subn(rf"^  book: {book}$", f"  book: {book}\n  lang: en", head, count=1, flags=re.M)
    assert n == 1, (book, "extra.book")
    body = yaml.safe_dump({"nav": nav}, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000)
    cfg = ROOT / book / "mkdocs.en.yml"
    cfg.write_text("# 由 tools/i18n.py 生成，不要手改（英文书名与导航分组在 i18n-en.yml）\n" + head + body, encoding="utf-8")
    return cfg


def build(book: str, site: Path, mkdocs: str) -> None:
    untranslated = merge(book)
    cfg = write_config(book)
    total = len(list((ROOT / book / "docs").rglob("*.md")))
    print(f"==> building en/{book}（已译 {total - len(untranslated)}/{total} 页）", flush=True)
    subprocess.run([mkdocs, "build", "--strict", "--config-file", str(cfg), "--site-dir", str(site / "en" / book)], check=True)


# ---------------------------------------------------------------------------- 英文门户
EN_BOOKS = [("python", "Python", "Advanced Python", "Data model, typing, concurrency and performance"),
            ("cpp", "C++", "Advanced C++", "RAII, move semantics, memory order, memory pools"),
            ("cs", "CS", "CS Fundamentals", "OS, architecture, networking and algorithms"),
            ("math", "Math", "Math Fundamentals", "Linear algebra, probability, information theory, performance math"),
            ("llm", "LLM", "LLM Internals", "Transformer, KV cache, quantization and estimation"),
            ("cuda", "CUDA", "Advanced CUDA", "GEMM, FlashAttention, Tensor Cores"),
            ("train", "Training", "Distributed Training", "DDP, ZeRO, 3D parallelism and RL training"),
            ("serving", "Serving", "Inference Systems", "Scheduling, parallelism, PD disaggregation, frontier topics"),
            ("minisgl", "mini-sglang", "mini-sglang from Scratch", "Build a complete inference engine from scratch"),
            ("media", "Image & Video", "Image & Video Generation", "Inference, acceleration and serving of diffusion models"),
            ("sglang", "SGLang History", "SGLang Design Evolution", "Reading the source through its commit history")]
EN_PORTAL = ("", "roadmap/", "plan/", "cards/")   # 英文站已有的门户页（和 hooks/crosslinks.py 一致）；其余的链到中文版


def en_header(depth: int, active: str, zh_href: str, gh: str, logo: str) -> str:
    """英文门户页的顶栏：depth 是页面在 en/ 下的层数（en/ 本身是 0）"""
    up = "../" * depth                                   # 到 en/
    zh = up + "../"                                      # 到站点根
    def portal(p):
        return (up if p in EN_PORTAL else zh) + p
    tabs = "".join(f'        <a href="{up}{b}/" title="{name}">{short}</a>\n' for b, short, name, _ in EN_BOOKS)
    menu = "".join(f'          <a class="menu-item" href="{up}{b}/"><b>{name}</b><span>{desc}</span></a>\n' for b, _, name, desc in EN_BOOKS)
    on = lambda p: ' class="on"' if p == active else ""
    on_m = lambda p: " on" if p == active else ""
    return f"""<header>
  <nav class="wrap">
    <a class="brand" href="{up or './'}">{logo}<b>AI Infra</b><span class="brand-sub">Handbooks</span></a>
    <div class="tabs">
      <div class="tabs-books">
{tabs}      </div>
      <details class="menu menu--books">
        <summary>Books<svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></summary>
        <div class="menu-panel">
{menu}        </div>
      </details>
      <span class="sep"></span>
      <a{on('roadmap/')} href="{portal('roadmap/')}">Roadmap</a>
      <a href="{portal('practice/')}">Practice</a>
      <details class="menu">
        <summary>More<svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></summary>
        <div class="menu-panel menu-panel--right">
          <a class="menu-item menu-item--m{on_m('roadmap/')}" href="{portal('roadmap/')}"><b>Roadmap</b><span>Chapter-by-chapter route through all eleven books, week by week</span></a>
          <a class="menu-item menu-item--m" href="{portal('practice/')}"><b>Practice</b><span>Coding exercises for every chapter, graded in the browser</span></a>
          <a class="menu-item{on_m('plan/')}" href="{portal('plan/')}"><b>17-week plan</b><span>Chapters, exercises and checklists, week by week</span></a>
          <a class="menu-item" href="{portal('cards/')}"><b>Flashcards</b><span>Spaced repetition of self-tests and interview questions</span></a>
          <a class="menu-item" href="{portal('playground/')}"><b>Playground</b><span>A Python sandbox in the browser</span></a>
          <a class="menu-item" href="{portal('setup/')}"><b>Setup</b><span>One-click environment on a Mac</span></a>
          <a class="menu-item" href="{portal('search/')}"><b>Site search</b><span>Search all eleven books and the exercises</span></a>
        </div>
      </details>
    </div>
    <a class="icon-btn lang-btn" href="{zh_href}" hreflang="zh" lang="zh" title="切换到中文版">中文</a>
    <a class="icon-btn" href="https://github.com/AnranS/ai-infra-handbooks" title="GitHub" aria-label="GitHub">
      {gh}
    </a>
    <button class="icon-btn" id="theme" type="button" title="Toggle light / dark" aria-label="Toggle light / dark">
      <svg class="i moon" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
      <svg class="i sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
    </button>
  </nav>
</header>"""


def en_footer(depth: int) -> str:
    up = "../" * depth
    return f"""<footer>
  <div class="wrap">
    <span>© 2026 AI Infra Handbooks · open source<span id="busuanzi_container_site_uv" style="display:none"> · visitors <span id="busuanzi_value_site_uv"></span></span><span id="busuanzi_container_site_pv" style="display:none"> · views <span id="busuanzi_value_site_pv"></span></span></span>
    <span><a href="{up or './'}">Home</a> · <a href="https://github.com/AnranS/ai-infra-handbooks">GitHub</a></span>
  </div>
</footer>"""


def translated_titles() -> dict[str, str]:
    """已经译好的章节的英文标题（英文页的一级标题），键是 书/路径（不带 .md）"""
    out = {}
    for b in BOOKS:
        en = ROOT / b / "docs-en"
        for md in en.rglob("*.md") if en.exists() else []:
            t = h1(md)
            if t:
                out[f"{b}/{md.relative_to(en).as_posix()[:-3]}"] = t
    return out


def apply_pairs(text: str, pairs: list) -> tuple[str, int]:
    """按原文替换成译文（长的先换）。原文里的数字当通配：章数、题数变了（site_stats 会同步中文页），
    译文里同样的数字跟着换成新值，不会因为一个数字变了整句退回中文"""
    n = 0
    for zh, en in sorted(pairs, key=lambda kv: -len(kv[0])):
        if zh in text:
            text = text.replace(zh, en); n += 1
            continue
        nums = re.findall(r"\d+", zh)
        if not nums:
            continue
        pat = re.compile("".join(f"(\\d+)" if re.fullmatch(r"\d+", part) else re.escape(part) for part in re.split(r"(\d+)", zh)))
        def sub(m, en=en, nums=nums):
            out = en
            for old, new in zip(nums, m.groups()):
                if old != new:
                    out = re.sub(rf"(?<!\d){old}(?!\d)", new, out, count=1)
            return out
        text, k = pat.subn(sub, text)
        n += bool(k)
    return text, n


def localize_paths(text: str, depth: int) -> str:
    """中文门户页搬到 en/ 下多了一层：共享资源和还没有英文版的门户页要多退一级"""
    up = "../" * depth
    text = re.sub(r'(["\'(])\.\./(assets/|favicon\.svg)', lambda m: f"{m.group(1)}{up}../{m.group(2)}", text)
    text = re.sub(r'(["\'])\.\./(practice|cards|search|setup|playground)/', lambda m: f"{m.group(1)}{up}../{m.group(2)}/", text)
    return text


def portal_page(src: Path, pairs_file: str, depth: int, active: str, zh_href: str, titles: dict | None = None) -> tuple[str, list[str]]:
    zh_index = (ROOT / "portal" / "index.html").read_text(encoding="utf-8")
    logo = re.search(r'<svg class="logo".*?</svg>', zh_index, re.S).group(0)
    gh = re.search(r'<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58.*?</svg>', zh_index, re.S).group(0)
    text = src.read_text(encoding="utf-8")
    text = re.sub(r"<header>.*?</header>", lambda _: en_header(depth, active, zh_href, gh, logo), text, count=1, flags=re.S)
    text = re.sub(r"<footer>.*?</footer>", lambda _: en_footer(depth), text, count=1, flags=re.S)
    if titles is not None:
        text = translate_chapter_titles(text, titles)
    pairs = json.loads((ROOT / "i18n" / "en" / pairs_file).read_text(encoding="utf-8")) if pairs_file else []
    text, _ = apply_pairs(text, pairs)
    text = localize_paths(text, depth)
    left = sorted({m.group(0) for m in re.finditer(r"[^\"'<>\n]*[\u4e00-\u9fff][^\"'<>\n]*", text) if "中文" not in m.group(0)})
    return text, left


def translate_chapter_titles(text: str, titles: dict) -> str:
    """路线图 STAGES 里每一章的标题：译好的章节取英文页标题，其余按 i18n/en/titles.json 翻译"""
    names = json.loads((ROOT / "i18n" / "en" / "titles.json").read_text(encoding="utf-8"))
    out, book = [], None
    for line in text.split("\n"):
        m = re.search(r'book: "(\w+)"', line)
        if re.search(r"^\s*\{ id: \"", line):
            book = m.group(1) if m else None
        def rep(mm):
            key, title = mm.group(1), mm.group(2)
            b, path = key.split(":", 1) if ":" in key else (book, key)
            en = titles.get(f"{b}/{path}") or names.get(title, title)
            return f'["{key}", {json.dumps(en, ensure_ascii=False)}, {mm.group(3)}, "{mm.group(4)}"]'
        line = re.sub(r'\["([\w:/\-]+)", "([^"]*)", (\d), "([^"]*)"\]', rep, line)
        line = re.sub(r'\["(\w+_\w+)", "([\w/\-]+)", "([^"]*)"\]', lambda mm: f'["{mm.group(1)}", "{mm.group(2)}", {json.dumps(names.get(mm.group(3), mm.group(3)), ensure_ascii=False)}]', line)
        out.append(line)
    return "\n".join(out)


def portal(site: Path) -> None:
    titles = translated_titles()
    pages = [(ROOT / "portal/roadmap/index.html", "roadmap.json", 1, "roadmap/", "../../roadmap/", site / "en/roadmap/index.html", titles),
             (ROOT / "portal/plan/index.html", "plan.json", 1, "plan/", "../../plan/", site / "en/plan/index.html", None),
             (ROOT / "portal/cards/index.html", "cards.json", 1, "cards/", "../../cards/", site / "en/cards/index.html", None)]
    for src, pairs, depth, active, zh_href, out, t in pages:
        if not (ROOT / "i18n" / "en" / pairs).exists():
            print(f"跳过 {out.relative_to(site)}：还没有 i18n/en/{pairs}")
            continue
        text, left = portal_page(src, pairs, depth, active, zh_href, t)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"en 门户：{out.relative_to(site)}（未译片段 {len(left)} 处）")
    if (site / "en/cards").exists():                              # 学习卡页面的脚本是双语的，一份就够
        shutil.copy(ROOT / "portal/cards/cards.js", site / "en/cards/cards.js")
    data = ROOT / "portal/plan/data.js"
    if (ROOT / "i18n/en/plan-data.json").exists():
        pairs = json.loads((ROOT / "i18n/en/plan-data.json").read_text(encoding="utf-8"))
        text, _ = apply_pairs(data.read_text(encoding="utf-8"), pairs)
        (site / "en/plan").mkdir(parents=True, exist_ok=True)
        (site / "en/plan/data.js").write_text(text, encoding="utf-8")
    sys.path.insert(0, str(ROOT / "tools"))
    import site_stats
    site_stats.write_chapters(site / "en/roadmap/chapters.json", roadmap=site / "en/roadmap/index.html", lang="en")


# ---------------------------------------------------------------------------- 翻译辅助
def split_blocks(text: str):
    """把页面切成 (kind, text)：kind 是 "code"（整个围栏代码块，含首尾两行）或 "text" """
    lines = text.split("\n")
    out, buf, i = [], [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            buf.append(lines[i]); i += 1
            continue
        fence, j = m["fence"], i + 1
        while j < len(lines):
            s = lines[j].strip()
            if s.startswith(fence) and set(s) == {fence[0]}:
                break
            j += 1
        if buf:
            out.append(("text", "\n".join(buf))); buf = []
        out.append(("code", "\n".join(lines[i:j + 1])))
        i = j + 1
    if buf:
        out.append(("text", "\n".join(buf)))
    return out


COMMENT = re.compile(r"(?P<pre>(?:#|//)\s?)(?P<body>[^\n]*[㐀-鿿][^\n]*)$")


HASH_LANGS = {"python", "py", "pycon", "bash", "sh", "shell", "console", "toml", "yaml", "yml", "ini", "dockerfile", "make", "makefile", "r"}
SLASH_LANGS = {"c", "cpp", "c++", "cc", "h", "hpp", "cu", "cuda", "rust", "rs", "js", "javascript", "ts", "typescript", "go", "java", "triton"}


def comment_markers(first_line: str) -> tuple[str, ...]:
    """按代码块的语言决定注释符号：Python 里的 // 是整除，C++ 里的 # 是预处理指令"""
    m = re.match(r"\s*(?:`{3,}|~{3,})\s*([\w+#.-]*)", first_line)
    lang = (m.group(1) if m else "").lower()
    if lang in HASH_LANGS:
        return ("#",)
    if lang in SLASH_LANGS:
        return ("//",)
    return ("#", "//")


def is_output_block(first_line: str) -> bool:
    return 'title="输出"' in first_line or bool(re.search(r"^\s*(`{3,}|~{3,})text\b", first_line))


def _comment_pos(line: str, markers: tuple[str, ...] = ("#", "//")) -> int | None:
    """这一行里中文注释的注释符号在哪一列：符号前面是行首或空白、不在字符串里、后面的注释含中文；没有时返回 None"""
    for pos in range(len(line)):
        for mk in markers:
            if line.startswith(mk, pos) and (pos == 0 or line[pos - 1].isspace()) and not _in_string(line, pos):
                return pos if CJK.search(line[pos + len(mk):]) else None
    return None


def _split_comment(line: str, markers: tuple[str, ...]) -> tuple[int, str, str] | None:
    """(注释符号的列, 符号, 注释正文去掉首尾空白)；没有中文注释时返回 None"""
    pos = _comment_pos(line, markers)
    if pos is None:
        return None
    mk = next(m for m in markers if line.startswith(m, pos))
    return pos, mk, line[pos + len(mk):].strip()


def code_comments(code: str) -> list[str]:
    """代码块里含中文的注释，不包括输出块和字符串里的中文"""
    lines = code.split("\n")
    if is_output_block(lines[0]):
        return []
    markers = comment_markers(lines[0])
    return [c[2] for c in (_split_comment(line, markers) for line in lines[1:-1]) if c]


def _in_string(line: str, pos: int) -> bool:
    """pos 处的 # 是不是在字符串字面量里（粗略：数它前面的引号）"""
    head = line[:pos]
    return head.count('"') % 2 == 1 or head.count("'") % 2 == 1


def workdir(book: str) -> Path:
    d = ROOT / book / ".i18n-en" / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract(book: str, page: str) -> None:
    text = (ROOT / book / "docs" / page).read_text(encoding="utf-8")
    parts, codes, comments = [], [], {}
    for kind, chunk in split_blocks(text):
        if kind == "code":
            parts.append(PLACEHOLDER.format(len(codes)))
            codes.append(chunk)
            for c in code_comments(chunk):
                comments.setdefault(c, "")
        else:
            parts.append(chunk)
    base = workdir(book) / page.replace("/", "__")
    Path(str(base) + ".src.md").write_text("\n".join(parts), encoding="utf-8")
    Path(str(base) + ".codes.json").write_text(json.dumps(codes, ensure_ascii=False, indent=1), encoding="utf-8")
    cf = Path(str(base) + ".comments.json")                  # 已经填过的译文保留，新出现的注释补成空串
    old = json.loads(cf.read_text(encoding="utf-8")) if cf.exists() else {}
    cf.write_text(json.dumps({k: old.get(k, "") for k in comments}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{base}.src.md：{len(codes)} 个代码块，{len(comments)} 条中文注释")


def slugify(text: str) -> str:
    from pymdownx.slugs import slugify as make
    return make(case="lower")(text, "-")


def heading_ids(text: str) -> list[tuple[int, str]]:
    """页面里每个标题（代码块外）的级别和 MkDocs 生成的锚点（重复时加 _1、_2，和 toc 扩展一致）"""
    seen, out = set(), []
    for kind, chunk in split_blocks(text):
        if kind == "code":
            continue
        for line in chunk.split("\n"):
            m = HEADING.match(line)
            if not m:
                continue
            title = re.sub(r"\s*\{[^}]*\}$", "", m.group(2))
            plain = re.sub(r"<[^>]+>", "", re.sub(r"`([^`]*)`", r"\1", re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)))
            sid, k = slugify(plain), 1
            base = sid
            while sid in seen:
                sid = f"{base}_{k}"; k += 1
            seen.add(sid)
            out.append((len(m.group(1)), sid))
    return out


def assemble(book: str, page: str) -> None:
    base = workdir(book) / page.replace("/", "__")
    codes = json.loads(Path(str(base) + ".codes.json").read_text(encoding="utf-8"))
    comments = json.loads(Path(str(base) + ".comments.json").read_text(encoding="utf-8"))
    missing = [k for k, v in comments.items() if not v]
    if missing:
        raise SystemExit(f"comments.json 还有 {len(missing)} 条没填：{missing[:3]}")
    en = Path(str(base) + ".en.md").read_text(encoding="utf-8")
    used = set(int(n) for n in re.findall(r"⟦CODE (\d+)⟧", en))
    if used != set(range(len(codes))):
        raise SystemExit(f"占位不全：缺 {sorted(set(range(len(codes))) - used)}，多 {sorted(used - set(range(len(codes))))}")

    dfile = Path(str(base) + ".diagrams.json")             # 示意图（不是程序输出的文字块）可以整块翻译
    diagrams = json.loads(dfile.read_text(encoding="utf-8")) if dfile.exists() else {}

    def put_code(m):
        """只换注释里的文字：字符串字面量、输出块里的中文原样保留（它们是验证过的输出的一部分）"""
        n = int(m.group(1))
        if str(n) in diagrams:
            indent = re.match(r"\s*", codes[n]).group(0)
            return f"{indent}<!-- i18n:diagram {block_hash(codes[n])} -->\n" + diagrams[str(n)].rstrip("\n")
        lines = codes[n].split("\n")
        lines[0] = en_fence(lines[0])                      # 围栏上的中文标题（title="输出"）换成英文
        if is_output_block(lines[0]):
            return "\n".join(lines)
        markers = comment_markers(lines[0])
        for i in range(1, len(lines) - 1):
            c = _split_comment(lines[i], markers)
            if c and c[2] in comments:
                pos, mk, key = c
                lines[i] = lines[i][:pos + len(mk)] + lines[i][pos + len(mk):].replace(key, comments[key], 1)
        return "\n".join(lines)
    en = re.sub(r"⟦CODE (\d+)⟧", put_code, en)

    zh_ids = heading_ids((ROOT / book / "docs" / page).read_text(encoding="utf-8"))
    lines, k = en.split("\n"), 0
    blocks = split_blocks(en)
    out = []
    for kind, chunk in blocks:
        if kind == "code":
            out.append(chunk); continue
        new = []
        for line in chunk.split("\n"):
            m = HEADING.match(line)
            if m:
                if k >= len(zh_ids):
                    raise SystemExit(f"英文版的标题比中文版多：{line}")
                level, sid = zh_ids[k]; k += 1
                if len(m.group(1)) != level:
                    raise SystemExit(f"第 {k} 个标题级别不一致：中文 {level} 级，英文 {line}")
                if level >= 2 and not re.search(r"\{#[^}]*\}\s*$", line):
                    line = f"{line} {{#{esc_anchor(sid)}}}"
            new.append(line)
        out.append("\n".join(new))
    if k != len(zh_ids):
        raise SystemExit(f"英文版的标题比中文版少：中文 {len(zh_ids)} 个，英文 {k} 个")
    target = ROOT / book / "docs-en" / page
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    left = visible_cjk("\n".join(out))
    print(f"写入 {target.relative_to(ROOT)}；正文里仍含中文的行 {len(left)} 行" + ("".join("\n    " + l[:120] for l in left[:8]) if left else ""))


def visible_cjk(text: str) -> list[str]:
    """正文里还剩的中文（不算代码块、链接地址、锚点 {#…}）"""
    out = []
    for kind, chunk in split_blocks(text):
        if kind == "code":
            continue
        for line in chunk.split("\n"):
            line = re.sub(r"\]\([^)]*\)", "](…)", line)
            line = re.sub(r"\{#[^}]*\}", "", line)
            line = re.sub(r'(href|src)="[^"]*"', "", line)
            if CJK.search(line):
                out.append(line.strip())
    return out


def _code_blocks(text: str) -> list[str]:
    return [c for k, c in split_blocks(text) if k == "code"]


DIAGRAM = re.compile(r"^\s*<!-- i18n:diagram ([0-9a-f]{10}) -->\s*$")

# 代码块围栏上的中文标题（```text title="输出"）：英文页换成英文，check 比对时两边都按这张表归一化
TITLE_EN = {
    "输出": "output",
    "输出（本机示例）": "output (on this machine)",
    "运行结果": "what it prints",
    "一次运行的结果": "one run",
    "一次运行的结果（x86 服务器）": "one run (an x86 server)",
    "生成的 CUDA（节选）": "the generated CUDA (excerpt)",
    "编译器的报错（节选）": "the compiler's error (excerpt)",
    "ASan 的报告（节选）": "the ASan report (excerpt)",
    "LeakSanitizer 的报告（节选）": "the LeakSanitizer report (excerpt)",
    "ThreadSanitizer 的报告（节选）": "the ThreadSanitizer report (excerpt)",
    "UBSan 的报告": "the UBSan report",
    "路径 @ 提交 L起-止": "path @ commit Lstart-end",
}
_TITLE = re.compile(r'title="([^"]*)"')


def en_fence(line: str) -> str:
    """围栏行里的中文标题换成英文；表里没有的原样保留"""
    return _TITLE.sub(lambda m: f'title="{TITLE_EN.get(m.group(1), m.group(1))}"', line)


def block_hash(code: str) -> str:
    return hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]


def _en_blocks(text: str) -> list[tuple[str, str | None]]:
    """英文页的代码块，以及它前面有没有"译过的示意图"标记（标记里是对应中文代码块的哈希）"""
    out, prev = [], ""
    for kind, chunk in split_blocks(text):
        if kind == "code":
            last = [l for l in prev.split("\n") if l.strip()][-1:] or [""]
            m = DIAGRAM.match(last[0])
            out.append((chunk, m.group(1) if m else None))
        else:
            prev = chunk
    return out


def stale(book: str, page: str) -> list[str]:
    """英文页和中文页对不上的地方：代码块（除了被译成英文的注释）必须逐字相同，标题的级别与锚点必须一致。
    中文页改了代码而英文页没跟上时，英文页上的代码就不再是验证过的那一份"""
    zh = (ROOT / book / "docs" / page).read_text(encoding="utf-8")
    en = (ROOT / book / "docs-en" / page).read_text(encoding="utf-8")
    zc, eb, msgs = _code_blocks(zh), _en_blocks(en), []
    if len(zc) != len(eb):
        msgs.append(f"代码块数量不同：中文 {len(zc)} 个，英文 {len(eb)} 个")
    else:
        for i, (a, (b, tag)) in enumerate(zip(zc, eb)):
            if tag is not None:                              # 译过的示意图：只要中文原块没变就行
                if tag != block_hash(a):
                    msgs.append(f"第 {i + 1} 个代码块是译过的示意图，中文原图改过了，要重新翻译（{a.strip().splitlines()[0][:40]}…）")
                continue
            al, bl = a.split("\n"), b.split("\n")
            if len(al) != len(bl):
                msgs.append(f"第 {i + 1} 个代码块行数不同（中文 {len(al)} 行，英文 {len(bl)} 行）")
                continue
            markers = comment_markers(al[0])
            for j, (x, y) in enumerate(zip(al, bl)):
                if x == y or (j == 0 and en_fence(x) == y):   # 围栏：允许标题按 TITLE_EN 译成英文
                    continue
                c = _split_comment(x, markers) if 0 < j < len(al) - 1 and not is_output_block(al[0]) else None
                if c is None or not y.startswith(x[:c[0] + len(c[1])]):
                    msgs.append(f"第 {i + 1} 个代码块第 {j + 1} 行不同：{x.strip()[:70]}")
                    break
    zh_ids = heading_ids(zh)
    en_ids = []
    for kind, chunk in split_blocks(en):
        if kind == "text":
            for line in chunk.split("\n"):
                m = HEADING.match(line)
                if m:
                    sid = re.search(r"\{#([^}\s]+)\}\s*$", line)
                    en_ids.append((len(m.group(1)), sid.group(1).replace(chr(92) + "_", "_") if sid else None))
    if [l for l, _ in zh_ids] != [l for l, _ in en_ids]:
        msgs.append(f"标题的数量或级别不同：中文 {len(zh_ids)} 个，英文 {len(en_ids)} 个")
    else:
        for (lv, a), (_, b) in zip(zh_ids, en_ids):
            if lv >= 2 and b != a:
                msgs.append(f"标题锚点不一致：中文是 #{a}，英文写的是 #{b}")
    return msgs


def check(books: list[str]) -> int:
    """所有已译页面和中文版逐一对照（build.sh 在构建英文版之前运行，对不上就构建失败）"""
    bad = 0
    for b in books:
        for path in sorted((ROOT / b / "docs-en").rglob("*.md")) if (ROOT / b / "docs-en").exists() else []:
            page = path.relative_to(ROOT / b / "docs-en").as_posix()
            if not (ROOT / b / "docs" / page).exists():
                print(f"✗ {b}/docs-en/{page}：中文版没有这一页了"); bad += 1
                continue
            for msg in stale(b, page):
                print(f"✗ {b}/docs-en/{page}：{msg}"); bad += 1
    if bad:
        print("中文页面改过之后，英文页要跟上：代码改了用 python3 tools/i18n.py sync <书> <页面>；"
              "标题或段落结构改了，重新 extract / assemble 这一页")
    return 1 if bad else 0


def sync(book: str, page: str) -> None:
    """中文页只改了代码时：把英文页的代码块换成中文页的最新版本，已经译过的注释按"注释前的代码相同"沿用英文，
    新出现的中文注释原样留下并列出来（手工译好再运行 check）"""
    zh = (ROOT / book / "docs" / page).read_text(encoding="utf-8")
    path = ROOT / book / "docs-en" / page
    en = path.read_text(encoding="utf-8")
    zc, blocks = _code_blocks(zh), split_blocks(en)
    ec = [c for k, c in blocks if k == "code"]
    if len(zc) != len(ec):
        raise SystemExit(f"代码块数量变了（中文 {len(zc)}，英文 {len(ec)}）：重新 extract / assemble 这一页")
    todo, out, n, prev = [], [], 0, ""
    for kind, chunk in blocks:
        if kind != "code":
            out.append(chunk)
            prev = chunk
            continue
        last = [l for l in prev.split("\n") if l.strip()][-1:] or [""]
        if DIAGRAM.match(last[0]):                           # 译过的示意图原样保留；中文原图改了的话 check 会报
            out.append(chunk)
            n += 1
            continue
        olds = chunk.split("\n")                           # 旧英文块：注释前的代码相同的行，沿用它的英文注释
        zl = zc[n].split("\n")
        markers = comment_markers(zl[0])
        new = []
        for j, x in enumerate(zl):
            c = _split_comment(x, markers) if 0 < j < len(zl) - 1 and not is_output_block(zl[0]) else None
            if c is None:
                new.append(x)
                continue
            head = x[:c[0] + len(c[1])]
            hit = next((y for y in olds if y.startswith(head) and not CJK.search(y[len(head):])), None)
            if hit is None:
                todo.append(x.strip())
            new.append(hit if hit is not None else x)
        out.append("\n".join(new))
        n += 1
    path.write_text("\n".join(out), encoding="utf-8")
    print(f"已同步 {path.relative_to(ROOT)}" + (f"；这些注释还是中文，要手工译：" + "".join("\n    " + t for t in todo) if todo else ""))


def status(books: list[str]) -> None:
    for b in books:
        pages = sorted(p.relative_to(ROOT / b / "docs").as_posix() for p in (ROOT / b / "docs").rglob("*.md"))
        done = [p for p in pages if (ROOT / b / "docs-en" / p).exists()]
        print(f"{b:8s} {len(done):3d} / {len(pages):3d}")


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    cmd, args = argv[0], argv[1:]
    if cmd == "build":
        site = Path(args[args.index("--site") + 1]) if "--site" in args else ROOT / "_site"
        mkdocs = args[args.index("--mkdocs") + 1] if "--mkdocs" in args else "mkdocs"
        for b in [a for a in args if a in BOOKS] or BOOKS:
            build(b, site.resolve(), mkdocs)
    elif cmd == "extract":
        extract(args[0], args[1])
    elif cmd == "assemble":
        assemble(args[0], args[1])
    elif cmd == "status":
        status(args or BOOKS)
    elif cmd == "check":
        sys.exit(check(args or BOOKS))
    elif cmd == "sync":
        sync(args[0], args[1])
    elif cmd == "portal":
        portal((Path(args[args.index("--site") + 1]) if "--site" in args else ROOT / "_site").resolve())
    elif cmd in ("merge", "config"):
        for b in args or BOOKS:
            print(b, merge(b) if cmd == "merge" else write_config(b))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
