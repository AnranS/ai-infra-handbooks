"""MkDocs hook：按导航顺序扫描各章代码块的文件名，算出每一章新建 / 修改了哪些文件，渲染成文件树。

页面是唯一的源：一章里 ```python title="qwen/x.py" 的块就是那一章写（或改）的文件，不用另外维护清单。
    @@files@@         本章之后的文件树：+ 本章新建、~ 本章修改，其余是前面各章写好的
    @@files all@@     全书最终的文件树：每个文件标出在哪一章新建、被哪些章改过
    @@files table@@   逐章一张表：新建 / 修改 / 本章的脚本
英文站（extra.lang == en）用英文标注；代码块在中英文页里逐字相同，所以两边算出来的树一样。
"""
import re
from pathlib import Path

TITLE = re.compile(r'^```python title="([\w./-]+\.py)"', re.M)
NAV = re.compile(r"([\w./-]+\.md)\s*$", re.M)
H1 = re.compile(r"^# (.+)$", re.M)
LABEL = re.compile(r"^(（[一二三四五六七八九十]+）|第 0 步|Step 0|\d+\.)")
_scan_cache: dict = {}


def _chapters(config) -> list[dict]:
    """导航顺序里有代码块的页面：{"page", "label", "new", "mod", "scripts", "files"}（files 是到这一章为止的全部文件）"""
    key = config["config_file_path"]
    if key in _scan_cache:
        return _scan_cache[key]
    docs = Path(config["docs_dir"])
    nav = Path(key).read_text(encoding="utf-8").split("\nnav:", 1)[1]
    seen: dict[str, str] = {}                                   # 文件 -> 新建它的章
    out = []
    for rel in NAV.findall(nav):
        text = (docs / rel).read_text(encoding="utf-8")
        titles = TITLE.findall(text)
        if not titles:
            continue
        h1 = H1.search(text)
        m = LABEL.match(h1.group(1)) if h1 else None
        label = m.group(1).rstrip(".") if m else rel[:-3]
        files = [t for t in dict.fromkeys(titles) if "/" in t]
        scripts = [t for t in dict.fromkeys(titles) if "/" not in t]
        new = [t for t in files if t not in seen]
        mod = [t for t in files if t in seen]
        for t in new:
            seen[t] = label
        out.append({"page": rel, "label": label, "new": new, "mod": mod, "scripts": scripts, "files": list(seen)})
    _scan_cache[key] = out
    return out


def _tree(entries: list[tuple[str, str]]) -> str:
    """entries: [(文件名, 标注 html)] -> <pre> 里的树（qwen/ 下的文件缩进，其余平铺）"""
    pkg = [(f, n) for f, n in entries if "/" in f]
    top = [(f, n) for f, n in entries if "/" not in f]
    width = max([len(f.split("/", 1)[1]) + 4 for f, _ in pkg] + [len(f) for f, _ in top] + [20]) + 2
    lines = []
    if pkg:
        lines.append(pkg[0][0].split("/")[0] + "/")
        for i, (f, note) in enumerate(pkg):
            branch = "└── " if i == len(pkg) - 1 else "├── "
            lines.append(f"{branch + f.split('/', 1)[1]:<{width}}{note}")
    for f, note in top:
        lines.append(f"{f:<{width}}{note}")
    return "\n".join(lines)


def _render(kind: str, page: str, chapters: list[dict], en: bool) -> str:
    T = {"new": "+ new in this chapter", "mod": "~ changed in this chapter", "script": "+ this chapter's script",
         "earlier": "scripts of the earlier chapters", "legend": "<code>+</code> new in this chapter · <code>~</code> changed in this chapter · the rest were written in earlier chapters",
         "created": "created in", "changed": "changed in", "scripts": "one script per chapter",
         "th": ("Chapter", "New files", "Changed files", "Script")} if en else \
        {"new": "+ 本章新建", "mod": "~ 本章修改", "script": "+ 本章的脚本", "earlier": "前面各章的脚本",
         "legend": "<code>+</code> 本章新建 · <code>~</code> 本章修改 · 没有标记的是前面各章写好的", "created": "", "changed": "修改",
         "scripts": "每章一个脚本", "th": ("章", "新建", "修改", "脚本")}
    if kind == "table":
        rows = "".join(f"<tr><td>{c['label']}</td><td>{', '.join(f'<code>{f}</code>' for f in c['new']) or '–'}</td>"
                       f"<td>{', '.join(f'<code>{f}</code>' for f in c['mod']) or '–'}</td><td>{', '.join(f'<code>{s}</code>' for s in c['scripts'])}</td></tr>"
                       for c in chapters)
        th = "".join(f"<th>{h}</th>" for h in T["th"])
        return f'<table class="aig-filetable"><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table>'
    if kind == "all":
        created = {f: c["label"] for c in chapters for f in c["new"]}
        changed: dict[str, list[str]] = {}
        for c in chapters:
            for f in c["mod"]:
                changed.setdefault(f, []).append(c["label"])
        entries = []
        for f, lab in created.items():
            note = f'<span class="old">{T["created"]} {lab}' if en else f'<span class="old">{lab}'
            if f in changed:
                note += ("; " + T["changed"] + " " + ", ".join(changed[f])) if en else ("，" + "、".join(changed[f]) + T["changed"])
            entries.append((f, note + "</span>"))
        first, last = chapters[0]["scripts"][0], chapters[-1]["scripts"][-1]
        entries.append((f"{first} … {last}", f'<span class="old">{T["scripts"]}</span>'))
        return f'<pre class="aig-filetree">{_tree(entries)}</pre>'
    cur = next((c for c in chapters if c["page"] == page), None)
    if cur is None:
        return ""
    entries = []
    for f in cur["files"]:
        if f in cur["new"]:
            entries.append((f, f'<span class="new">{T["new"]}</span>'))
        elif f in cur["mod"]:
            entries.append((f, f'<span class="mod">{T["mod"]}</span>'))
        else:
            entries.append((f, ""))
    earlier = [s for c in chapters[: chapters.index(cur)] for s in c["scripts"]]
    if earlier:
        entries.append((f"{earlier[0]} … {earlier[-1]}" if len(earlier) > 1 else earlier[0], f'<span class="old">{T["earlier"]}</span>'))
    for s in cur["scripts"]:
        entries.append((s, f'<span class="new">{T["script"]}</span>'))
    return f'<pre class="aig-filetree">{_tree(entries)}</pre>\n<p class="aig-filetree-legend">{T["legend"]}</p>'


def on_page_markdown(markdown, page, config, files):
    if "@@files" not in markdown:
        return markdown
    en = (config.get("extra") or {}).get("lang") == "en"
    chapters = _chapters(config)
    return re.sub(r"^@@files( all| table)?@@$", lambda m: _render((m.group(1) or "").strip() or "this", page.file.src_path, chapters, en),
                  markdown, flags=re.M)
