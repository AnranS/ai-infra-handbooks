"""检查构建好的站点（_site/）里所有站内链接和锚点：目标页面要存在，#锚点要在目标页面里有对应的 id。

用法：python3 tools/check_links.py _site     （CI 在 build.sh 之后运行；有坏链接时返回 1）
外部链接（http/https）不检查；各手册自带的 404.html 不检查（GitHub Pages 只用站点根目录的 404.html）。
"""

from __future__ import annotations

import html.parser
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path


class Page(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and a.get("name"):
            self.ids.add(a["name"])
        for k in ("href", "src"):
            if a.get(k):
                self.links.append(a[k])


def main(site: Path) -> int:
    site = site.resolve()
    pages: dict[Path, Page] = {}
    for f in site.rglob("*.html"):
        if f.name == "404.html" and f.parent != site:
            continue
        p = Page()
        p.feed(f.read_text(encoding="utf-8", errors="replace"))
        pages[f] = p
    broken: dict[str, list[str]] = defaultdict(list)
    for f, p in pages.items():
        if f.name == "404.html":                     # 根目录的 404 页用绝对路径，按站点根目录解析
            continue
        for url in p.links:
            u = urllib.parse.urlsplit(url)
            if u.scheme or url.startswith(("//", "#/", "mailto:", "javascript:", "data:")):
                continue
            path = urllib.parse.unquote(u.path)
            target = f if not path else (site / path.lstrip("/") if path.startswith("/") else f.parent / path).resolve()
            if target.is_dir() or path.endswith("/"):
                target = target / "index.html"
            src = str(f.relative_to(site))
            if not target.exists():
                broken[f"缺页面 {target.relative_to(site) if target.is_relative_to(site) else target}"].append(src)
                continue
            frag = urllib.parse.unquote(u.fragment)
            if frag and target.suffix == ".html" and not frag.startswith("/") and target in pages and frag not in pages[target].ids:
                broken[f"缺锚点 {target.relative_to(site)}#{frag}"].append(src)
    print(f"检查了 {len(pages)} 个页面，{sum(len(p.links) for p in pages.values())} 个链接：", end="")
    if not broken:
        print("没有坏链接")
        return 0
    print(f"{len(broken)} 个坏目标")
    for k, srcs in sorted(broken.items()):
        print(f"  {k}  ← {', '.join(sorted(set(srcs))[:3])}{' 等' if len(set(srcs)) > 3 else ''}")
    return 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site")))
