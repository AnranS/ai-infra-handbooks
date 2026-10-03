"""生成站点根目录的 sitemap.xml：一个索引，指向首页等独立页面的地图（sitemap-portal.xml）和八本手册各自的 sitemap.xml。

MkDocs 只给每本手册生成自己的站点地图，首页、学习路线、练习题这些页面不在任何地图里。
注意 GitHub Pages 的项目站点放不了域名根目录的 robots.txt，这张索引要在搜索引擎的站长工具里提交才会被读到。

用法：python3 tools/sitemap.py _site（build.sh 在复制完独立页面之后调用）
"""

import sys
from pathlib import Path

SITE = "https://anrans.github.io/ai-infra-handbooks/"
BOOKS = ["python", "cpp", "cs", "math", "llm", "cuda", "train", "serving", "minisgl", "media", "sglang"]
PORTAL = ["", "roadmap/", "plan/", "practice/", "cards/", "setup/", "search/"]


def main(out: Path) -> None:
    urls = "".join(f"  <url><loc>{SITE}{p}</loc></url>\n" for p in PORTAL if (out / p / "index.html").exists())
    (out / "sitemap-portal.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n' + urls + "</urlset>\n", encoding="utf-8")
    maps = ["sitemap-portal.xml"] + [f"{b}/sitemap.xml" for b in BOOKS if (out / b / "sitemap.xml").exists()]
    (out / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "".join(f"  <sitemap><loc>{SITE}{m}</loc></sitemap>\n" for m in maps) + "</sitemapindex>\n", encoding="utf-8")
    print(f"sitemap: {len(maps)} 张地图 -> {out / 'sitemap.xml'}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
