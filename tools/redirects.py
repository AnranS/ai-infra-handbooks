"""给搬了家的页面在旧地址留一个跳转页，旧链接和书签继续能用（GitHub Pages 不支持服务端重定向）。

用法：python3 tools/redirects.py _site        （build.sh 在各书构建完之后调用）
"""

import sys
from pathlib import Path

# 旧路径 -> 新路径（都相对站点根目录）
MOVED = {
    # 2026-10：数学基础从大模型原理手册拆成单独的一本
    **{f"llm/math/{p}/": f"math/{p}/" for p in
       ("linear-algebra", "probability", "information-theory", "calculus", "floating-point", "performance-math")},
}

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>页面已移动</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="{to}">
<meta http-equiv="refresh" content="0; url={to}">
<script>location.replace("{to}" + location.hash)</script>
</head><body><p>这一页已移到 <a href="{to}">新的位置</a>。</p></body></html>
"""


def main(site: Path) -> None:
    for old, new in MOVED.items():
        out = site / old / "index.html"
        if out.exists():
            raise SystemExit(f"{old} 现在有真实页面，不能再放跳转页")
        to = "../" * old.rstrip("/").count("/") + "../" + new          # 从旧页面目录回到站点根目录
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(PAGE.format(to=to), encoding="utf-8")
    print(f"redirects: {len(MOVED)} 个旧地址 -> {site}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
