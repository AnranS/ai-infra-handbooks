"""给 _site 里每个页面引用的本站 CSS / JS 加上版本号 ?v=<内容哈希>。

GitHub Pages 让浏览器把文件缓存 10 分钟（max-age=600），刷新页面时样式和脚本直接用缓存。
刚部署完，新页面就会配上缓存里的旧脚本，例如新加的交互小工具在旧脚本里还没有，只显示一个空框。
加上内容哈希后，文件一改网址就变，浏览器会去取新的；文件没改，哈希不变，缓存照样有效。
只处理页面里写死的 <script src> 和 <link href>，脚本运行时再加载的文件（Worker、fetch 的数据）不管。

用法：python3 tools/cachebust.py _site（build.sh 在生成完全部页面之后调用）
"""

import hashlib
import re
import sys
from pathlib import Path

REF = re.compile(r'(<(?:script|link)\b[^>]*?\s(?:src|href)=")([^"?#:]+\.(?:js|css))(")')


def main(site: Path) -> None:
    hashes: dict[Path, str] = {}
    n_pages = n_refs = 0
    for page in site.rglob("*.html"):
        html = page.read_text(encoding="utf-8")

        def sub(m: re.Match) -> str:
            nonlocal n_refs
            url = m.group(2)
            if url.startswith("/"):                                  # 站点在 /ai-infra-handbooks/ 下，绝对路径不好解析，页面里也没有
                return m.group(0)
            target = (page.parent / url).resolve()
            if not target.is_file():                                 # 比如 404 页里用脚本拼出来的地址
                return m.group(0)
            if target not in hashes:
                hashes[target] = hashlib.md5(target.read_bytes()).hexdigest()[:8]
            n_refs += 1
            return f"{m.group(1)}{url}?v={hashes[target]}{m.group(3)}"

        new = REF.sub(sub, html)
        if new != html:
            page.write_text(new, encoding="utf-8")
            n_pages += 1
    print(f"cachebust: {n_pages} 个页面、{n_refs} 处引用加了版本号（{len(hashes)} 个文件）")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "_site"))
