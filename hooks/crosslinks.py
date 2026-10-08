"""MkDocs hook: rewrite cross-handbook links such as `cuda://advanced/attention/` into relative links.

The three handbooks are deployed side by side under one site (python/, cuda/, llm/), so a page can link
to another handbook with `[text](cuda://kernels/gemm/)` and the link keeps working wherever the site is
hosted. `root://roadmap/` links to pages at the shared site root, such as the learning roadmap. Links with a scheme are left alone by MkDocs' own link validation, so the placeholders pass
`mkdocs build --strict`; this hook turns them into real relative hrefs after rendering.
"""

import re

# 英文站（en/）已经有英文版的门户页；其余的 root:// 链接从英文页面指回中文门户
EN_PORTAL = ("", "roadmap/", "plan/", "cards/")
SITES = ("python", "cpp", "cs", "math", "cuda", "train", "llm", "serving", "minisgl", "media", "sglang")
_LINK = re.compile(r'href="(%s)://' % "|".join(SITES))
_ROOT = re.compile(r'href="root://')


def on_page_content(html, page, config, files):
    depth = len(page.url.split("/")[:-1])      # "inference/kv-cache/" -> 2, "" (home page) -> 0
    to_root = "../" * (depth + 1)              # up through this handbook, to the shared site root
    html = _LINK.sub(lambda m: f'href="{to_root}{m.group(1)}/', html)
    if (config.get("extra") or {}).get("lang") == "en":
        def root(m):
            target = m.group(1)
            first = target.split("#")[0].split("?")[0]
            first = first.split("/")[0] + "/" if "/" in first else first
            return f'href="{to_root}{"" if first in EN_PORTAL else "../"}{target}'
        return re.sub(r'href="root://([^"]*)', root, html)
    return _ROOT.sub(f'href="{to_root}', html)
