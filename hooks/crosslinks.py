"""MkDocs hook: rewrite cross-handbook links such as `cuda://advanced/attention/` into relative links.

The three handbooks are deployed side by side under one site (python/, cuda/, llm/), so a page can link
to another handbook with `[text](cuda://kernels/gemm/)` and the link keeps working wherever the site is
hosted. `root://roadmap/` links to pages at the shared site root, such as the learning roadmap. Links with a scheme are left alone by MkDocs' own link validation, so the placeholders pass
`mkdocs build --strict`; this hook turns them into real relative hrefs after rendering.
"""

import re

SITES = ("python", "cpp", "cuda", "train", "llm", "serving", "minisgl")
_LINK = re.compile(r'href="(%s)://' % "|".join(SITES))
_ROOT = re.compile(r'href="root://')


def on_page_content(html, page, config, files):
    depth = len(page.url.split("/")[:-1])      # "inference/kv-cache/" -> 2, "" (home page) -> 0
    to_root = "../" * (depth + 1)              # up through this handbook, to the shared site root
    html = _LINK.sub(lambda m: f'href="{to_root}{m.group(1)}/', html)
    return _ROOT.sub(f'href="{to_root}', html)
