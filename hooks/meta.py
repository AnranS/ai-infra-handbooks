"""MkDocs hook：每一页的描述（<meta name="description">，以及分享卡片的 og:description）取自这一页开头的导语。

每页正文开头都有一段 <p class="lead"> 导语；不写这个钩子的话，同一本书所有页面的描述都是 site_description，
搜索结果和分享卡片看不出是哪一章。页面自己在 front matter 里写了 description 的，以页面为准。
"""

import html
import re

_LEAD = re.compile(r'<p class="lead">(.*?)</p>', re.S)
_LIMIT = 120                                                  # 搜索结果的摘要大约显示这么多个汉字


def on_page_content(content, page, config, files):
    if page.meta.get("description"):
        return content
    m = _LEAD.search(content)
    if not m:
        return content
    text = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", m.group(1)))).strip()
    # 模板把描述原样放进 content="..."，英文双引号换成中文引号，免得截断属性
    parts = text.split('"')
    text = "".join(p + ("“" if i % 2 == 0 else "”") for i, p in enumerate(parts[:-1])) + parts[-1]
    if len(text) > _LIMIT:
        text = text[:_LIMIT - 1].rstrip("，、；：,.;: ") + "…"
    page.meta["description"] = text
    return content
