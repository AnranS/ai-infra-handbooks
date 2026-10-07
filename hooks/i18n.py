"""MkDocs hook：英文站（config.extra.lang == "en"）里，代码块标题等界面上的中文换成英文。

为了让各书的代码校验脚本照常识别，英文页面的代码块原样保留了 title="输出" 这类写法；这里只在渲染出的 HTML 里替换。
"""

_TITLES = {
    '<span class="filename">输出</span>': '<span class="filename">Output</span>',
    "（本机示例）</span>": " (local example)</span>",
}


def on_page_content(html, page, config, files):
    if (config.get("extra") or {}).get("lang") != "en":
        return html
    for zh, en in _TITLES.items():
        html = html.replace(zh, en)
    return html
