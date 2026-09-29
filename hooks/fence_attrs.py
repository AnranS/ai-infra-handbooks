"""MkDocs hook: strip the attributes that cpp/tools/check_code.py reads from code fences.

Fences like ```cpp title="x.cpp" sanitize="thread" expect="fail"``` carry instructions for the example checker.
pymdownx.superfences only understands a few options (title, linenums, hl_lines); unknown ones make it reject the
fence and render the code as plain paragraphs. This hook removes the checker-only attributes before rendering,
so the page shows a normal code block with its title while the Markdown source keeps the instructions.
"""

import re

_KEYS = "sanitize|expect|error|flags|libs|with|lib|project|run"
_FENCE = re.compile(r"^([ \t]*`{3,}[\w+-]*[^\n`]*)$", re.M)
_ATTR = re.compile(r'\s+(?:%s)="[^"]*"' % _KEYS)


def on_page_markdown(markdown, page, config, files):
    return _FENCE.sub(lambda m: _ATTR.sub("", m.group(1)), markdown)
