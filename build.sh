#!/usr/bin/env bash
# 构建三本手册，输出到 _site/：python/、cuda/、llm/ 三个子站，加上根目录的总入口页。
# 用法：./build.sh            （使用 PATH 里的 mkdocs）
#       MKDOCS=.venv/bin/mkdocs ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
MKDOCS=${MKDOCS:-mkdocs}

rm -rf _site
mkdir -p _site
for book in python cuda llm serving; do
  echo "==> building $book"
  "$MKDOCS" build --strict --config-file "$book/mkdocs.yml" --site-dir "$PWD/_site/$book"
done
cp portal/index.html portal/favicon.svg _site/
touch _site/.nojekyll
echo "done: _site/"
