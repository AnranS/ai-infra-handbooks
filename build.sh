#!/usr/bin/env bash
# 构建七本手册，输出到 _site/：python/、cpp/、cuda/、train/、llm/、serving/、minisgl/ 七个子站，加上根目录的总入口页、roadmap/ 学习路线图、plan/ 冲刺计划、practice/ 练习题和 search/ 全站搜索。
# 用法：./build.sh            （使用 PATH 里的 mkdocs）
#       MKDOCS=.venv/bin/mkdocs ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
MKDOCS=${MKDOCS:-mkdocs}

# 首页、路线图、README 等处写的章数、题数与实际同步；路线图没把某一章排进任何一周时构建失败
"${PYTHON:-python3}" tools/site_stats.py --fix

rm -rf _site
mkdir -p _site
for book in python cpp cuda train llm serving minisgl; do
  echo "==> building $book"
  "$MKDOCS" build --strict --config-file "$book/mkdocs.yml" --site-dir "$PWD/_site/$book"
done
cp -R portal/. _site/
mkdir -p _site/assets/brand                                     # logo 与封面：网页图标、分享链接的预览图
cp assets/brand/logo.svg assets/brand/cover.png _site/assets/brand/
cp assets/brand/logo.svg _site/favicon.svg
"${PYTHON:-python3}" tools/site_stats.py --chapters _site/roadmap/chapters.json   # 各章页面上的学习条读它
"${PYTHON:-python3}" practice/build.py _site/practice
"${PYTHON:-python3}" tools/cards.py _site/cards/cards.json   # 学习卡：需要 markdown 与 pymdown-extensions（和 mkdocs 同一个环境）
"${PYTHON:-python3}" tools/search_index.py _site
touch _site/.nojekyll
echo "done: _site/"
