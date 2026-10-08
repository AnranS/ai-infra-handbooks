#!/usr/bin/env bash
# 构建十二本手册，输出到 _site/：python/、cpp/、cs/、math/、cuda/、scratch/、train/、llm/、serving/、minisgl/、media/、sglang/ 十二个子站，加上根目录的总入口页、roadmap/ 学习路线图、plan/ 冲刺计划、practice/ 练习题和 search/ 全站搜索，以及 en/ 下的英文版。
# 用法：./build.sh            （使用 PATH 里的 mkdocs）
#       MKDOCS=.venv/bin/mkdocs ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
MKDOCS=${MKDOCS:-mkdocs}

# 首页、路线图、README 等处写的章数、题数与实际同步；路线图没把某一章排进任何一周时构建失败
"${PYTHON:-python3}" tools/site_stats.py --fix
"${PYTHON:-python3}" tools/export_examples.py --check   # <书>/examples/ 里的文件和正文里的代码块逐字一致

rm -rf _site
mkdir -p _site
for book in python cpp cs math cuda scratch train llm serving minisgl media sglang; do
  echo "==> building $book"
  "$MKDOCS" build --strict --config-file "$book/mkdocs.yml" --site-dir "$PWD/_site/$book"
done
cp -R portal/. _site/
"${PYTHON:-python3}" tools/i18n.py check                    # 已译页面的代码块、标题和中文版一致（中文改了代码而英文没跟上时构建失败）
"${PYTHON:-python3}" tools/i18n.py build --site _site --mkdocs "$MKDOCS"   # 英文版：en/<书>/，没译的页面先显示中文原文
"${PYTHON:-python3}" tools/i18n.py portal --site _site       # 英文的路线图与计划页：由中文页面按 i18n/en/ 的对照表生成
"${PYTHON:-python3}" tools/redirects.py _site               # 搬了家的页面：旧地址留跳转页（中英文站都建好之后再放）
mkdir -p _site/assets/brand                                     # logo 与封面：网页图标、分享链接的预览图
cp assets/brand/logo.svg assets/brand/cover.png _site/assets/brand/
cp assets/brand/logo.svg _site/favicon.svg
"${PYTHON:-python3}" tools/site_stats.py --chapters _site/roadmap/chapters.json   # 各章页面上的学习条读它
"${PYTHON:-python3}" practice/build.py _site/practice
"${PYTHON:-python3}" tools/cards.py _site/cards/cards.json   # 学习卡：需要 markdown 与 pymdown-extensions（和 mkdocs 同一个环境）
"${PYTHON:-python3}" tools/cards.py _site/en/cards/cards.json --lang en   # 英文学习卡：抽自各书的英文页，卡片 id 与中文版相同
"${PYTHON:-python3}" tools/search_index.py _site
"${PYTHON:-python3}" tools/sitemap.py _site              # 根目录的站点地图索引：独立页面 + 各本手册
touch _site/.nojekyll
echo "done: _site/"
