REF=${REF:-29f6d408c0}
echo "基准：$(git log -1 --date=short --format='%H %ad' "$REF")"
echo "到基准为止的提交数：$(git rev-list --count "$REF")，tag 数：$(git tag | wc -l)"
echo "v0.5.21 是 main 的祖先吗：$(git merge-base --is-ancestor v0.5.21 "$REF" && echo 是 || echo 否)"
echo "第一个提交：$(git log --reverse --date=short --format='%ad %h %an %s' "$REF" | head -1 | cut -c1-70)"
