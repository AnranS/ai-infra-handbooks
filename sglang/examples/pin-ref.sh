REF=${REF:-29f6d408c0}
echo "基准：$(git log -1 --date=short --format='%H %ad' "$REF")"
BASE=$(git log -1 --date=short --format=%ad "$REF")                   # 基准日：tag 也只数这一天及以前打的，上游以后再发版输出不变
echo "到基准为止的提交数：$(git rev-list --count "$REF")，tag 数：$(git for-each-ref --format='%(creatordate:short)' refs/tags | awk -v d="$BASE" '$1 <= d' | wc -l)"
echo "v0.5.21 是 main 的祖先吗：$(git merge-base --is-ancestor v0.5.21 "$REF" && echo 是 || echo 否)"
echo "第一个提交：$(git log --reverse --date=short --format='%ad %h %an %s' "$REF" | head -1 | cut -c1-70)"
