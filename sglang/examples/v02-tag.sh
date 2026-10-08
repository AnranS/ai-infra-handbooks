echo "v0.2.0：$(git log -1 --date=short --format='%ad %h %s' v0.2.0 | cut -c1-70)"
echo "v0.1.5 → v0.2.0 的提交数：$(git rev-list --count v0.1.5..v0.2.0)"
echo "这段时间提交最多的作者："; git shortlog -sn v0.1.5..v0.2.0 | head -6
