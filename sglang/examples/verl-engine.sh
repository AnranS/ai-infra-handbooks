REF=${REF:-29f6d408c0}
git show --stat=100 --format='%ad  %an  %s' --date=short e3e0bc50a9 | grep -v '^$' | cut -c1-96
echo "-- verl_engine.py @ v0.4.6 的方法："; git show v0.4.6:python/sglang/srt/entrypoints/verl_engine.py | grep -E '^class |^    def ' | sed 's/^ *//; s/(.*//' | tr '\n' ' '; echo
echo "-- verl_engine.py 的最后一个提交：$(git log -1 --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/entrypoints/verl_engine.py | cut -c1-80)"
