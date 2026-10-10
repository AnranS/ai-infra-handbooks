git log -S 'batch_wait_when_idle' --reverse --date=short --format='%ad %h %s' "$REF" -- sglang_omni/scheduling/simple_scheduler.py | head -1
git show -s --format=%B 7fa879d9 | sed -n '3,17p'
