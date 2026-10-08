REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %<(15)%an %s' "$REF" | head -4
