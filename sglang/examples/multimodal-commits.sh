REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | awk '$1 <= "2024-08-31"' | grep -iE 'llava|yi-vl|vision|video' | cut -c1-96 | head -18
