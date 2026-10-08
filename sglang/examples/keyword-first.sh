REF=${REF:-29f6d408c0}
for kw in 'chunk(ed)? prefill' 'overlap' 'xgrammar' 'disagg' 'deepep' 'eplb' 'diffusion'; do
  printf '%-22s %s\n' "$kw" "$(git log --reverse --date=short --format='%ad %h %s' "$REF" | grep -iE "$kw" | head -1 | cut -c1-78)"
done
