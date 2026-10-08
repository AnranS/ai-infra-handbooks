REF=${REF:-29f6d408c0}
for h in b36afed4a7 4ac8e09df0 1801cd199f a40229f6f8 904655c5fd 932e263725 ed0fdbf35b 11391b2a1c 00d620b77d 062f6f7ae8; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
