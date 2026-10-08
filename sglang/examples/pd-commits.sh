for h in c7c7dbebbe 6d3b35fae9 4c31ae9f6d a9499885e9 ab4b5606e4 4dce1cc608 e65b9f21e3 bf98d2e377 711efe7814 e0673969b9 3f57b00a59; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
