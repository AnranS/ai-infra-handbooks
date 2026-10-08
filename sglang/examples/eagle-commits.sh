for h in 9c05c6898e b0524c3789 ad20b7957e 815dce0554 8c8779cd05 013021b6a1 f9905d59a8 862dd76c76 9fafa62db7 9fb48f951f e983e43248 b26bc86b36; do
  git log -1 --date=short --format='%ad  %h  %s' $h 2>/dev/null | cut -c1-96
done
