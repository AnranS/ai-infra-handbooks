for h in 21463e321a c553e1604c f44db16c8e 4c56e5dbee ca75741e86 23c764b18a febe21ce03 77e929a1a2 f0653886a5 cba1cdbc46 ccfe5c009d 7a80f56513 0d47788025 32fa1e9cc2; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
