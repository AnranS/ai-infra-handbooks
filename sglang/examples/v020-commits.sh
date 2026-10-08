for h in 665815969a c126a6ccba d774acad5c eedc12e12e 679ebcbbdc cdcbde5fc3 7cd4f244a4 e1eae1fd15; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
