for h in 0463f7fb52 d774acad5c cdcbde5fc3 048685430d f86c1e611f 23cc66f7b6 b48edff67f; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-100
done
