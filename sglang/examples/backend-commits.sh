for h in 36f6fc5093 c76040e31b a53fe428f9 b6944f97a6 5d7edc8e55 26c0f13126 20c90be23d 1c63e79756 e983e43248 51d25405a7; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
