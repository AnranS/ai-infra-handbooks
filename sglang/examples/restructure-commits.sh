for h in cdcbde5fc3 87e8c090e9 ab7875941b 75ce37f401 3a6e8b6d78 fec185ce0c 3efa798116 f86c1e611f 3f0fe08d37 36d5acfca5 63ba2f8d7b 99ec439da4 f202ed9712 4ae0969c0a 32eb6e96f2; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
