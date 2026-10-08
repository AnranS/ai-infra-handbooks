for h in 22085081bb d774acad5c 7d671e4ad2 cdcbde5fc3 99ec439da4 c76040e31b 2d96da813e 419a57e771 815dce0554 03464890e0; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
