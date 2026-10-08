for h in 22085081bb 01ca82d765 26f0bedc8f 0463f7fb52 d774acad5c 665815969a cdcbde5fc3 e1eae1fd15 f86c1e611f 99ec439da4 dbec2f1847 7d671e4ad2 976bc302e5 cbedd1db1d 419a57e771 815dce0554 6c7a152c5a c76040e31b c7c7dbebbe f44db16c8e 0d47788025 70c471a868 ce32bc2ba9 53ca15529a 7bc1dae095 b36afed4a7 49dfa1d891; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-80
done | sort
