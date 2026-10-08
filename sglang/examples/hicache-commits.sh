for h in 51caee740f 5d6e9467d4 13387e6b7a 6c7a152c5a 10b544ae9b 0e0ec70200 e119f04215 a023856b12 9d33fcfb8e 299803343d 2cd2e27f80 9f78f391ae 8b6966d020 1ccd59c715; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
