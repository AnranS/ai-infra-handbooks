for h in 3839be2913 530ff541cf cbedd1db1d 2e4a5907c9 e3b3acfa6f c8f31042a8 8c7bb39dfb 6e316588f8 9768c50d90 53ca15529a 49dfa1d891; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
