for h in 3839be2913 530ff541cf f9633fa9b9 976bc302e5 62832bb272 699384cb01 cbedd1db1d 4b0a1c9365 3d32e4a32c 2e4a5907c9 e3b3acfa6f e835a50021; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
