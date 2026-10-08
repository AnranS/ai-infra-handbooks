for h in cd10654e7e 983bfcf386 fd28640dc5 9183c23eca 923f518337 e3e0bc50a9 bc92107b03 ce32bc2ba9 89588179cf 96a5e4dd79 21028b5507; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
