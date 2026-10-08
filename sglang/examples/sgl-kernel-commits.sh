for h in 419a57e771 5c91a315d7 47eb139f81 84d96b3ae5 9286740eff 6b45a21d16 110e006673 c553e1604c 54b9a2de0a c32c4ef79c; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
