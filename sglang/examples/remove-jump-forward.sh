git show --format='%ad  %h  %an%n%s%n%b' --date=short --stat=90 935cda944b | grep -v '^$' | grep -iE 'jump|^2025|Misc|constrained|schedule|infer|batch' | head -20
