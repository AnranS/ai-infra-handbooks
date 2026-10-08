REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %s' "$REF" | grep -iE 'remove.*vllm|vllm.*depend|decouple.*vllm|adapt vllm' | cut -c1-100
