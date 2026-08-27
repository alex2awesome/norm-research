#!/bin/bash
cd "$(dirname "$0")"
base="https://nrars.org/0%20The%20Book%20of%20Weeks/archive/new/01%20text"
ok=0; fail=0
while read -r f; do
  [ -s "01_text/$f" ] && continue
  if curl -sk --fail -A "academic-research-collection" "$base/$f" -o "01_text/$f"; then ok=$((ok+1)); else fail=$((fail+1)); echo "FAIL $f" >> dl_failures.log; fi
  sleep 1
done < download_list.txt
echo "done ok=$ok fail=$fail"
