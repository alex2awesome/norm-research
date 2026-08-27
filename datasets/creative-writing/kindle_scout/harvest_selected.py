#!/usr/bin/env python3
"""Enumerate all Kindle-Press-selected campaigns from archived /selected pages."""
import gzip, json, os, re, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
os.makedirs(f"{HERE}/selected_raw", exist_ok=True)
SEL = f"{HERE}/selected_ids.json"
winners = json.load(open(SEL)) if os.path.exists(SEL) else {}
page = max(winners.values()) if winners else 1
while page < 40:
    f = f"{HERE}/selected_raw/page{page}.html.gz"
    if os.path.exists(f) and page in set(winners.values()):
        page += 1
        continue
    url = f"http://web.archive.org/web/20171208id_/https://kindlescout.amazon.com/selected?page={page}"
    try:
        h = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode(errors="ignore")
    except Exception as e:
        print("page", page, "err", e, flush=True)
        time.sleep(90)
        continue
    with gzip.open(f, "wt") as fh:
        fh.write(h)
    ids = re.findall(r"/p/([A-Z0-9]{8,})", h)
    new = [i for i in dict.fromkeys(ids) if i not in winners]
    for i in new:
        winners[i] = page
    json.dump(winners, open(SEL, "w"))
    print(f"page {page}: {len(new)} new (total {len(winners)})", flush=True)
    if not new:
        break
    page += 1
    time.sleep(6)
print("TOTAL selected campaigns:", len(winners), flush=True)
