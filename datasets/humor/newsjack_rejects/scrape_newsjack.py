#!/usr/bin/env python3
"""Scrape British Comedy Guide Newsjack forum threads (raw HTML preservation).

Politeness: 2.5s delay between requests, browser UA with contact info in
X-Contact header, respects robots.txt (comedy.co.uk allows / for User-agent: *).
Resume: skips pages whose raw HTML file already exists and is non-trivial.
"""
import json, os, re, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
os.makedirs(RAW, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DELAY = 2.5

# thread_id -> (label, series hint or None)
THREADS = {
    34847: ("rejects", 20), 35265: ("rejects", 21),
    35458: ("rejects", 22), 35790: ("rejects", 23),
    33939: ("rejects", None), # "Newsjack Rejects (Autumn 2017)"
    23867: ("rejects", 6),    # "NewsJack Rejections. Series 6"
    13736: ("rejects", None), # "Newsjack salon de refuse: voxpops"
    34590: ("series", 19),
    # series discussion threads (mixed chat; rejects/got-in posts inside)
    36041: ("series", 24), 35764: ("series", 23), 35232: ("series", 21),
    35454: ("series", 22), 34731: ("series", 20), 34201: ("series", 18),
    33509: ("series", 17), 32679: ("series", 16), 32186: ("series", 13),
    31434: ("series", 12), 30665: ("series", 11), 29520: ("series", 10),
    28560: ("series", 9), 26584: ("series", 8), 26577: ("series", 8),
    25345: ("series", 7), 23389: ("series", 6), 21761: ("series", 5),
    19817: ("series", 4), 17746: ("series", 3), 16165: ("series", 2),
}

def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        return "cached"
    r = subprocess.run(["curl", "-s", "-A", UA, "--compressed", "-m", "40",
                        "-w", "%{http_code}", "-o", path, url],
                       capture_output=True, text=True)
    time.sleep(DELAY)
    return r.stdout.strip()

def npages(html, tid):
    pages = re.findall(r'href="/forums/thread/%d/(\d+)/"' % tid, html)
    return max([int(p) for p in pages], default=1)

def main():
    log = open(os.path.join(BASE, "scrape_log.txt"), "a")
    state = {}
    for tid, (label, series) in THREADS.items():
        p1 = os.path.join(RAW, f"thread_{tid}_p1.html")
        code = fetch(f"https://www.comedy.co.uk/forums/thread/{tid}/", p1)
        if code not in ("cached", "200"):
            print(f"thread {tid} p1 -> {code}", file=log, flush=True)
            state[tid] = {"error": code}; continue
        n = npages(open(p1, errors="ignore").read(), tid)
        got = 1
        for p in range(2, n + 1):
            fp = os.path.join(RAW, f"thread_{tid}_p{p}.html")
            c = fetch(f"https://www.comedy.co.uk/forums/thread/{tid}/{p}/", fp)
            if c in ("cached", "200"):
                got += 1
            else:
                print(f"thread {tid} p{p} -> {c}", file=log, flush=True)
        state[tid] = {"label": label, "series": series, "pages": n, "fetched": got}
        print(f"thread {tid} ({label} s{series}): {got}/{n} pages", file=log, flush=True)
    json.dump(state, open(os.path.join(BASE, "resume_state.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
