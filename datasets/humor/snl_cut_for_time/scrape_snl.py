#!/usr/bin/env python3
"""Collect SNL cut-for-time vs aired data.

(a) Fandom Cut_For_Time index (already fetched via MediaWiki API ->
    raw/cut_for_time_wikitext.json).
(b) Aired positives: snltranscripts.jt.org season category listings for
    seasons 2013+ (categories 13..18 and 2019..2023), raw HTML preserved,
    plus a small sample of full transcripts.

Politeness: 2.5s delay, browser UA, robots.txt allows everything scraped here.
Resume: cached files skipped.
"""
import json, os, re, subprocess, time

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
TR = os.path.join(RAW, "transcript_samples")
os.makedirs(TR, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
DELAY = 2.5
CATS = ["13", "14", "15", "16", "17", "18", "2019", "2020", "2021", "2022", "2023"]

def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 2000:
        return "cached"
    r = subprocess.run(["curl", "-sL", "-A", UA, "--compressed", "-m", "40",
                        "-w", "%{http_code}", "-o", path, url],
                       capture_output=True, text=True)
    time.sleep(DELAY)
    return r.stdout.strip()

def main():
    log = open(os.path.join(BASE, "scrape_log.txt"), "a")
    state = {"categories": {}}
    for c in CATS:
        page, got = 1, 0
        while True:
            url = (f"https://snltranscripts.jt.org/category/{c}/" if page == 1
                   else f"https://snltranscripts.jt.org/category/{c}/page/{page}")
            fp = os.path.join(RAW, f"category_{c}_p{page}.html")
            code = fetch(url, fp)
            if code not in ("cached", "200"):
                if os.path.exists(fp) and code != "cached":
                    os.rename(fp, fp + f".err{code}")
                break
            got += 1
            html = open(fp, errors="ignore").read()
            if f"/category/{c}/page/{page+1}" not in html:
                break
            page += 1
        state["categories"][c] = got
        print(f"category {c}: {got} pages", file=log, flush=True)
    json.dump(state, open(os.path.join(BASE, "resume_state.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
