#!/usr/bin/env python3
"""Kindle Scout (2014-2018) campaign recovery from Wayback.

Kindle Scout = reader-nominated, EDITORIALLY decided book deals (Kindle Press).
Verdict = editorial accept/reject on full campaigns; excerpt text is inline.

Stage 1 (done offline): CDX enumeration -> ks_campaigns.json
  {campaign_id: [[timestamp, original_url], ...]} (200/301 captures only)
Stage 2: per campaign, fetch the LATEST main-page capture (status most final)
  + the expand-excerpt capture when present; extract fields; store raw HTML.

Verdict signals recorded per campaign:
  - page status text (days left / in review / selected / not selected)
  - any capture URL carrying ref_=ks_sel (crawled via Selected carousel)
  Final accept labels are cross-checked later against the Kindle Press
  published-books list (external join).

Gentle pacing (3s), resumable, append-only. Wayback raw bytes via id_ modifier.
"""
import gzip
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPS_F = os.path.join(HERE, "ks_campaigns.json")
OUT_F = os.path.join(HERE, "campaigns.jsonl")
RAW_DIR = os.path.join(HERE, "raw")
STATE = os.path.join(HERE, "state.json")
UA = {"User-Agent": "norm-research/0.1 (academic; contact alex2awesome@gmail.com)"}
RATE = 3.0


def fetch(url, t=90):
    for a in range(5):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=t).read().decode(errors="ignore")
        except Exception as e:
            time.sleep(120 if ("429" in str(e) or "503" in str(e)) else 5 * (a + 1))
    return ""


def extract(h):
    d = {}
    m = re.search(r"<title>\s*([^<]+?)\s*</title>", h)
    d["page_title"] = m.group(1) if m else None
    m = re.search(r'class="[^"]*cover-title[^"]*"[^>]*>\s*([^<]+)', h)
    if not m:
        m = re.search(r'<h1[^>]*>\s*([^<]{2,120})', h)
    d["book_title"] = m.group(1).strip() if m else None
    m = re.search(r"[Bb]y\s+<[^>]+>\s*([^<]{2,60})", h)
    d["author"] = m.group(1).strip() if m else None
    d["genres"] = sorted(set(re.findall(
        r">\s*(Mystery, Thriller & Suspense|Science Fiction & Fantasy|Romance|"
        r"Literature & Fiction|Teen & Young Adult|Action & Adventure|Contemporary Fiction|Horror)\s*<", h)))
    m = re.search(r'id="action-bar"\s+class="action-bar-([a-z-]+)"', h)
    d["action_bar"] = m.group(1) if m else None
    m = re.search(r'id="action-bar"[^>]*>\s*<div class="message">\s*(.*?)</div>', h, re.S)
    if m:
        d["action_msg"] = re.sub(r"<[^>]+>|\s+", " ", m.group(1)).strip()[:200]
    paras = re.findall(r"<p[^>]*>\s*([^<]{120,})</p>", h)
    d["excerpt_paras"] = len(paras)
    d["excerpt_chars"] = sum(len(p) for p in paras)
    return d


def main():
    camps = json.load(open(CAMPS_F))
    done = set()
    if os.path.exists(OUT_F):
        for line in open(OUT_F):
            done.add(json.loads(line)["campaign_id"])
    os.makedirs(RAW_DIR, exist_ok=True)
    out = open(OUT_F, "a")
    todo = sorted(camps)
    print(f"{len(todo)} campaigns, {len(done)} done", flush=True)
    for i, cid in enumerate(todo):
        if cid in done:
            continue
        caps = camps[cid]
        main_caps = [c for c in caps if "/expand-excerpt" not in c[1]]
        exp_caps = [c for c in caps if "/expand-excerpt" in c[1]]
        rec = {"campaign_id": cid,
               "n_captures": len(caps),
               "sel_carousel": any("ks_sel" in u for _, u in caps)}
        pick = max(main_caps) if main_caps else None
        if pick:
            h = fetch(f"http://web.archive.org/web/{pick[0]}id_/{pick[1]}")
            if h:
                with gzip.open(os.path.join(RAW_DIR, f"{cid}.html.gz"), "wt") as fh:
                    fh.write(h)
                rec.update(extract(h))
                rec["capture"] = pick[0]
            time.sleep(RATE)
        if exp_caps:
            e = max(exp_caps)
            h = fetch(f"http://web.archive.org/web/{e[0]}id_/{e[1]}")
            if h:
                with gzip.open(os.path.join(RAW_DIR, f"{cid}.excerpt.html.gz"), "wt") as fh:
                    fh.write(h)
                rec["has_expand_excerpt"] = True
            time.sleep(RATE)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(todo)}", flush=True)
    out.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
