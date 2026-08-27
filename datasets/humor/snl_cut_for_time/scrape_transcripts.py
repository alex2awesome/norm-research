#!/usr/bin/env python3
"""Scrape snltranscripts.jt.org transcript pages for ALL catalog entries that
carry a transcript URL (2,548 aired + 20 cut-for-time). Polite 3s delay,
resume-safe by output presence. PIPELINE NOTE (declared): aired transcripts =
fan transcription; cut-for-time = mostly Whisper-from-audio — class-correlated
source; the 20 cut WITH fan transcripts are the within-class pipeline contrast."""
import hashlib, json, re, time
from pathlib import Path
import requests

D = Path(__file__).resolve().parent
OUT = D / "transcripts_fan"
OUT.mkdir(exist_ok=True)
rows = [json.loads(l) for l in open(D / "snl_catalog.jsonl")]
todo = [r for r in rows if "snltranscripts" in (r.get("url") or "")]
print(f"transcript URLs: {len(todo)}")
UA = {"User-Agent": "Mozilla/5.0 (research corpus; contact alexander.spangher@gmail.com)"}
done = fails = 0
for r in todo:
    key = hashlib.sha256(r["url"].encode()).hexdigest()[:16]
    fp = OUT / f"{key}.json"
    if fp.exists():
        done += 1
        continue
    try:
        resp = requests.get(r["url"], headers=UA, timeout=30)
        if resp.status_code != 200:
            print(f"FAIL {resp.status_code} {r['url'][:70]}", flush=True)
            fails += 1
            time.sleep(10)
            continue
        html = resp.text
        # transcript body: strip tags crudely; refined extraction downstream
        body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S)
        body = re.sub(r"<[^>]+>", "\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        fp.write_text(json.dumps({"url": r["url"], "title": r.get("title"),
                                  "verdict": r["verdict"], "episode": r.get("episode"),
                                  "raw_text": body[:200000]}))
        done += 1
        if done % 100 == 0:
            print(f"{done}/{len(todo)}", flush=True)
    except Exception as e:
        print(f"ERR {type(e).__name__} {r['url'][:60]}", flush=True)
        fails += 1
        time.sleep(15)
    time.sleep(3)
print(f"SNL_TRANSCRIPT_SCRAPE_DONE done={done} fails={fails}")
