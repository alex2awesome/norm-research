#!/usr/bin/env python3
"""peer_revealed identity-leak audit, step 1: fetch authorships for every row of
datasets/peer-review/vat_3y/revealed.jsonl from OpenAlex (rows carry OpenAlex work
ids). Batch filter endpoint, 50 ids/call, polite pool. Output: one jsonl row per
work — author ids/names, institution ids/names (first listing per author), and
cited_by_count (for later fame-stratified analyses). Append-safe: skips ids
already fetched.
"""
import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[2] / "datasets/peer-review/vat_3y/revealed.jsonl"
OUT = HERE / "openalex_authorships.jsonl"
MAILTO = "alex2awesome@gmail.com"

ids = []
for line in open(SRC):
    r = json.loads(line)
    wid = r["id"].rsplit("/", 1)[-1]
    if wid.startswith("W"):
        ids.append(wid)
ids = sorted(set(ids))

done = set()
if OUT.exists():
    for line in open(OUT):
        try:
            done.add(json.loads(line)["work_id"])
        except Exception:
            pass
todo = [w for w in ids if w not in done]
print(f"{len(ids)} unique works, {len(done)} already fetched, {len(todo)} to go", flush=True)

S = requests.Session()
with open(OUT, "a") as fh:
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        url = ("https://api.openalex.org/works?filter=openalex_id:"
               + "|".join(chunk)
               + f"&per-page=50&select=id,display_name,publication_year,cited_by_count,authorships&mailto={MAILTO}")
        for attempt in range(5):
            try:
                resp = S.get(url, timeout=60)
                if resp.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(5 * (attempt + 1))
        else:
            print(f"batch {i//50}: FAILED after retries, skipping", flush=True)
            continue
        got = resp.json().get("results", [])
        for w in got:
            auth = [{
                "author_id": (a.get("author") or {}).get("id"),
                "author_name": (a.get("author") or {}).get("display_name"),
                "institutions": [{"id": inst.get("id"), "name": inst.get("display_name"),
                                  "type": inst.get("type"), "country": inst.get("country_code")}
                                 for inst in (a.get("institutions") or [])],
            } for a in (w.get("authorships") or [])]
            fh.write(json.dumps({
                "work_id": w["id"].rsplit("/", 1)[-1],
                "title": w.get("display_name"),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count"),
                "authors": auth,
            }) + "\n")
        fh.flush()
        print(f"batch {i//50 + 1}/{(len(todo)+49)//50}: +{len(got)}", flush=True)
        time.sleep(0.3)
print("FETCH_DONE", flush=True)
