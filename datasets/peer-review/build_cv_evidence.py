#!/usr/bin/env python3
"""Build claim-verification evidence: abstract (claim source) + body (verification target).
For the same 2400 ICLR paper_ids as peer_review_fullpaper_evidence.jsonl, pulls sections from
the PDF DB and assembles body = experiments+results+evaluation+method (where baselines/numbers/
results live). cv modules extract the claim from `abstract`, verify against `body`.

Output: peer_review_cv_evidence.jsonl  {paper_id, y, abstract, body}
"""
import csv, gzip, json, sqlite3, sys

DB = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prpdfs.db"
SRC = "datasets/peer-review/peer_review_fullpaper_evidence.jsonl"
OUT = "datasets/peer-review/peer_review_cv_evidence.jsonl"
BODY_KEYS = ["experiments", "experiment", "results", "result", "evaluation", "method", "methods", "results and discussion"]
CAP = 20000

def main():
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    ids = {r["paper_id"]: r for r in rows}
    con = sqlite3.connect(DB); cur = con.cursor()
    cur.execute("SELECT paper_id, sections FROM pdf_versions WHERE version=0")
    secmap = {}
    for pid, sections in cur.fetchall():
        secmap[pid] = sections
    con.close()

    out = []
    miss = 0
    for r in rows:
        forum = r["paper_id"][5:] if r["paper_id"].startswith("iclr_") else r["paper_id"]
        sections = secmap.get(forum)
        body = ""
        if sections:
            try:
                d = json.loads(sections)
                if isinstance(d, dict):
                    low = {k.lower().strip(): v for k, v in d.items() if isinstance(v, str)}
                    parts = []
                    for k in BODY_KEYS:
                        v = low.get(k)
                        if v and v not in parts:
                            parts.append(v)
                        if sum(len(x) for x in parts) >= CAP:
                            break
                    body = ("\n\n").join(parts)[:CAP]
            except Exception:
                pass
        if not body.strip():
            miss += 1
        out.append({"paper_id": r["paper_id"], "y": r["y"], "abstract": r["abstract"], "body": body})

    with open(OUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} -> {OUT}  (empty body: {miss})", flush=True)

if __name__ == "__main__":
    main()
