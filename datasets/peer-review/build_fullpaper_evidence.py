#!/usr/bin/env python3
"""Build section-targeted full-paper evidence per code-metric aspect, for ICLR papers.

Reads splits/train.csv.gz (ICLR, judgement label) + peer_review_pdfs.db (sections + full_text),
maps splits id iclr_<forum> -> DB <forum>, and assembles per-aspect evidence from the RIGHT
sections (so code-metrics see real methods/baselines/github, not just the abstract).

Evidence per aspect (capped ~5000 chars to fit Gemma-4 4K context):
  a163 positioning/baselines -> related_work + experiments + abstract
  a130 novelty/significance  -> abstract + introduction
  a214 reproducibility       -> windows around github/reproducibility/data-availability in full_text + abstract
  a25 claim-evidence         -> abstract + results + conclusion
  a45 dataset provenance     -> abstract + experiments + evaluation

Output: peer_review_fullpaper_evidence.jsonl  {paper_id, y, abstract, ev:{aspect:text}}
"""
import csv, gzip, json, random, re, sqlite3, sys
from collections import OrderedDict

DB = "/tmp/prpdfs.db"  # set by caller after gunzip
SPLIT = "datasets/peer-review/splits/train.csv.gz"
OUT = "datasets/peer-review/peer_review_fullpaper_evidence.jsonl"
SAMPLE = 2400
CAP = 5000

# section key aliases (lowercased) -> canonical
SEC = {
    "a163": ["related work", "relatedWork", "experiments", "experiment", "evaluation", "abstract", "introduction"],
    "a130": ["abstract", "introduction", "background"],
    "a25":  ["abstract", "results", "result", "results and discussion", "conclusion", "conclusions", "discussion"],
    "a45":  ["abstract", "experiments", "experiment", "evaluation", "results", "method", "methods"],
}
# a214 handled separately via full_text windowing

_REPRO_RE = re.compile(r"github\.com|gitlab\.com|bitbucket\.org|huggingface\.co|zenodo|figshare|"
                       r"code (?:is|will be|can be) (?:available|released|made)|"
                       r"reproducib|data (?:is|will be|can be) (?:available|released|made)|"
                       r"open[- ]source|our code", re.I)

def window(text, hits, radius=500, cap=CAP):
    out, seen = [], set()
    for m in hits:
        a, b = max(0, m.start() - radius), min(len(text), m.end() + radius)
        if a in seen: continue
        seen.add(a)
        out.append("..." + text[a:b].replace("\n", " ") + "...")
        if sum(len(x) for x in out) > cap: break
    return " ".join(out)[:cap]

def build_ev(sections, full_text, abstract):
    sec = {}
    if sections:
        try:
            d = json.loads(sections)
            if isinstance(d, dict):
                sec = {k.lower().strip(): v for k, v in d.items() if isinstance(v, str)}
        except Exception:
            pass
    ab = abstract or sec.get("abstract", "")
    ev = {}
    for aid, keys in SEC.items():
        parts = []
        for k in keys:
            v = sec.get(k)
            if v and k not in parts:
                parts.append(v)
            if sum(len(x) for x in parts) >= CAP: break
        text = ("\n\n[SECTION]\n").join(parts)[:CAP]
        ev[aid] = (text if text.strip() else ab[:CAP])
    # a214: repro windows from full_text + abstract
    ft = full_text or ""
    hits = list(_REPRO_RE.finditer(ft))[:8]
    rep = window(ft, hits) if hits else ""
    ev["a214"] = (rep + "\n\n[ABSTRACT]\n" + ab)[:CAP] if rep else ab[:CAP]
    return ev, ab

def main():
    db = sys.argv[1] if len(sys.argv) > 1 else DB
    con = sqlite3.connect(db); cur = con.cursor()
    # index sections/full_text by paper_id (ICLR, version 0)
    cur.execute("SELECT paper_id, sections, full_text FROM pdf_versions WHERE version=0")
    dbrows = {}
    for pid, sections, ft in cur.fetchall():
        dbrows[pid] = (sections, ft)
    con.close()

    rows = []
    with gzip.open(SPLIT, "rt", errors="ignore") as fh:
        for d in csv.DictReader(fh):
            if "ICLR" not in (d.get("venue") or ""): continue
            try: y = int(d.get("judgement"))
            except: continue
            if y not in (0, 1): continue
            pid = (d.get("paper_id") or d.get("id") or "")
            forum = pid[5:] if pid.startswith("iclr_") else pid  # strip iclr_ prefix
            if forum not in dbrows: continue
            rows.append((pid, y, forum, d.get("text") or ""))
    print(f"ICLR papers in splits with DB full text: {len(rows)}", flush=True)

    if SAMPLE and len(rows) > SAMPLE:
        random.seed(0)
        pos = [r for r in rows if r[1] == 1]; neg = [r for r in rows if r[1] == 0]
        k = SAMPLE // 2
        rows = random.sample(pos, min(k, len(pos))) + random.sample(neg, min(k, len(neg)))
        random.shuffle(rows)

    n_empty = 0
    with open(OUT, "w") as fh:
        for pid, y, forum, abstract in rows:
            sections, ft = dbrows[forum]
            ev, ab = build_ev(sections, ft, abstract)
            if not any(v.strip() for v in ev.values()): n_empty += 1
            fh.write(json.dumps({"paper_id": pid, "y": y, "abstract": ab, "ev": ev}) + "\n")
    print(f"wrote {len(rows)} -> {OUT}  (empty evidence: {n_empty})", flush=True)

if __name__ == "__main__":
    main()
