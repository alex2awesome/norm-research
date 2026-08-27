#!/usr/bin/env python3
"""Removal-cell population v2 — MATCHED NORMALIZATION of both classes (leak fix
2026-08-19: v1 had source<->class confound; wayback texts carried HTML entities/
fragments, live-API texts carried markdown/zero-width chars; V .988 = pipeline
fingerprint, RETRACTED). Both classes now pass one renderer: unescape x3, NFKC,
strip invisibles/markdown/urls, ascii quotes, lowercase, whitespace collapse;
torn fragments (<25 chars or mid-sentence start) dropped.
Post-fix probes (grouped-OOF by month): char2-4 LR .30, word1-2 LR .25 — residual
separation is content-shaped (slurs/meta-joke vs question formats), not format
tells. DECLARED CONFOUNDS: length/capture-completeness (short = low-effort real
signal AND possible truncation artifact — not separable while sources differ);
matched-pipeline rebuild (wayback-fetch kept jokes) = gold-standard upgrade if
this cell becomes load-bearing.
"""
import gzip, html, json, re, unicodedata
from pathlib import Path

B = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/humor/reddit_jokes")
INVIS = re.compile(r"[​‌‍⁠﻿\xa0]")
MD = re.compile(r"[*_~`>#\\\[\]()|]")


def norm(t):
    for _ in range(3):
        t2 = html.unescape(t)
        if t2 == t:
            break
        t = t2
    t = unicodedata.normalize("NFKC", t)
    t = INVIS.sub(" ", t)
    for a, b in (("…", "..."), ("’", "'"), ("‘", "'"), ("“", '"'),
                 ("”", '"'), ("—", "-"), ("–", "-")):
        t = t.replace(a, b)
    t = MD.sub(" ", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[.]{2,}", ".", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    rows = [json.loads(l) for l in gzip.open(B / "removal_cell/population.jsonl.gz", "rt")]
    out, n_frag = [], 0
    for r in rows:
        t = norm(str(r["text"]))
        if len(t) < 25 or (t and t[0] in '",)]'):
            n_frag += 1
            continue
        r2 = dict(r)
        r2["text"] = t
        out.append(r2)
    with gzip.open(B / "removal_cell/population_v2.jsonl.gz", "wt") as fh:
        for r in sorted(out, key=lambda x: x["row_id"]):
            fh.write(json.dumps(r) + "\n")
    pos = sum(r["judgement"] for r in out)
    man = {"cell": "jokes_removal_v2 (VERDICT, matched normalization)",
           "n": len(out), "pos_rate": round(pos / len(out), 4),
           "dropped_fragments": n_frag,
           "leak_fix": "v1 RETRACTED (pipeline fingerprint); see module docstring",
           "declared_confounds": ["length/capture-completeness", "created era",
                                  "over_18", "removal-reason mix"]}
    (B / "removal_cell/manifest_v2.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))
    print("REMOVAL_V2_DONE")


if __name__ == "__main__":
    main()
