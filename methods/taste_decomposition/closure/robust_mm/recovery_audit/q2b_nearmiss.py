#!/usr/bin/env python3
"""Recovery-audit Q2(b,c): near-miss autopsy for the 16 missed held-out concepts.

The mechanical tau detector is out of range across registers (max cross-register
cosine .72 < tau .78), so raw cosine cannot decide matches.  What it CAN do is
rank candidates within the same compressed range.  Calibration set: the 8 held-out
concepts the Opus full-recall judges DID match -- their concept->matched-proposal
cosines give the empirical operating band of a TRUE cross-register match.  A missed
concept whose top-1 candidate sits inside that band is a DETECTOR-side near miss
(plausibly a match the judges scored strictly); one whose top-1 sits well below is
a PROPOSER-side miss (nothing adjacent was ever proposed).

Emits the borderline pairs for human re-read.  CPU only, embeddings cached.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RMM = HERE.parent
sys.path.insert(0, str(RMM))
from embed_lib import embed, crit_text, bank_concept_texts  # noqa: E402

det = json.loads((RMM / "m3_detection.json").read_text())
recall = json.loads((RMM / "m3_recall.json").read_text())
bank = bank_concept_texts()

props = {}
for rep in ("rep1", "rep2", "rep3"):
    d = json.loads((RMM / f"proposals_{rep}.json").read_text())
    props[rep] = {p["pid"]: p for p in d["proposals"]}

# ---- calibration: cosine of TRUE (judge-matched) cross-register pairs -------
calib = []
for r in recall["records"]:
    if not r["match_primary"]:
        continue
    ct = crit_text(r["concept"], bank.get(r["concept"], ""))
    for pid in r["matched_pids"]:
        p = props[r["rep"]][pid]
        pt = crit_text(p["name"], p["instruction"])
        cv = embed([ct, pt], verbose=False)
        cos = float(cv[0] @ cv[1])
        calib.append({"rep": r["rep"], "kind": r["kind"], "concept": r["concept"],
                      "pid": pid, "proposal_name": p["name"], "cos": cos})

calib_cos = np.array([c["cos"] for c in calib])
band_lo = float(np.quantile(calib_cos, 0.25))
band_min = float(calib_cos.min())

# ---- per missed held-out concept: top-1 candidate vs the band ---------------
missed = [r for r in recall["records"] if r["kind"] == "heldout" and not r["match_primary"]]
det_by = {}
for rep in ("rep1", "rep2", "rep3"):
    for c in det["replicates"][rep]["concepts"]:
        det_by[(rep, c["concept"])] = c

autopsy = []
for r in missed:
    c = det_by[(r["rep"], r["concept"])]
    top1 = c["top5"][0]
    # judge-either? (matched by one judge but not both -> borderline at judge level)
    either = bool(r["match_either"])
    verdict = ("DETECTOR_NEAR_MISS" if top1["cos"] >= band_lo else
               ("BORDERLINE" if top1["cos"] >= band_min else "PROPOSER_MISS"))
    if either:
        verdict = "JUDGE_SPLIT (one judge called it a match)"
    p = props[r["rep"]].get(top1["pid"], {})
    autopsy.append({
        "rep": r["rep"], "concept": r["concept"], "stratum": r["stratum"],
        "alone_auc": r["alone_auc_fitmine"],
        "top1_cos": top1["cos"], "top1_pid": top1["pid"], "top1_name": top1["name"],
        "top1_instruction": p.get("instruction", "")[:400],
        "concept_definition": (bank.get(r["concept"], "") or "")[:400],
        "top5": [{"cos": round(t["cos"], 3), "name": t["name"]} for t in c["top5"]],
        "judge_either_match": either,
        "verdict": verdict,
    })

autopsy.sort(key=lambda a: -a["top1_cos"])

out = {
    "calibration_true_match_cosines": {
        "n_pairs": len(calib), "min": band_min,
        "q25": band_lo, "median": float(np.median(calib_cos)),
        "max": float(calib_cos.max()),
        "pairs": calib,
    },
    "n_missed_heldout": len(missed),
    "verdict_counts": {},
    "autopsy": autopsy,
}
for a in autopsy:
    v = a["verdict"].split(" ")[0]
    out["verdict_counts"][v] = out["verdict_counts"].get(v, 0) + 1

(HERE / "q2b_nearmiss.json").write_text(json.dumps(out, indent=1))

print(f"TRUE-match cosine band (n={len(calib)}): min={band_min:.3f} q25={band_lo:.3f} "
      f"median={np.median(calib_cos):.3f} max={calib_cos.max():.3f}\n")
print("verdicts:", out["verdict_counts"], "\n")
for a in autopsy:
    print(f"[{a['verdict']}] {a['rep']} ({a['stratum']}, alone {a['alone_auc']:.3f}) "
          f"cos={a['top1_cos']:.3f}\n  CONCEPT : {a['concept']}\n"
          f"    def : {a['concept_definition'][:180]}\n"
          f"  NEAREST: {a['top1_name']}  [{a['top1_pid']}]\n"
          f"    instr: {a['top1_instruction'][:180]}\n")
