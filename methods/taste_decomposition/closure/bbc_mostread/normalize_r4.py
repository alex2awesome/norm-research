#!/usr/bin/env python3
"""Normalize proposals_r4.json (harness_bbc collect format) into the
<tag>_proposals_fleet.json contract that the campaign-standard species.py
(copied verbatim from mathse_vote) consumes.

Field map, recorded not silent:
  description      -> instruction
  parent (B only)  -> upstream_parent ('surface-only' when absent)
  mixed 'yes'/'no' -> bool
  pid              -> sha1(proposer|name|description)[:16]  (stable, content-keyed)
  tier             -> 'S' for every proposal (the r2 fleet was SEALED; no directed
                      sweep has run on this cell)
k_A=15 / k_B=10 are the frozen per-round constants (fleet emitted 8x15 A, 8x10 B).
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = json.loads((HERE / "proposals_r4.json").read_text())
out_p = HERE / "bbc_mostread_r4_proposals_fleet.json"

props = []
for p in src["proposals"]:
    pid = hashlib.sha1(f"{p['proposer']}|{p['name']}|{p.get('description','')}"
                       .encode()).hexdigest()[:16]
    rec = dict(track=p["track"], name=p["name"],
               instruction=p.get("description", ""),
               rationale=p.get("rationale", ""),
               pid=pid, proposer=p["proposer"], family=p["family"], tier="S")
    if p["track"] == "B":
        rec["upstream_parent"] = p.get("parent") or "surface-only"
        rec["mixed"] = str(p.get("mixed", "no")).strip().lower() in ("yes", "true", "1")
    props.append(rec)

assert len(props) == src["n_proposals"] == 200, (len(props), src["n_proposals"])
assert len({p["pid"] for p in props}) == len(props), "pid collision"
nA = sum(1 for p in props if p["track"] == "A")
nB = len(props) - nA
assert (nA, nB) == (120, 80), (nA, nB)

out = {"tag": "bbc_mostread_r4", "cell": "bbc_mostread", "round": "4",
       "k_A": 15, "k_B": 10,
       "normalized_from": "proposals_r4.json (harness_bbc collect format)",
       "proposals": props}
out_p.write_text(json.dumps(out, indent=1))
print(f"wrote {out_p.name}: {nA} A + {nB} B, "
      f"{len({p['proposer'] for p in props})} proposers, "
      f"{len({p['family'] for p in props})} families")
