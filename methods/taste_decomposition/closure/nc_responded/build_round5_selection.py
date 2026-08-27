#!/usr/bin/env python3
"""Assemble the ROUND-5 scored set: the campaign's cap round, spent on the two
addenda that landed mid-campaign.

Composition (25 criteria, the frozen per-round scoring budget):
  * 16 = FREEZE ADDENDUM 3 decomposition components -- 8 MIXED parents x 2 components
    (one candidate-real, one surface). Parents were ranked by |alone-AUC - .5| on
    FIT+MINE only. Parents are retired from the Track-B readouts once these score.
  * 5  = FREEZE ADDENDUM 4 position-in-container channels: textual fingerprints of a
    comment's position in its docket. (The programmatic audit of the REAL position
    variables is separate -- see position_audit.py.)
  * 4  = the round's 2 planted probe pairs, as in every other round.

RECORDED DEVIATION. The frozen split is k_A=15 / k_B=10. Round 5's content is dictated
by the two addenda, so the AUTHOR-PROPOSED split is 10 A / 15 B instead. The scored
total is unchanged at 25, and the routing is decided by the blind audit as always --
the proposed track only feeds the misrouting rate. Recorded here rather than silently
absorbed.

Usage: python build_round5_selection.py
"""
from __future__ import annotations

import json
from pathlib import Path

from select_round_set import PROBES

HERE = Path(__file__).resolve().parent


def main():
    dec = json.loads((HERE / "round5_decomposition.json").read_text())
    pos = json.loads((HERE / "round5_position_raw.json").read_text())["criteria"]

    out = {"round": 5,
           "rule": "FREEZE ADDENDUM 3 decomposition components + FREEZE ADDENDUM 4 "
                   "position-in-container channels + 2 planted probe pairs",
           "recorded_deviation": "author-proposed split is 10 A / 15 B, not k_A=15/k_B=10; "
                                 "scored total unchanged at 25; routing decided by the blind audit",
           "A": [], "B": [], "probes": [], "substitutions": {}}

    for c in dec["components"]:
        rec = {"source": "addendum3_decomposition", "stratum": f"component_{c['kind']}",
               "parent_uid": c["parent_blind_id"], "parent_name": c["parent_name"],
               "name": c["name"], "instruction": c["instruction"],
               "why_separable": c.get("why_separable"),
               "upstream_parent": c.get("parent_upstream"),
               "mixed": False}  # the point of decomposition is that components are NOT mixed
        (out["A"] if c["kind"] == "candidate_real" else out["B"]).append(rec)

    for c in pos:
        out["B"].append({"source": "addendum4_position", "stratum": "position_in_container",
                         "name": c["name"], "instruction": c["instruction"],
                         "upstream_parent": c.get("upstream_parent"),
                         "mixed": bool(c.get("mixed"))})

    for p in PROBES[5]:
        out["probes"].append(p["pair"])
        out["A"].append({"source": "planted_probe", "stratum": "probe", "probe_pair": p["pair"],
                         "name": p["a"]["name"], "instruction": p["a"]["instruction"]})
        out["B"].append({"source": "planted_probe", "stratum": "probe", "probe_pair": p["pair"],
                         "name": p["b"]["name"], "instruction": p["b"]["instruction"],
                         "upstream_parent": p["b"].get("upstream_parent"),
                         "mixed": p["b"].get("mixed")})

    for track in ("A", "B"):
        for k, c in enumerate(out[track]):
            c["id"] = f"{track}{k + 1:02d}"
            c.setdefault("probe_pair", None)
    out["n_A"], out["n_B"] = len(out["A"]), len(out["B"])
    out["n_total"] = out["n_A"] + out["n_B"]
    (HERE / "round5_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("A", "B")}, indent=1))
    for t in ("A", "B"):
        print(f"--- {t} ---")
        for c in out[t]:
            print(f"  {c['id']} [{c['stratum']}] {c['name']}")


if __name__ == "__main__":
    main()
