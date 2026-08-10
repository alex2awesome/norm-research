#!/usr/bin/env python3
"""Round pipeline step 3: fold the blind audit (+ arbiter) into a final routing
and emit the round's scoreable criterion file.

Inputs
  round{r}_audit_verdicts.json   [{aid, verdict: quality_relevant|incidental,
                                  confidence, reason}]   (blind Sonnet-class auditor)
  round{r}_arbiter.json          optional [{aid, final: A|B, reason}] for disputes
Outputs
  round{r}_routing.json          per-criterion final track + misrouting bookkeeping
  round{r}_criteria.json         the 25 criteria in scoring order (cid/track/name/
                                 instruction) consumed by score_round_gemma.py

Routing rule (frozen): the AUDIT governs. A proposal whose audit verdict
contradicts its proposing track is re-routed per the audit; the misrouting rate is
reported.  Disputes escalate to the frontier arbiter, whose call is final.
Planted probes are scored like any other criterion and their audit verdict is the
instrument check (a probe routed to A, or its counterpart routed to B, is a FAIL).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round
    prov = json.loads((HERE / f"round{r}_proposals_provenance.json").read_text())
    key = json.loads((HERE / f"round{r}_audit_key.json").read_text())
    verdicts = {d["aid"]: d for d in
                json.loads((HERE / f"round{r}_audit_verdicts.json").read_text())}
    arb_path = HERE / f"round{r}_arbiter.json"
    arb = ({d["aid"]: d for d in json.loads(arb_path.read_text())}
           if arb_path.exists() else {})

    by_cid = {c["cid"]: c for c in prov["criteria"]}
    decisions, n_mis, n_disp = [], 0, 0
    for aid, cid in key.items():
        c = by_cid[cid]
        v = verdicts.get(aid)
        assert v is not None, f"audit missing for {aid} ({cid})"
        audit_track = "A" if v["verdict"] == "quality_relevant" else "B"
        proposed = c["proposed_track"]
        disputed = audit_track != proposed
        n_mis += int(disputed)
        final = audit_track
        arb_note = None
        if disputed and aid in arb:
            n_disp += 1
            final = arb[aid]["final"]
            arb_note = arb[aid].get("reason")
        decisions.append({
            "cid": cid, "aid": aid, "name": c["name"],
            "proposed_track": proposed, "audit_track": audit_track,
            "final_track": final, "disputed": disputed,
            "audit_confidence": v.get("confidence"), "audit_reason": v.get("reason"),
            "arbiter_reason": arb_note,
            "planted_probe": bool(c.get("PLANTED_PROBE")),
            "counterpart_cid": c["provenance"].get("counterpart_cid"),
            "proposer": c["provenance"]["proposer"],
            "family": c["provenance"]["family"]})

    probes = [d for d in decisions if d["planted_probe"]]
    probe_pass = all(d["final_track"] == "B" for d in probes)
    counterparts = {d["counterpart_cid"] for d in probes if d["counterpart_cid"]}
    cp_pass = all(d["final_track"] == "A" for d in decisions
                  if d["cid"] in counterparts)

    out = {"round": r, "n_criteria": len(decisions),
           "misrouting_rate": n_mis / max(1, len(decisions)),
           "n_disputes_arbitrated": n_disp,
           "planted_probes_routed_to_B": probe_pass,
           "probe_counterparts_routed_to_A": cp_pass,
           "PROBE_GATE_PASS": bool(probe_pass and cp_pass),
           "decisions": sorted(decisions, key=lambda d: d["cid"])}
    (HERE / f"round{r}_routing.json").write_text(json.dumps(out, indent=1))

    crits = [{"cid": d["cid"], "track": d["final_track"],
              "name": by_cid[d["cid"]]["name"],
              "instruction": by_cid[d["cid"]]["instruction"]}
             for d in out["decisions"]]
    (HERE / f"round{r}_criteria.json").write_text(json.dumps(crits, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "decisions"}, indent=1))
    print(f"wrote round{r}_criteria.json ({len(crits)} criteria: "
          f"{sum(c['track']=='A' for c in crits)} A / "
          f"{sum(c['track']=='B' for c in crits)} B)")


if __name__ == "__main__":
    main()
