#!/usr/bin/env python3
"""Merge the round-2 blind audit with the frontier arbiter's ruling on disputes.

Usage:  python finalize_routing_r2.py <blind_id> <bank|nuisance> "<reason>"
(one triple per disputed criterion; pass none if there were no disputes).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    rulings = {}
    for i in range(0, len(args), 3):
        rulings[args[i]] = {"ruling": args[i + 1], "reason": args[i + 2]}

    prov = {p["blind_id"]: p for p in json.loads((HERE / "round4_proposals_provenance.json").read_text())}
    aud = {v["id"]: v for v in json.loads((HERE / "round4_routing_audit.json").read_text())["verdicts"]}

    final = []
    for bid in sorted(prov):
        p, a = prov[bid], aud[bid]
        audit_route = "A" if a["label"] == "quality_relevant" else "B"
        rec = {
            "blind_id": bid, "src_id": p["src_id"], "name": p["name"],
            "proposed_track": p["track"],
            "audit_label": a["label"], "audit_route": audit_route,
            "audit_confidence": a["confidence"], "audit_justification": a["justification"],
            "misrouted": audit_route != p["track"],
            "final_route": audit_route,
        }
        if bid in rulings:
            r = rulings[bid]
            rec["arbiter"] = {
                "model": "Opus (frontier arbiter; criterion text + auditor verdict + proposer rationale)",
                "ruling": r["ruling"],
                "reason": r["reason"],
                "upheld": "auditor" if (r["ruling"] == "nuisance") == (audit_route == "B") else "proposer",
            }
            rec["final_route"] = "A" if r["ruling"] == "bank" else "B"
        final.append(rec)

    n_mis = sum(r["misrouted"] for r in final)
    out = {
        "round": 4,
        "protocol": ("prereg step 4: proposals pooled, provenance stripped, stable-sha256 order, "
                     "FRESH independent Sonnet auditor sees ONLY round4_proposals_blinded.json; "
                     "disputes with the proposing track adjudicated by a frontier (Opus) arbiter"),
        "self_audit": False,
        "n_proposals": len(final),
        "n_proposed_A": sum(r["proposed_track"] == "A" for r in final),
        "n_proposed_B": sum(r["proposed_track"] == "B" for r in final),
        "n_misrouted": n_mis,
        "misrouting_rate": n_mis / len(final),
        "n_final_A": sum(r["final_route"] == "A" for r in final),
        "n_final_B": sum(r["final_route"] == "B" for r in final),
        "final": final,
    }
    (HERE / "round4_routing_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "final"}, indent=2))


if __name__ == "__main__":
    main()
