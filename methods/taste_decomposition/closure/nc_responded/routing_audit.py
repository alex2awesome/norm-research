#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) for the N&C RESPONDED closure campaign.

  build     pool the round's 25 post-phrasing criteria, strip ALL provenance (track,
            source, stratum, upstream tags, rationale), stable-sha256 order, emit the
            file the FRESH independent auditor sees.
  finalize  merge the auditor's verdicts with the frontier arbiter's rulings on
            disputes, apply the sign-contradiction re-audit trigger, and write the
            final A / B routing.

The proposer rationales are withheld from the auditor because Track-B rationales name
the mindset outright ("spurious", "not merit").  The auditor sees name + instruction
only, exactly as a judge would.

Usage:
  python routing_audit.py build --round 1
  python routing_audit.py finalize --round 1 [--rule BLIND_ID=bank|nuisance="reason" ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

AUDIT_INSTRUCTION = """You are an independent auditor. Below are candidate criteria for scoring
PUBLIC COMMENTS submitted to United States federal agencies on proposed rules. Each
would be scored 0-10 by a judge reading only one comment's text.

Classify EACH criterion as exactly one of:

  "quality_relevant" -- it measures something a careful regulatory analyst would call
      substantive merit of the comment as regulatory input: evidence, analysis,
      specificity about the rule, feasibility, actionable alternatives, first-hand
      knowledge, engagement with the agency's reasoning.

  "incidental" -- it is predictive-but-not-merit: a surface or shape feature, a
      formatting or length proxy, boilerplate or template similarity, a topic or
      agency-style marker, a submitter-identity or professional-drafting fingerprint,
      a tone or register marker, or the bare presence of a word or phrase.

DECISION AIDS.
  * If the criterion could be scored high by a comment that says nothing substantive,
    it is incidental.
  * If the instruction tells the judge to score the SHAPE of the document rather than
    what the document establishes, it is incidental -- even when the shape usually
    co-occurs with merit. Say so in your justification if you see this.
  * "Related to quality" is not enough. The question is what the JUDGE would be
    scoring when following this exact instruction.

Emit exactly one JSON object and nothing else:

{"verdicts": [
  {"id": "P01", "label": "quality_relevant" or "incidental",
   "confidence": "high"/"medium"/"low",
   "justification": "<one or two sentences>"},
  ... one entry for EVERY criterion ...
]}
"""


def cmd_build(a):
    r = a.round
    fin = json.loads((HERE / f"round{r}_criteria_final.json").read_text())
    pool = []
    for track in ("A", "B"):
        for c in fin[track]:
            pool.append({"src_id": c["id"], "track": track, "name": c["name"],
                         "instruction": c["instruction"], "source": c.get("source"),
                         "stratum": c.get("stratum"), "probe_pair": c.get("probe_pair")})
    salt = f"nc-closure-round{r}-blind"
    pool.sort(key=lambda c: hashlib.sha256(f"{salt}|{c['name']}".encode()).hexdigest())
    for k, c in enumerate(pool):
        c["blind_id"] = f"P{k + 1:02d}"

    (HERE / f"round{r}_proposals_blinded.json").write_text(json.dumps({
        "n": len(pool),
        "scale": "0-10 integer, NA if the comment gives no evidence bearing on the criterion",
        "task": AUDIT_INSTRUCTION,
        "criteria": [{"id": c["blind_id"], "name": c["name"], "instruction": c["instruction"]}
                     for c in pool],
    }, indent=1))
    (HERE / f"round{r}_proposals_provenance.json").write_text(json.dumps(
        [{"blind_id": c["blind_id"], "src_id": c["src_id"], "track": c["track"],
          "name": c["name"], "source": c["source"], "stratum": c["stratum"],
          "probe_pair": c["probe_pair"]} for c in pool], indent=1))

    body = AUDIT_INSTRUCTION + "\n\n" + "\n\n".join(
        f"--- CRITERION {c['blind_id']} ---\nNAME: {c['name']}\nINSTRUCTION: {c['instruction']}"
        for c in pool) + "\n"
    (HERE / f"round{r}_audit_prompt.txt").write_text(body)
    print(f"r{r}: {len(pool)} blinded criteria -> round{r}_audit_prompt.txt")


def cmd_finalize(a):
    r = a.round
    rulings = {}
    for spec in a.rule or []:
        bid, rest = spec.split("=", 1)
        ruling, _, reason = rest.partition(":")
        rulings[bid] = {"ruling": ruling, "reason": reason}

    prov = {p["blind_id"]: p for p in json.loads((HERE / f"round{r}_proposals_provenance.json").read_text())}
    aud = {v["id"]: v for v in json.loads((HERE / f"round{r}_routing_audit.json").read_text())["verdicts"]}

    final = []
    for bid in sorted(prov):
        p, au = prov[bid], aud[bid]
        route = "A" if au["label"] == "quality_relevant" else "B"
        rec = {"blind_id": bid, "src_id": p["src_id"], "name": p["name"],
               "proposed_track": p["track"], "source": p["source"], "stratum": p["stratum"],
               "probe_pair": p["probe_pair"],
               "audit_label": au["label"], "audit_route": route,
               "audit_confidence": au.get("confidence"),
               "audit_justification": au.get("justification"),
               "misrouted": route != p["track"], "final_route": route}
        if bid in rulings:
            rl = rulings[bid]
            rec["arbiter"] = {
                "model": "Opus (frontier arbiter; criterion text + auditor verdict + proposer rationale)",
                "ruling": rl["ruling"], "reason": rl["reason"],
                "upheld": "auditor" if (rl["ruling"] == "nuisance") == (route == "B") else "proposer"}
            rec["final_route"] = "A" if rl["ruling"] == "bank" else "B"
        final.append(rec)

    probes = {}
    for rec in final:
        if rec["probe_pair"]:
            probes.setdefault(rec["probe_pair"], []).append(
                {"proposed": rec["proposed_track"], "final": rec["final_route"], "name": rec["name"]})
    probe_report = {k: {"members": v,
                        "separated": len({m["final"] for m in v}) == 2,
                        "correct": all(m["final"] == m["proposed"] for m in v)}
                    for k, v in probes.items()}

    n_mis = sum(x["misrouted"] for x in final)
    fin = json.loads((HERE / f"round{r}_criteria_final.json").read_text())
    bysrc = {c["id"]: c for t in ("A", "B") for c in fin[t]}
    out = {
        "round": r,
        "protocol": ("prereg step 4: 25 post-phrasing criteria pooled, provenance stripped, "
                     "stable-sha256 order, FRESH independent Sonnet-class auditor sees only "
                     "the blinded file; disputes adjudicated by a frontier (Opus) arbiter; "
                     "2 planted probe pairs per round"),
        "self_audit": False,
        "n_proposals": len(final),
        "n_proposed_A": sum(x["proposed_track"] == "A" for x in final),
        "n_proposed_B": sum(x["proposed_track"] == "B" for x in final),
        "n_misrouted": n_mis, "misrouting_rate": n_mis / len(final),
        "n_final_A": sum(x["final_route"] == "A" for x in final),
        "n_final_B": sum(x["final_route"] == "B" for x in final),
        "planted_probes": probe_report,
        "all_probes_separated": all(v["separated"] for v in probe_report.values()),
        "final": final,
        # score-block selectors used by readout.load_round_scores and
        # track_b_discount.load_b_blocks.  `id` MUST be the BLIND id, because that is
        # what score_round_gemma.py writes into roundN_scores.npz `crit_ids` and what
        # keys roundN_score_report.json's collapse gate.  src_id is carried alongside.
        "A": [{"id": x["blind_id"], "src_id": x["src_id"], "name": x["name"]}
              for x in final if x["final_route"] == "A"],
        "B": [{"id": x["blind_id"], "src_id": x["src_id"], "name": x["name"],
               "upstream_parent": bysrc.get(x["src_id"], {}).get("upstream_parent"),
               "mixed": bysrc.get(x["src_id"], {}).get("mixed")}
              for x in final if x["final_route"] == "B"],
    }
    (HERE / f"round{r}_routing_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("final", "A", "B")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--round", type=int, required=True)
    f = sub.add_parser("finalize"); f.add_argument("--round", type=int, required=True)
    f.add_argument("--rule", action="append",
                   help="BLIND_ID=bank:reason or BLIND_ID=nuisance:reason")
    a = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)
