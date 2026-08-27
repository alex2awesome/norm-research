#!/usr/bin/env python3
"""Emit `<campaign>/certA_strict.json` — the Track-A missing-mass certificate of record —
by pairing each campaign's tau-era species file with the strict two-judge merge written by
`species_merge.py apply --track A`.

One certificate per campaign. The terminal round is the certificate of record; earlier
rounds are carried as the trajectory (tau-era only — only the terminal round was re-judged).

Usage:  python write_certA.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# campaign dir, cell tag, terminal round, all rounds present (trajectory)
CELLS = [
    ("peer_curation_ext", "peer_curation", 5, [1, 2, 3, 4, 5]),
    ("peer_revealed", "peer_revealed", 5, [1, 2, 3, 4, 5]),
    ("maps_hw_si", "hashtagwars_verdict", 4, [1, 2, 3, 4]),
    ("press_verdict", "press_verdict", 2, [1, 2]),
    ("mathse_vote", "mathse_vote", 3, [1, 3]),
    ("mathse_accepted", "mathse_accepted", 2, [1, 2]),
]

JUDGE_CAVEAT = (
    "Both legs (gpt-5.6-sol, gpt-5.6-luna) are one family, exactly as the Sonnet-era merges "
    "used claude-sonnet-5 on both legs. 'Both judges must say SAME' is therefore a weaker "
    "independence claim than a cross-family pair would give; record this wherever the "
    "figure is quoted."
)


def gt_block(gt):
    j = gt.get("jackknife_LOPO_missing_mass") or {}
    return {"N_proposals": gt.get("N_proposals"), "S_obs": gt.get("S_obs"),
            "f1": gt.get("f1"), "f2": gt.get("f2"),
            "missing_mass": gt.get("good_turing_missing_mass"),
            "cross_proposer_recapture": gt.get("cross_proposer_recapture"),
            "LOO_proposer_jackknife": {"min": j.get("min"), "max": j.get("max"),
                                       "mean": j.get("mean"), "values": j.get("values")}}


def main():
    summary = []
    for d, cell, term, rounds in CELLS:
        cdir = HERE / d
        strict_p = cdir / f"{cell}_r{term}_species_strictA.json"
        tau_p = cdir / f"{cell}_r{term}_species.json"
        if not strict_p.exists():
            print(f"SKIP {cell}: {strict_p.name} not written yet")
            continue
        strict = json.loads(strict_p.read_text())
        tau = json.loads(tau_p.read_text())
        gt_s = strict["tracks"]["A"]["good_turing"]
        gt_t = tau["tracks"]["A"]["good_turing"]
        bm = strict["blind_merge"]["A"]

        traj = []
        for r in rounds:
            p = cdir / f"{cell}_r{r}_species.json"
            if p.exists():
                g = json.loads(p.read_text())["tracks"]["A"]["good_turing"]
                traj.append({"round": r, "tau_missing_mass": g["good_turing_missing_mass"],
                             "S_obs": g["S_obs"], "f1": g["f1"],
                             "terminal": r == term})

        cert = {
            "cell": cell,
            "campaign_dir": d,
            "track": "A",
            "certificate_of_record_round": term,
            "status": "STRICT TWO-JUDGE BLIND PAIRWISE MERGE (Track-A backfill 2026-08-11)",
            "old_tau_only": gt_block(gt_t),
            "strict_merged": gt_block(gt_s),
            "delta_missing_mass_strict_minus_tau": round(
                gt_s["good_turing_missing_mass"] - gt_t["good_turing_missing_mass"], 4),
            "merge": {"judges": bm["judges"], "n_pairs_adjudicated": bm["n_pairs_adjudicated"],
                      "n_merge_edges_strict": bm["n_merge_edges_strict"],
                      "anchor_all_pass": bm["anchor_all_pass"],
                      "anchor_battery": bm["anchor_battery"],
                      "shortlist_rule": bm["rule"]},
            "trajectory_tau_only": traj,
            "framing": (
                "A CORRECTION with per-cell direction that is not predictable in advance, NOT a "
                "deflation. species_merge.py restarts from the raw pool as singletons and only "
                "adds an edge where a CROSS-PROPOSER pair is called SAME by BOTH judges, so it "
                "systematically undoes the within-proposer merges tau had made. Whether a cell's "
                "mass rises or falls depends on how much of its tau clustering was "
                "within-proposer, which is a per-cell empirical fact."
            ),
            "judge_family_caveat": JUDGE_CAVEAT,
            "cross_instrument_calibration": (
                "aops_curation r1 Track A was re-judged on this same sol+luna pair as the "
                "anchor. Sonnet-strict .5583 vs sol+luna-strict .4917 (delta -.0667) on an "
                "identical packet. The Sonnet figure sits inside the sol+luna LOO band "
                "[.4667,.5905], but the judge-family delta EXCEEDS the Sonnet tau->strict "
                "correction (+.0583) it would have to be small against. These certificates are "
                "therefore a NEW-INSTRUMENT RE-SURVEY: quote them as sol+luna strict masses, "
                "never as cell-by-cell deltas against Sonnet-era numbers."
            ),
            "artifacts": {
                "strict_species": strict_p.name,
                "tau_species_untouched": tau_p.name,
                "packet": f"{cell}_r{term}_bmergeA_packet.json",
                "judge_sol": f"{cell}_r{term}_bmergeA_judge_sol.json",
                "judge_luna": f"{cell}_r{term}_bmergeA_judge_luna.json",
            },
        }
        (cdir / "certA_strict.json").write_text(json.dumps(cert, indent=1))
        summary.append((cell, term, gt_t["good_turing_missing_mass"],
                        gt_s["good_turing_missing_mass"], gt_t["S_obs"], gt_s["S_obs"],
                        gt_t["f1"], gt_s["f1"], bm["n_merge_edges_strict"],
                        bm["anchor_all_pass"]))
        print(f"wrote {d}/certA_strict.json")

    print(f"\n{'cell':26s} {'rd':>3s} {'tau M':>7s} {'strict M':>9s} {'delta':>7s} "
          f"{'S tau->str':>12s} {'f1 tau->str':>12s} {'edges':>6s} anchors")
    for c, t, mt, ms, st, ss, f1t, f1s, e, ap in summary:
        print(f"{c:26s} {t:3d} {mt:7.4f} {ms:9.4f} {ms - mt:+7.4f} "
              f"{str(st) + '->' + str(ss):>12s} {str(f1t) + '->' + str(f1s):>12s} {e:6d} "
              f"{'PASS' if ap else 'FAIL'}")


if __name__ == "__main__":
    main()
