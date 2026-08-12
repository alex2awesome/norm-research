#!/usr/bin/env python3
"""F2 battery summariser: reads every results/f2_deconf_<cell>.json and emits the
two markdown tables for notes/2026-08-11__f2_deconfounded_fusion.md.

Table 1 = the five arms + the PRIMARY and SECONDARY increments.
Table 2 = the E-value-analog column (X, Y, RR, Z, verdict) + the §13 flags.

Read-only. Usage: python3 f2_summarise.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
NOTE = HERE.parents[2] / "notes" / "2026-08-11__f2_deconfounded_fusion.md"

ORDER = [
    ("Peer review", ["peer_verdict", "peer_curation", "peer_revealed"]),
    ("Regulatory (N&C)", ["nc_responded"]),
    ("Creative writing", ["cw_community"]),
    ("Humor", ["hashtagwars_verdict", "cap_finalist", "jokes_community"]),
    ("Math", ["mathse_accepted_verdict", "mathse_vote_score"]),
    ("Journalism/press", ["press_verdict"]),
]


def f4(x):
    return "—" if x is None else f"{x:.4f}".lstrip("0")


def sg(x):
    return "—" if x is None else ("+" if x >= 0 else "−") + f"{abs(x):.4f}".lstrip("0")


def boot(b):
    if not b:
        return "—"
    return f"{sg(b['estimate'])} [{sg(b['ci95'][0])},{sg(b['ci95'][1])}] {b['p_gt_0']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    t1 = ["| field | cell | n_E | (a) bank_enr | (b) NUIS | (c) enr+NUIS | (d) +T | (e) +T₀ | PRIMARY (d)−(c) [CI] P | WY band | SECONDARY (e)−(c) P | §11 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
    t2 = ["| cell | Δ (d)−(c) | X | Y | RR=(X−.5)/(Y−.5) | X/Y | M̂ (strict?) | Z | verdict |",
          "|---|---:|---:|---:|---:|---:|---|---:|---|"]
    t3 = ["| cell | bank E-refit (a) | bank full-strength on E | gap | >.02 | (c*) | (d*) | COMPANION (d*)−(c*) [CI] P | E-refit primary (contrast) | matched X | matched RR | matched verdict |",
          "|---|---:|---:|---:|:-:|---:|---:|---|---:|---:|---:|---|"]
    missing, flags = [], []
    for field, cells in ORDER:
        for c in cells:
            p = RESULTS / f"f2_deconf_{c}.json"
            if not p.exists():
                missing.append(c)
                continue
            d = json.loads(p.read_text())
            a = d["arms"]
            wy = d.get("westfall_yarkoni_reliability_band", {}).get("band", {})
            wyt = ("—" if not wy else f"[{sg(wy['lo'])},{sg(wy['hi'])}]")
            t1.append("| {} | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                field, c, d["n_E"], f4(a["a_VA_enr_nl"]), f4(a["b_NUIS_nl"]),
                f4(a["c_VA_enr_plus_NUIS_nl"]), f4(a["d_VAT_dec_trained_nl"]),
                f4(a["e_VAT_dec_untrained_nl"]),
                boot(d["PRIMARY_stacked_increment_d_minus_c"]), wyt,
                f"{sg(d['SECONDARY_untrained_increment_e_minus_c']['estimate'])} "
                f"{d['SECONDARY_untrained_increment_e_minus_c']['p_gt_0']:.2f}",
                d["fused_must_beat_bank"]["verdict"]))
            if d.get("spurious_alone_gt_065"):
                flags.append(f"`{c}` spurious-alone {d['spurious_alone_b']:.4f} > .65")

            e = d.get("evalue_analog")
            if not e:
                t2.append(f"| `{c}` | {sg(d['PRIMARY_stacked_increment_d_minus_c']['estimate'])} "
                          "| _pending_ | | | | | | |")
                continue
            m = e.get("missing_mass", {})
            mh = ("—" if not m.get("available") else
                  f"{m['M_hat']:.2f} ({'strict' if m.get('strict_marker_present') else 'τ-era'})")
            xs = (e.get("X_statement") or f4(e.get("X")))
            t2.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                c, sg(e.get("primary_delta_3seed")), xs, f4(e.get("Y_strongest_found_channel")),
                "—" if e.get("robustness_ratio_excess") is None else f"{e['robustness_ratio_excess']:.2f}",
                "—" if e.get("X_over_Y") is None else f"{e['X_over_Y']:.2f}",
                mh, f4(e.get("Z")), e.get("verdict", "—")))
            ms = d.get("matched_strength_companion")
            em = d.get("evalue_analog_matched") or {}
            if ms and ms.get("applicable"):
                s1 = ms["stage1"]["VA_enr_full_nl_on_E_seedmean_probs"]
                am = ms["arms_matched"]
                t3.append("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    c, f4(a["a_VA_enr_nl"]), f4(s1),
                    sg(ms["enriched_bank_gap_fullfit_minus_Erefit"]),
                    "**Y**" if ms["gap_exceeds_trigger"] else "n",
                    f4(am["c_star_bankfull_plus_NUIS_nl"]), f4(am["d_star_plus_T_nl"]),
                    boot(ms["COMPANION_increment_dstar_minus_cstar"]),
                    sg(ms["primary_Erefit_increment_for_contrast"]),
                    (em.get("X_statement") or f4(em.get("X"))) if em else "—",
                    "—" if not em or em.get("robustness_ratio_excess") is None
                        else f"{em['robustness_ratio_excess']:.2f}",
                    em.get("verdict", "—") if em else "—"))
            elif ms:
                t3.append(f"| `{c}` | {f4(a['a_VA_enr_nl'])} | — | — | — | — | — | "
                          "_n/a — E is the whole population; companion identical to primary_ "
                          f"| {sg(d['PRIMARY_stacked_increment_d_minus_c']['estimate'])} | — | — | — |")
            if e.get("Z_FLAG"):
                flags.append(f"`{c}` {e['Z_FLAG'][:60]}")
            if e.get("NON_MONOTONE"):
                flags.append(f"`{c}` NON_MONOTONE sweep")

    out = ("### Arms and increments\n\n" + "\n".join(t1) +
           "\n\n### Matched-strength companion (D1b-style two-stage)\n\n" + "\n".join(t3) +
           "\n\nWhere `>.02` is **Y**, the E-refit primary is a matched-footing readout and is "
           "**not** comparable to any full-strength bank comparison (including a closure "
           "campaign's same-rows verdict); the COMPANION is the quotable number there.\n"
           "\n### E-value analog\n\n" + "\n".join(t2) +
           "\n\n`X` reported as `> AUC(T)` means the sweep never crossed zero even at a "
           "channel as strong as T itself: **not absorbable by any single channel weaker "
           "than T**.\n")
    if flags:
        out += "\n**§13 flags:** " + "; ".join(sorted(set(flags))) + "\n"
    if missing:
        out += "\n**Cells not yet landed:** " + ", ".join(missing) + "\n"
    print(out)
    if args.write and NOTE.exists():
        t = NOTE.read_text()
        if "<!-- RESULTS -->" in t:
            t = t.replace("<!-- RESULTS -->", out)
        else:
            i = t.index("## Results")
            j = t.index("## Artifacts")
            t = t[:i] + "## Results\n\n" + out + "\n" + t[j:]
        NOTE.write_text(t)
        print("wrote", NOTE)


if __name__ == "__main__":
    main()
