#!/usr/bin/env python3
"""Assemble the FULL-GRID V+A+T master fused ledger.

Reads whatever arms have landed and emits (a) one fused per-cell JSON
`results/vat_fullgrid_<cell>.json` and (b) the master markdown table for
`notes/2026-08-10__vat_fullgrid.md`.  Re-runnable at any time: cells with missing
arms simply carry `null` in those columns, so the note can be checkpointed after
every landing without hand-editing numbers.

Sources, in order of precedence per quantity:
  VA_lin / VA_nl / VAT_lin / VAT_nl / T on E  <- results/vat_stack_<cell>.json
      (Direction-1 mirror; when a cell has several bank "arms" -- e.g. round0 vs
      round4-mined -- the TERMINAL arm is the ledger row and the earlier ones are
      kept in the per-cell JSON.  When a cell has several dense seeds, the
      mean-probability ensemble is the ledger row and per-seed rows are kept.)
  bank at full training strength ("fullfit@E") <- results/fullfit_at_E.json
  V3 <- results/v3_grid_<cell>.json  (written by the GPU chain agents)
  hand-entered caption rows <- results/vat_fullgrid_<cell>.json if it already
      exists and carries `MANUAL: true` (cap_finalist / cap_crowd / cw_community
      predate this script's schema).

No GPU, no network, read-mostly.  Never deletes a result file.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

# ledger row order = the strict list's field grouping
ROW_ORDER = [
    ("Peer review", ["peer_verdict", "peer_curation", "peer_revealed"]),
    ("Regulatory (N&C)", ["nc_responded", "nc_outcome", "nc_agree"]),
    ("Creative writing", ["cw_community"]),
    ("Humor", ["hashtagwars_verdict", "cap_finalist", "cap_crowd", "jokes_community"]),
    ("Math", ["mathse_accepted_verdict", "mathse_vote_score", "aops_curation"]),
    ("Software code", ["code_v3"]),
    ("Journalism/press", ["press_verdict"]),
]
Y_TYPE = {
    "peer_verdict": "verdict", "peer_curation": "curation", "peer_revealed": "revealed",
    "nc_responded": "verdict", "nc_outcome": "curation", "nc_agree": "agree",
    "cw_community": "community", "hashtagwars_verdict": "verdict",
    "cap_finalist": "curation", "cap_crowd": "community",
    "jokes_community": "community (replication)",
    "mathse_accepted_verdict": "verdict", "mathse_vote_score": "vote",
    "aops_curation": "curation", "code_v3": "verdict", "press_verdict": "verdict",
}


def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}".lstrip("0") if 0 < abs(x) < 1 else ("—" if x is None else f"{x:.{nd}f}")


def _boot(b):
    if not b:
        return "—"
    e, (lo, hi), p = b["estimate"], b["ci95"], b["p_gt_0"]
    s = lambda v: ("+" if v >= 0 else "−") + f"{abs(v):.4f}".lstrip("0")
    return f"{s(e)} [{s(lo)},{s(hi)}] {p:.2f}"


def pick_terminal_arm(d):
    """Terminal bank arm of a vat_stack JSON + a dict of the non-terminal ones."""
    if "ensemble" in d and d.get("ensemble"):
        return "ensemble", d["ensemble"], {k: v for k, v in d.get("per_dense_seed", {}).items()}
    arms = d.get("arms", {})
    if not arms:
        return None, None, {}
    # terminal = the highest-numbered round if rounds are present, else the sole arm
    keys = list(arms)
    rounds = [k for k in keys if k.startswith("round")]
    if rounds:
        term = max(rounds, key=lambda k: int("".join(c for c in k if c.isdigit()) or 0))
    else:
        term = keys[0]
    return term, arms[term], {k: v for k, v in arms.items() if k != term}


def load_cell(cell):
    row = {"cell": cell, "y_type": Y_TYPE.get(cell, "?")}
    p = RESULTS / f"vat_stack_{cell}.json"
    if p.exists():
        d = json.loads(p.read_text())
        term, a, others = pick_terminal_arm(d)
        row.update({
            "n_E": d.get("n_E"), "n_groups_E": d.get("n_groups_E"),
            "pos_rate_E": d.get("pos_rate_E"), "terminal_arm": term,
            "T": a.get("T_E"), "VA_lin": a.get("VA_lin_E"), "VA_nl": a.get("VA_nl_E_mean"),
            "VAT_lin": a.get("VAT_lin_E"), "VAT_nl": a.get("VAT_nl_E_mean"),
            "boot_VAT_minus_VA": a.get("boot_group_VAT_nl_minus_VA_nl"),
            "boot_VAT_minus_T": a.get("boot_group_VAT_nl_minus_T"),
            "other_arms": others, "vat_stack_source": str(p),
        })
        # newer vat_stack JSONs carry their own freshly-computed fullfit@E block
        f = d.get("fullfit_at_E") or {}
        v = (f.get("VA_nl_fullfit_at_E_seedmean_probs")
             or f.get("VA_nl_fullfit_at_E_mean_of_seed_aucs"))
        if v is not None:
            row["VA_nl_fullfit_at_E"] = v
        if d.get("va_reproduction_gate"):
            row["va_reproduction_gate"] = d["va_reproduction_gate"].get("pass")
        if d.get("POOLED_DO_NOT_QUOTE"):
            # pooled AUC is not a quotable readout for this cell (its splits contradict);
            # the within-group readout is the real one and must travel with the row
            row["POOLED_DO_NOT_QUOTE"] = True
            row["within_group"] = d.get("within_repo") or d.get("within_group")
    ff = RESULTS / "fullfit_at_E.json"
    if ff.exists():
        f = json.loads(ff.read_text()).get(cell, {})
        v = (f.get("va_nl_oof_mean3_at_E") or f.get("va_nl_oof_seed0_at_E")
             or f.get("pop_nl_fullpop"))
        if v is not None:                      # never clobber a vat_stack-supplied value
            row["VA_nl_fullfit_at_E"] = v
    d1b = RESULTS / "direction1b.json"
    if d1b.exists():
        b = json.loads(d1b.read_text()).get(cell) or {}
        if "combiner_bank_plus_T" in b:
            row["D1b"] = b["combiner_bank_plus_T"]
            row["D1b_detail"] = b
    v3p = RESULTS / f"v3_grid_{cell}.json"
    if v3p.exists():
        v = json.loads(v3p.read_text())
        row["V3"] = v.get("auc_E", v.get("V3"))
        row["V3_detail"] = v
    man = RESULTS / f"vat_fullgrid_{cell}.json"
    if man.exists():
        d = json.loads(man.read_text())
        if d.get("MANUAL"):
            row.update({k: v for k, v in d.items() if k not in ("cell",)})
    return row


def standing_rule(row):
    """design note SS11: max(VAT_nl, V3) must beat VA_nl, else AUTO-FABLE-AUDIT."""
    # D1b is the matched-training-strength fused arm; VAT_nl (E-refit) is fitted on ~20%
    # of the rows the fullfit bank sees, so where D1b exists it is the fair comparison
    fused = [x for x in (row.get("VAT_nl"), row.get("V3"), row.get("D1b")) if x is not None]
    banks = [x for x in (row.get("VA_nl"), row.get("VA_nl_fullfit_at_E")) if x is not None]
    if not fused or not banks:
        return "PENDING", None
    best, bank = max(fused), max(banks)
    return ("PASS" if best > bank else "AUTO-FABLE-AUDIT"), best - bank


def main():
    lines = ["| field | cell | y-type | n_E | grp | T | VA_lin | VA_nl | VA_nl@full | VAT_lin | VAT_nl | V3 | D1b | VAT−VA_nl [CI] P | VAT−T [CI] P | §11 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    flags, rows = [], {}
    for field, cells in ROW_ORDER:
        for cell in cells:
            r = load_cell(cell)
            rows[cell] = r
            verdict, margin = standing_rule(r)
            if verdict == "AUTO-FABLE-AUDIT":
                flags.append((cell, margin, r))
            lines.append("| {} | `{}`{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                field, cell, " ‖" if r.get("POOLED_DO_NOT_QUOTE") else "", r["y_type"],
                r.get("n_E") or "—", r.get("n_groups_E") or "—",
                _f(r.get("T")), _f(r.get("VA_lin")), _f(r.get("VA_nl")),
                _f(r.get("VA_nl_fullfit_at_E")), _f(r.get("VAT_lin")), _f(r.get("VAT_nl")),
                _f(r.get("V3")), _f(r.get("D1b")),
                _boot(r.get("boot_VAT_minus_VA")), _boot(r.get("boot_VAT_minus_T")),
                verdict))
            out = RESULTS / f"vat_fullgrid_{cell}.json"
            if not (out.exists() and json.loads(out.read_text()).get("MANUAL")):
                payload = dict(r)
                payload["standing_rule_sec11"] = verdict
                payload["standing_rule_margin_vs_best_bank"] = margin
                out.write_text(json.dumps(payload, indent=2, default=str))
    print("\n".join(lines))

    # ---- derived decomposition quantities (design note SS0) ----------------
    d = ["", "| cell | Δ_interact = VA_nl−VA_lin | Δ_beyond = T−VA_nl | fusion gain = max(fused)−max(bank) |",
         "|---|---:|---:|---:|"]
    def sg(v):
        return "—" if v is None else ("+" if v >= 0 else "−") + f"{abs(v):.4f}".lstrip("0")
    for _, cells in ROW_ORDER:
        for cell in cells:
            r = rows[cell]
            va_l, va_n, T = r.get("VA_lin"), r.get("VA_nl"), r.get("T")
            fused = [x for x in (r.get("VAT_nl"), r.get("V3"), r.get("D1b")) if x is not None]
            banks = [x for x in (va_n, r.get("VA_nl_fullfit_at_E")) if x is not None]
            d.append("| `{}` | {} | {} | {} |".format(
                cell,
                sg(None if (va_l is None or va_n is None) else va_n - va_l),
                sg(None if (T is None or va_n is None) else T - va_n),
                sg(None if (not fused or not banks) else max(fused) - max(banks))))
    print("\n".join(d))
    print("\nAUTO-FABLE-AUDIT flags:", [f[0] for f in flags] or "none")


if __name__ == "__main__":
    main()
