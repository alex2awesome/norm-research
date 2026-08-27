"""CW HELD-OUT PROMOTION BATCH certification (pre-registered one-batch test-side eval).

Adapted from methods/metric_seam/battery/cert_agentic.py (frozen, audited pattern) --
that script's exact machinery is reused: battery_common.load_ctx for split/fields/ops,
eval_hybrids_task.paired_boot (B=2000), certificates.spearman, the frozen-flavor gate
(rho_test >= max(rho_base+0.10, 0.60) against the ORIGINAL v0/v1/v2 codegen baseline),
P(cand>h0) on test, and train-test gap with TRAIN RHO RECOMPUTED here (not taken from
the queue's own self-reported numbers).

DEVIATIONS FROM cert_agentic.py (kept minimal, listed in full):
  1. CANDS is built from the census promotion queue
     (outputs/metric_seam_pilot/battery/effort_ladder/census/PROMOTION_QUEUE.json,
     22 entries, all task=creative_writing) instead of a hardcoded agentic-fleet list;
     candidate.py paths resolve under battery/effort_ladder/census/creative_writing__<aid>/
     (census cells), not hybrids/programs_agentic/.
  2. h0 path is UNCHANGED machinery: ctx["hyb"] / f"{aid}_h0.py" = programs_cw/<aid>_h0.py,
     the same lookup cert_agentic.py already used for its 4 creative_writing rows.
  3. NEW -- field-completeness check. Two candidates (a207 cell 10: conflict_source,
     stakes_proof; a189 cell 12: form_status) declare NEW LLM_FIELDS that were extracted
     for TRAIN items only when the cell was built. TEST-side values for exactly these 3
     fields were extracted in a prior, separate step (build_heldout_test_fields_cw.py,
     glm-4.7 via the zai_anthropic subscription API, reusing each cell's own prompt
     template byte-for-byte) and appended to the shared field_results.jsonl cache (backed
     up first, dedup-verified). This script independently RE-CHECKS coverage at run time
     (every candidate's LLM_FIELDS x test-split presence, not just the 2 known cells) and,
     for any candidate still missing >=10% of a declared field's test coverage, evaluates
     it BOTH with the field as extracted (partial) AND with the field forced fully absent
     (fully degraded), and marks it FIELD-INCOMPLETE instead of reporting a single number.
  4. NEW -- test-judge-coverage flag. Two aspects (a216, a81) have <90% TEST judge-score
     coverage (pre-existing gap in the shared judge store, unrelated to this batch, not
     backfilled here -- out of scope / needs sign-off). These are computed (n>=20 still
     clears the correlation floor) but flagged AMBIGUOUS_LOW_JUDGE_COVERAGE rather than
     given a clean G1/promotion verdict, per the pre-registration's explicit stop-and-report
     condition for judge coverage <90% on test.
  5. NEW -- derived "G1_verdict" (PASS/FAIL/NA) computed from the SAME gate bootstrap
     cert_agentic.py already runs (P_gate_cand: candidate vs the frozen v0/v1/v2 codegen
     baseline flavor, gate_floor=0.60, margin=0.10, B=2000) -- PASS iff P_gate_cand>=0.5,
     matching eval_hybrids_task.py's own n_cert convention. The bootstrap arithmetic itself
     is unchanged; only this PASS/FAIL label is new.
  6. NEW -- promotion verdict. PROMOTED iff P(cand>h0) on test >= 0.90 AND delta_test
     (test_rho_cand - test_rho_h0) >= 0; both criteria also reported separately (a
     candidate can clear one and not the other). WASH / REGRESSED otherwise, based on
     delta_test sign, unless FIELD-INCOMPLETE or AMBIGUOUS_LOW_JUDGE_COVERAGE applies.
  7. NEW -- the queue entry's own disclosed caveats (contract, self_adversary, note,
     train_rho_h0/cand as originally self-reported, rel_gain) are carried through into the
     report VERBATIM for audit traceability -- cert_agentic.py has no queue to carry
     through.
  8. Output path: outputs/metric_seam_pilot/battery/effort_ladder/census/
     cw_heldout_report.json (cert_agentic.py writes outputs/metric_seam_pilot/battery/
     agentic_cert.json). NEW -- compact summary print table.
  9. NEW -- no candidate.py / h0 / harness file / the queue file is modified anywhere in
     this script (read-only over all of them); test judge scores are read ONLY inside this
     script's own evaluation calls, never written back to any census cell dir.

All other machinery -- load_ctx, run_prog, spearman, paired_boot, the gate_floor/margin/B
constants, frozen-baseline-flavor selection off code_scores.json (best TRAIN rho among
v0_keyword/v1_structure/v2_holistic), and train rho recomputed (never read from the
queue's self-reported numbers) -- is reused byte-for-byte from cert_agentic.py /
eval_hybrids_task.py.

Usage: python3 cert_census_cw.py
-> outputs/metric_seam_pilot/battery/effort_ladder/census/cw_heldout_report.json
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE, ROOT  # noqa: E402
from eval_hybrids_task import paired_boot, FLAVORS  # noqa: E402
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman  # noqa: E402

CENSUS = ROOT / "outputs/metric_seam_pilot/battery/effort_ladder/census"
EFFORT_LADDER = CENSUS.parent
TASK = "creative_writing"
LOW_JUDGE_COVERAGE_AIDS = {"a216", "a81"}  # <90% test judge coverage, flagged not fixed


def rho_on(sel, col, judge):
    s = [d for d in sel if col.get(d) is not None]
    if len(s) < 20:
        return float("nan"), 0
    return spearman([col[d] for d in s], [judge[d] for d in s]), len(s)


def field_presence(f_orig, aid, field, ids):
    fmap = f_orig.get(aid, {})
    present = sum(1 for d in ids if field in fmap.get(d, {}))
    return present, len(ids)


def strip_fields(fmap, fields, ids):
    """Return a copy of fmap with `fields` forced absent for `ids` (fully-degraded run)."""
    out = {}
    for d, v in fmap.items():
        if d in ids:
            out[d] = {k: val for k, val in v.items() if k not in fields}
        else:
            out[d] = v
    return out


def main():
    queue = json.load(open(CENSUS / "PROMOTION_QUEUE.json"))["queue"]
    assert len(queue) == 22, f"expected 22 queue entries, got {len(queue)}"
    for e in queue:
        assert e["task"] == TASK, f"unexpected task {e['task']} for {e['aspect']}"

    ctx = load_ctx(TASK)
    judge_all = ctx["judge"]
    items = ctx["items"]
    ops = ctx["ops"]
    f_orig = ctx["f_orig"]
    train = sorted(ctx["train"])
    test = sorted(ctx["test"])
    test_set, train_set = set(test), set(train)

    cs_path = ctx["outdir"] / "code_scores.json"
    code = json.load(open(cs_path)) if cs_path.exists() else {}

    report = {}
    rows_for_table = []

    for e in queue:
        aid = e["aspect"]
        cell = e["cell"]
        judge = judge_all.get(aid, {})
        if not judge:
            report[aid] = {"cell": cell, "error": "no judge channel for this aspect"}
            print(f"{aid}: SKIP no judge channel")
            continue

        cand_path = EFFORT_LADDER / e["candidate"]
        if not cand_path.exists():
            report[aid] = {"cell": cell, "error": f"missing candidate file: {cand_path}"}
            print(f"{aid}: STOP missing candidate file {cand_path}")
            continue
        h0_path = ctx["hyb"] / f"{aid}_h0.py"
        if not h0_path.exists():
            report[aid] = {"cell": cell, "error": f"missing h0 file: {h0_path}"}
            print(f"{aid}: STOP missing h0 file {h0_path}")
            continue

        cand = load_mod(cand_path)
        h0 = load_mod(h0_path)
        fmap = f_orig.get(aid, {})

        # --- field-completeness check (deviation 3) ---
        llm_fields = list(getattr(cand, "LLM_FIELDS", {}) or {})
        field_status = {}
        incomplete_fields = []
        for f in llm_fields:
            tr_p, tr_n = field_presence(f_orig, aid, f, train)
            te_p, te_n = field_presence(f_orig, aid, f, test)
            field_status[f] = {"train_coverage": f"{tr_p}/{tr_n}", "test_coverage": f"{te_p}/{te_n}"}
            if te_n and te_p / te_n < 0.90:
                incomplete_fields.append(f)
        field_incomplete = bool(incomplete_fields)

        col_c = run_prog(cand.score, items, fmap, ops)
        col_0 = run_prog(h0.score, items, fmap, ops)

        degraded = None
        if field_incomplete:
            fmap_degraded = strip_fields(fmap, incomplete_fields, test_set)
            col_c_degraded = run_prog(cand.score, items, fmap_degraded, ops)
            te_sel_d = [d for d in test if d in judge]
            r_te_c_d, n_te_d = rho_on(te_sel_d, col_c_degraded, judge)
            degraded = {"fields_forced_absent": incomplete_fields,
                        "test_rho_cand_degraded": round(r_te_c_d, 4) if r_te_c_d == r_te_c_d else None,
                        "n_test_degraded": n_te_d}

        # --- frozen codegen baseline (best TRAIN flavor), same recipe as cert_agentic.py ---
        best_fl, best_tr, base_col = None, -2, None
        for fl in FLAVORS:
            col = code.get(f"{aid}_{fl}") or {}
            sel = [d for d in train if d in judge and col.get(d) is not None]
            if len(sel) < 30:
                continue
            r = spearman([col[d] for d in sel], [judge[d] for d in sel])
            if r == r and r > best_tr:
                best_fl, best_tr, base_col = fl, r, col

        tr_sel = [d for d in train if d in judge]
        te_sel = [d for d in test if d in judge]
        r_tr_c, _ = rho_on(tr_sel, col_c, judge)
        r_tr_0, _ = rho_on(tr_sel, col_0, judge)
        r_te_c, n_te = rho_on(te_sel, col_c, judge)
        r_te_0, _ = rho_on(te_sel, col_0, judge)

        row = {"cell": cell,
               "n_test": n_te,
               "train_rho_recomputed": {"cand": round(r_tr_c, 4), "h0": round(r_tr_0, 4)},
               "test_rho": {"cand": round(r_te_c, 4) if r_te_c == r_te_c else None,
                            "h0": round(r_te_0, 4) if r_te_0 == r_te_0 else None},
               "gap_train_minus_test": {"cand": round(r_tr_c - r_te_c, 4) if r_tr_c == r_tr_c and r_te_c == r_te_c else None,
                                        "h0": round(r_tr_0 - r_te_0, 4) if r_tr_0 == r_tr_0 and r_te_0 == r_te_0 else None},
               "delta_test": round(r_te_c - r_te_0, 4) if r_te_c == r_te_c and r_te_0 == r_te_0 else None}

        sel_b = [d for d in te_sel if col_c.get(d) is not None and col_0.get(d) is not None]
        _, p_beat0, used = paired_boot(sel_b, col_c, col_0, judge, gate_floor=-2, margin=-2)
        row["P_cand_gt_h0"] = p_beat0

        if base_col is not None:
            sel_g = [d for d in sel_b if base_col.get(d) is not None]
            pg_c, _, _ = paired_boot(sel_g, col_c, base_col, judge)
            pg_0, _, _ = paired_boot(sel_g, col_0, base_col, judge)
            r_te_b, _ = rho_on(sel_g, base_col, judge)
            row["gate"] = {"baseline_flavor": best_fl, "rho_test_baseline": round(r_te_b, 3),
                           "P_gate_cand": pg_c, "P_gate_h0": pg_0}
            row["G1_verdict"] = "PASS" if (pg_c is not None and pg_c >= 0.5) else "FAIL"
        else:
            row["gate"] = None
            row["G1_verdict"] = "NA"

        # --- promotion verdict (deviation 6) ---
        low_judge_cov = aid in LOW_JUDGE_COVERAGE_AIDS
        if field_incomplete:
            verdict = "FIELD-INCOMPLETE"
        elif low_judge_cov:
            verdict = "AMBIGUOUS_LOW_JUDGE_COVERAGE"
        elif p_beat0 is not None and p_beat0 >= 0.90 and row["delta_test"] is not None and row["delta_test"] >= 0:
            verdict = "PROMOTED"
        elif row["delta_test"] is not None and row["delta_test"] < 0:
            verdict = "REGRESSED"
        else:
            verdict = "WASH"
        row["verdict"] = verdict
        row["criteria"] = {"P_cand_gt_h0_ge_0.90": (p_beat0 is not None and p_beat0 >= 0.90),
                           "delta_test_nonnegative": (row["delta_test"] is not None and row["delta_test"] >= 0)}
        row["field_completeness"] = {"status": "FIELD-INCOMPLETE" if field_incomplete else "COMPLETE",
                                     "llm_fields": field_status,
                                     "incomplete_fields": incomplete_fields,
                                     "degraded_eval": degraded}
        row["low_judge_coverage"] = low_judge_cov

        # --- disclosed caveats carried through verbatim (deviation 7) ---
        row["queue_disclosed"] = {k: e.get(k) for k in
                                  ("train_rho_h0", "train_rho_cand", "rel_gain", "contract",
                                   "self_adversary", "note", "date") if k in e}

        report[aid] = row
        g = row.get("gate") or {}
        print(f"{aid} [{cell}]: train {r_tr_c:+.3f}/{r_tr_0:+.3f}  test cand {r_te_c:+.3f} vs h0 {r_te_0:+.3f}  "
              f"(gap {row['gap_train_minus_test']['cand']:+.3f}/{row['gap_train_minus_test']['h0']:+.3f})  "
              f"P(c>h0)={p_beat0}  G1={row['G1_verdict']}  verdict={verdict}"
              f"{'  [FIELD-INCOMPLETE: '+','.join(incomplete_fields)+']' if field_incomplete else ''}"
              f"{'  [LOW JUDGE COV n='+str(n_te)+']' if low_judge_cov else ''}")
        rows_for_table.append((aid, cell, r_tr_c, r_te_c, r_te_0, p_beat0, verdict))

    out = CENSUS / "cw_heldout_report.json"
    json.dump(report, open(out, "w"), indent=1)
    print(f"\n-> {out}")

    print(f"\n{'aspect':7s} {'cell':5s} {'train':>7s} {'test cand':>10s} {'test h0':>9s} {'P':>6s}  verdict")
    for aid, cell, tr, tc, t0, p, v in rows_for_table:
        tr_s = f"{tr:+.3f}" if tr == tr else "  nan"
        tc_s = f"{tc:+.3f}" if tc == tc else "  nan"
        t0_s = f"{t0:+.3f}" if t0 == t0 else "  nan"
        p_s = f"{p:.3f}" if p is not None else "  NA"
        print(f"{aid:7s} {str(cell):5s} {tr_s:>7s} {tc_s:>10s} {t0_s:>9s} {p_s:>6s}  {v}")


if __name__ == "__main__":
    main()
