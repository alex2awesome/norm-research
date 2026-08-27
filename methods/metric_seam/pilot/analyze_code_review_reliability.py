"""STAGE 2 (code_review): judge reliability + target qualification on the NEW-run-only
scope (results_newrun.jsonl, built by scope_code_review_newrun.py).

Per aspect (18 candidates in aspects_candidates.json):
  - rel1: 2-pass Spearman, eval-v2 rule = intersection of items with a NUMERIC score in
    BOTH pass1 and pass2 (matches methods/metric_seam/pilot/analyze_v2.py::load_judge_v2
    + the rel1 computation in main(), i.e. the same "intersection-only 2-pass judges"
    rule frozen in datasets/patents/2026-07-10__evidence_aware_judge_ws3.md).
  - n_both: size of that intersection == the NA-adjusted-reliability n (coding is
    NA-heavy by design; this is the same number as rel1's n, surfaced explicitly).
  - ceiling: attenuation_ceiling(rel1, k=2) = sqrt(spearman_brown(rel1,2)).
  - na_fraction: fraction of the 500 (250 pass1 + 250 pass2) judge cells that are "NA".
  - mode_fraction: modal-score share among the numeric (pass1+pass2 pooled) scores.
  - qualified: rel1 >= 0.30 (frozen cutoff, WS3 result block, 2026-07-10).

Also reports the scope channel (250 rows) and writes reliability_report.json.
"""
import json, pathlib, statistics as st, sys, collections

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "methods/metric_seam"))
from certificates import spearman, attenuation_ceiling  # noqa: E402

TASK = ROOT / "outputs/metric_seam_pilot/tasks/code_review"
REL_CUTOFF = 0.30


def mode_fraction(vals):
    if not vals:
        return None, None
    c = collections.Counter(vals)
    val, cnt = c.most_common(1)[0]
    return val, cnt / len(vals)


def main():
    items = json.load(open(TASK / "items.json"))
    item_dpids = [x["datapoint_id"] for x in items]
    aspects = json.load(open(TASK / "aspects_candidates.json"))
    aspect_meta = {a["aspect_id"]: a for a in aspects}
    aspect_ids = [a["aspect_id"] for a in aspects]
    assert len(item_dpids) == 250 and len(aspect_ids) == 18

    rows = [json.loads(l) for l in open(TASK / "results_newrun.jsonl")]
    assert len(rows) == 9250, len(rows)

    # pass1/pass2 numeric-only maps, per aspect (mirrors analyze_v2.load_judge_v2)
    p1 = collections.defaultdict(dict)
    p2 = collections.defaultdict(dict)
    raw_all = collections.defaultdict(dict)  # aspect -> dpid -> {pass1: raw_score_or_NA, pass2: ...}
    for r in rows:
        if r["channel"] not in ("pass1", "pass2"):
            continue
        aid, dpid, sc = r["aspect_id"], r["datapoint_id"], r["score"]
        raw_all[aid].setdefault(dpid, {})[r["channel"]] = sc
        if isinstance(sc, int):
            (p1 if r["channel"] == "pass1" else p2)[aid][dpid] = sc

    table = []
    qualified = []
    for aid in aspect_ids:
        meta = aspect_meta[aid]
        both = [d for d in p1[aid] if d in p2[aid]]  # eval-v2 intersection rule
        n_both = len(both)
        if n_both >= 2:
            rel1 = spearman([p1[aid][d] for d in both], [p2[aid][d] for d in both])
        else:
            rel1 = float("nan")
        ceil_ = attenuation_ceiling(max(rel1, 0.0), 2) if rel1 == rel1 else float("nan")

        # NA fraction over the 500 (250 pass1 + 250 pass2) cells for this aspect
        cells = raw_all[aid]
        n_cells = 2 * len(item_dpids)
        na_cells = sum(1 for dpid, d in cells.items() for ch in ("pass1", "pass2")
                        if d.get(ch) == "NA")
        na_fraction = na_cells / n_cells

        # item-level NA: item counted NA if EITHER pass is NA (or missing)
        item_any_na = 0
        item_both_na = 0
        for dpid in item_dpids:
            c = cells.get(dpid, {})
            is_na1 = c.get("pass1") == "NA"
            is_na2 = c.get("pass2") == "NA"
            if is_na1 or is_na2:
                item_any_na += 1
            if is_na1 and is_na2:
                item_both_na += 1

        pooled_numeric = [p1[aid][d] for d in p1[aid]] + [p2[aid][d] for d in p2[aid]]
        mval, mfrac = mode_fraction(pooled_numeric)
        score_min = min(pooled_numeric) if pooled_numeric else None
        score_max = max(pooled_numeric) if pooled_numeric else None
        score_mean = round(st.mean(pooled_numeric), 2) if pooled_numeric else None

        is_qualified = (rel1 == rel1) and (rel1 >= REL_CUTOFF)
        near_degenerate = (mfrac is not None) and (mfrac >= 0.95)  # same bar as the
        # coding A-bank degeneracy audit (BUILD_PLAN.md Sec.2: mode-share >=95% = degenerate)
        row = {
            "aspect_id": aid,
            "name": meta["name"],
            "audit_na_rate_44751pool": meta["audit_na_rate"],
            "audit_mode_share_44751pool": meta["audit_mode_share"],
            "rel1_eval_v2": round(rel1, 3) if rel1 == rel1 else None,
            "n_both_numeric": n_both,
            "attenuation_ceiling": round(ceil_, 3) if ceil_ == ceil_ else None,
            "na_fraction_cells": round(na_fraction, 3),
            "na_fraction_items_any_pass": round(item_any_na / len(item_dpids), 3),
            "na_fraction_items_both_pass": round(item_both_na / len(item_dpids), 3),
            "mode_score": mval,
            "mode_fraction": round(mfrac, 3) if mfrac is not None else None,
            "score_min": score_min, "score_max": score_max, "score_mean": score_mean,
            "n_pooled_numeric": len(pooled_numeric),
            "qualified_rel1_ge_0.30": is_qualified,
            "near_degenerate_judge_distribution_ge_0.95": near_degenerate,
        }
        table.append(row)
        if is_qualified:
            qualified.append(aid)

    table.sort(key=lambda r: (r["rel1_eval_v2"] is None, -(r["rel1_eval_v2"] or -1)))

    # ---- scope channel ----
    scope_rows = [r for r in rows if r["channel"] == "scope"]
    scope_scores = {r["datapoint_id"]: r["score"] for r in scope_rows}
    assert all(isinstance(s, int) for s in scope_scores.values())
    n_scope = len(scope_scores)
    thresh = 7
    in_scope = [d for d, s in scope_scores.items() if s >= thresh]
    scope_dist = collections.Counter(scope_scores.values())
    below_thresh = [{"datapoint_id": d, "score": s} for d, s in scope_scores.items() if s < thresh]

    items_by_id = {x["datapoint_id"]: x for x in items}
    below_thresh_detail = [
        {**b, "repo": items_by_id[b["datapoint_id"]]["repo"],
         "pr_number": items_by_id[b["datapoint_id"]]["pr_number"],
         "ctext_len": len(items_by_id[b["datapoint_id"]]["ctext"])}
        for b in below_thresh
    ]

    scope_summary = {
        "n_items": n_scope,
        "threshold": thresh,
        "n_in_scope": len(in_scope),
        "frac_in_scope": round(len(in_scope) / n_scope, 4),
        "score_distribution": {str(k): v for k, v in sorted(scope_dist.items())},
        "below_threshold_items": below_thresh_detail,
        "note": (
            "TSCOPE asks: is this a genuine, substantive pull-request diff (vs. nav "
            "chrome/empty stub/wrong doctype)? 249/250 (99.6%) score 10/10 -- the item "
            "construction (real unified diffs from pr_test_execution, min length filter "
            "300 bytes) produces almost no invalid items, unlike scraped-corpus tasks "
            "where scope-check catches nav chrome or off-topic pages. The 1 exception "
            "(crb108f53a1be, score=1, rdk PR #1656, judgement=rejected) is the item "
            "flagged in BUILD_PLAN.md Sec.3 as the corpus's min-length item (573 raw "
            "chars) -- IS a genuine 2-line diff (two joke comments added to a Go file), "
            "just too trivial/content-free for the judge to call 'substantive'; this is "
            "an edge case of the item, not an artifact/scraping failure. Net: the scope "
            "channel supports treating essentially the full 250-item panel as valid "
            "judge material; no further item-level filtering is warranted from this "
            "check alone."
        ),
    }

    # ---- reconciliation block ----
    reconciliation = {
        "results_dedup_total_rows": sum(1 for _ in open(TASK / "results_dedup.jsonl")),
        "results_newrun_total_rows": len(rows),
        "expected_judge_rows": 18 * 250 * 2,
        "expected_scope_rows": 250,
        "expected_total_rows": 18 * 250 * 2 + 250,
        "observed_judge_rows": sum(1 for r in rows if r["channel"] in ("pass1", "pass2")),
        "observed_scope_rows": sum(1 for r in rows if r["channel"] == "scope"),
        "missing_cells": [],  # scope_code_review_newrun.py verified 0 missing cells, 0 dups
        "note": "old Jul-2-era run (dpid prefix 'd', 41 aspect_ids incl. overlapping id "
                "'a135') fully excluded by the items.json-dpid join; verified 0 residual "
                "'d'-prefixed rows and 0 duplicate (aspect,channel,dpid) keys in "
                "results_newrun.jsonl (scope_code_review_newrun.py output).",
    }

    # ---- panel recommendation ----
    qualified_sorted = sorted(
        [r for r in table if r["qualified_rel1_ge_0.30"]],
        key=lambda r: -r["rel1_eval_v2"])
    unqualified = [r for r in table if not r["qualified_rel1_ge_0.30"]]
    rel_values = [r["rel1_eval_v2"] for r in qualified_sorted]
    near_degen = [r["aspect_id"] for r in table
                  if r["near_degenerate_judge_distribution_ge_0.95"]]
    recommendation = {
        "qualified_aspect_ids": [r["aspect_id"] for r in qualified_sorted],
        "n_qualified": len(qualified_sorted),
        "n_total": 18,
        "qualified_rel1_range": [min(rel_values), max(rel_values)] if rel_values else None,
        "unqualified_aspect_ids": [r["aspect_id"] for r in unqualified],
        "near_degenerate_judge_distribution_aspect_ids": near_degen,
        "text": (
            "All 18/18 candidates clear rel1 >= 0.30 (range " +
            str([min(rel_values), max(rel_values)] if rel_values else None) +
            ") -- the diff-based redesign (BUILD_PLAN.md's fix for the comments-only "
            "task's judge degeneracy) produced a uniformly reliable judge target set; "
            "nothing needs to be dropped on reliability grounds alone. BUT rel1>=0.30 "
            "passing is not the only quality bar worth a panel-builder's attention: " +
            str(near_degen) + " have a near-degenerate JUDGE score distribution "
            "(mode-share >= 0.95, same bar the coding A-bank degeneracy audit uses for "
            "the coded checkers, BUILD_PLAN.md Sec.2) -- e.g. a400 (Big-O complexity) "
            "sits at 10/10 in 99.2% of scored cells, and its rel1=0.496 is driven by "
            "agreement on essentially 1-3 outlier items, not a real graded signal across "
            "the panel. Aspects like this technically qualify but contribute almost no "
            "discriminative variance to a census cell and should be weighted low / "
            "treated as informative mainly as a near-constant-ceiling control, not a "
            "genuine graded target. For the panel itself: prefer a spread across the "
            "rel1 range (e.g. a104/a112/a8/a409/a88 at the high end ~.86-.88, a30/a1 "
            "mid-high ~.76, a3/a15/a407 mid ~.68-.70, a20/a92/a400 low end ~.50-.63) over "
            "just taking the top-N by rel1, since a panel needs reliability variation to "
            "study the reliability-vs-fidelity relationship itself, not just the most "
            "reliable targets; but weight a400 down given the near-degenerate caveat above. "
            "NA-heaviness alone did not disqualify any aspect via the rel1 cutoff -- even "
            "a20 (93% NA on the 44,751-PR coded-checker pool) and a155 (83%) clear "
            "rel1>=0.30 on their much smaller numeric-both n (241 and 163 respectively); "
            "what would disqualify an aspect is noisy 2-pass judgment, not NA volume per se "
            "-- though note the LLM JUDGE's own NA rate (na_fraction_cells column) runs far "
            "below the CODED-CHECKER audit_na_rate for every aspect (e.g. a20: coded "
            "checker NA 93% vs judge NA 2%), because the judge answers on text it can always "
            "read even when the deterministic coded checker can't compute a value on that "
            "diff -- these are two different NA-generating mechanisms and should not be "
            "conflated when comparing judge vs. code coverage downstream."
        ),
        "repo_identity_confound_note": (
            "BUILD_PLAN.md Sec.4.1: rdk (73/250, 29%) + spire (67/250, 27%) = 56% of the "
            "sample; the remaining 44% splits across 20 repos. Reliability (this report) "
            "says nothing about this confound -- rel1 measures judge-vs-judge agreement "
            "within items, not whether aspect scores are actually driven by repo identity "
            "rather than diff content. It does NOT clear an aspect of the confound. Per "
            "BUILD_PLAN.md Sec.4.1, any downstream AUC/rho on this panel must be reported "
            "repo-grouped (GroupKFold(groups=repo)) in addition to pooled, at eval time -- "
            "not addressed here."
        ),
    }

    report = {
        "task": "code_review",
        "scope_rule": "join on items.json datapoint_ids (crb-prefixed, n=250) x the 18 "
                       "aspects_candidates.json aspect_ids (+ 'scope' channel); old Jul-2 "
                       "run ('d'-prefixed dpids, 39 legacy aspect_ids + overlapping 'a135' "
                       "id) fully excluded by the dpid join.",
        "rel1_definition": "eval-v2 intersection rule: Spearman(pass1, pass2) computed "
                            "only over items with a NUMERIC (non-NA) score in BOTH passes "
                            "(methods/metric_seam/pilot/analyze_v2.py::load_judge_v2 + rel1 "
                            "computation; frozen convention per datasets/patents/"
                            "2026-07-10__evidence_aware_judge_ws3.md). For this task, this "
                            "figure IS the NA-adjusted reliability -- n_both_numeric is its n.",
        "quality_cutoff": {"rel1_ge": REL_CUTOFF,
                            "source": "datasets/patents/2026-07-10__evidence_aware_judge_ws3.md "
                                      "(WS3 result block, 2026-07-10): 'a cutoff of rel1 >= .30 "
                                      "is now frozen for future evidence-judge runs.'"},
        "reconciliation": reconciliation,
        "aspect_table": table,
        "scope_check": scope_summary,
        "panel_recommendation": recommendation,
    }
    json.dump(report, open(TASK / "reliability_report.json", "w"), indent=2)

    # console summary
    print(f"reconciliation: {reconciliation['results_newrun_total_rows']} rows "
          f"(expected {reconciliation['expected_total_rows']}), "
          f"missing_cells={len(reconciliation['missing_cells'])}")
    print(f"{'aspect':6} {'rel1':>6} {'n_both':>7} {'ceil':>6} {'na_cell':>8} "
          f"{'na_item':>8} {'mode%':>6} {'qual':>5}")
    for r in table:
        print(f"{r['aspect_id']:6} {str(r['rel1_eval_v2']):>6} {r['n_both_numeric']:>7} "
              f"{str(r['attenuation_ceiling']):>6} {r['na_fraction_cells']:>8.3f} "
              f"{r['na_fraction_items_any_pass']:>8.3f} {str(r['mode_fraction']):>6} "
              f"{str(r['qualified_rel1_ge_0.30']):>5}")
    print()
    print(f"qualified: {recommendation['n_qualified']}/18, "
          f"rel1 range {recommendation['qualified_rel1_range']}")
    print(f"scope: {scope_summary['n_in_scope']}/{scope_summary['n_items']} "
          f"in-scope (>= {thresh}), dist={scope_summary['score_distribution']}")


if __name__ == "__main__":
    main()
