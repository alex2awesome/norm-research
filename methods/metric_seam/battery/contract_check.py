"""WS1 E2+: run a criterion's FROZEN construct contract against a candidate program.

The adversary runs this EVERY crew round; a candidate that fails contract never reaches
a gate (runbook 2026-07-10 pre-registration). Gate verdict, pre-registered here before
any E2 data:
  PASS = (a) no INVERTED probe (score(text_neg) > score(text_pos)),
         (b) strict separation on >= 75% of probes (None/tie counts as not separated),
         (c) TRAIN score std >= min_std AND frac-at-mode <= max_frac_at_mode.
Probes run with extracted={} — they exercise the CODE path only (no Gemma fields exist
for synthetic probe texts; a candidate whose probe scores are all None is code-path-inert
and cannot pass). TEST SPLIT IS NEVER TOUCHED.

PROBE-MODE-DETECTION guard (2026-07-10 audit): extracted={} is a fingerprint of probe mode
a candidate could branch on (`if not extracted: ...`) to game probes while behaving like h0
on train. Every probe is therefore scored twice — with {} and with a sentinel NON-EMPTY dict
whose lookups still return None for every real field. Divergence = the candidate keys on the
fingerprint = hard FAIL. (Keying on a specific field's absence is indistinguishable from
legitimate missing-field handling and stays the adversary's line-audit job.)

HARNESS v2 (2026-07-10, post external-review): (1) TRAIN-ONLY EXECUTION — candidates are
invoked on train texts only; held-out texts are never passed to candidate code (makes the
pre-registered seal structural, was procedural). (2) COMPLETENESS/RANGE GATES — >=90% of
train items must score, scores must be finite and in [0,1] (a candidate crashing on 110/150
items could previously pass on the 40-item floor). Cells judged under v1: PR a41, humor
a342, PR a31, humor a189, PR a111 (2026-07-10, pre-v2); all later cells + E3/E4 use v2.

Usage: python3 contract_check.py <task> <aid> <candidate.py>
Exit 0 = contract PASS, 1 = FAIL.
"""
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from battery_common import load_ctx, load_mod, run_prog, BASE  # noqa: E402

EL = BASE / "battery/effort_ladder"
SEP_FRAC = 0.75
SENTINEL = {"__cc_sentinel__": None}  # non-empty; .get(<any real field>) -> None, same as {}


def probe_score(fn, text, ops):
    """Score text-only; returns (score, mode_detect) where mode_detect flags divergence
    between extracted={} and the sentinel dict."""
    def one(ex):
        try:
            return fn(text, dict(ex), ops)
        except Exception:
            return None
    s_empty, s_sent = one({}), one(SENTINEL)
    return s_empty, (s_empty != s_sent)


def evaluate_validity_controls(fn, controls, ops):
    """Evaluate typed G2 controls without weakening the historical CF gate.

    A positive contrast must separate in the intended direction.  A negative
    proxy trap contains two construct-satisfied texts and therefore must *not*
    separate in that direction.  Returning structured rows keeps this helper
    usable by tests and by future non-CLI callers.
    """
    rows = []
    for control in controls:
        polarity = control.get("polarity")
        if polarity not in {"positive", "negative_proxy_trap"}:
            raise ValueError(f"unknown validity-control polarity: {polarity!r}")
        sp, mp = probe_score(fn, control["text_pos"], ops)
        sn, mn = probe_score(fn, control["text_neg"], ops)
        separated = sp is not None and sn is not None and sp > sn
        if polarity == "positive":
            passed = separated
            outcome = "SEP" if separated else "FAIL"
        else:
            passed = not separated
            outcome = "FIRE" if separated else "NO_FIRE"
        rows.append({
            "control_id": control.get("control_id"),
            "polarity": polarity,
            "score_pos": sp,
            "score_neg": sn,
            "outcome": outcome,
            "passed": passed,
            "mode_detect": bool(mp or mn),
        })
    return rows


def main():
    task, aid, cand = sys.argv[1], sys.argv[2], sys.argv[3]
    contract = json.load(open(EL / "contracts" / f"{task}__{aid}.json"))
    ctx = load_ctx(task)
    mod = load_mod(pathlib.Path(cand))

    outcomes, n_mode_detect = [], 0
    for i, p in enumerate(contract["cf_probes"]):
        sp, dp_flag = probe_score(mod.score, p["text_pos"], ctx["ops"])
        sn, dn_flag = probe_score(mod.score, p["text_neg"], ctx["ops"])
        n_mode_detect += dp_flag + dn_flag
        o = ("NONE" if sp is None or sn is None else
             "SEP" if sp > sn else "INV" if sp < sn else "TIE")
        outcomes.append(o)
        flag = "  [MODE-DETECT]" if (dp_flag or dn_flag) else ""
        print(f"probe {i}: {o}  pos={sp} neg={sn}  ({p['why'][:80]}){flag}")

    # G2 is mandatory only for newly authored validity-v2 contracts.  Historical
    # contracts remain replayable but are explicitly not upgraded by absence.
    validity_rows = evaluate_validity_controls(
        mod.score, contract.get("validity_controls", []), ctx["ops"])
    g2_required = contract.get("construct_validity_version", 1) >= 2
    g2_has_both = ({r["polarity"] for r in validity_rows}
                   == {"positive", "negative_proxy_trap"})
    g2_ok = (not g2_required) or (
        g2_has_both and all(r["passed"] and not r["mode_detect"] for r in validity_rows))
    if validity_rows:
        for r in validity_rows:
            print(f"G2 {r['control_id']}: {r['polarity']} {r['outcome']} "
                  f"pos={r['score_pos']} neg={r['score_neg']}")
    elif g2_required:
        print("G2 FAIL: validity-v2 requires positive and negative_proxy_trap controls")
    else:
        print("G2 NOT EVALUATED: legacy contract; no construct-validity claim")

    fmap = ctx["f_orig"].get(aid, {})
    train_ids = sorted(ctx["train"])
    train_texts = {d: ctx["items"][d] for d in train_ids if d in ctx["items"]}
    col = run_prog(mod.score, train_texts, fmap, ctx["ops"])
    tr = [round(col[d], 6) for d in train_ids if col.get(d) is not None]
    frac_scored = len(tr) / max(1, len(train_texts))
    ok_complete = frac_scored >= 0.90
    ok_range = all(0.0 <= x <= 1.0 for x in tr)
    if len(tr) < 40:
        print(f"FAIL: only {len(tr)} scoreable train items")
        sys.exit(1)
    mean = sum(tr) / len(tr)
    std = (sum((x - mean) ** 2 for x in tr) / len(tr)) ** 0.5
    frac_mode = Counter(tr).most_common(1)[0][1] / len(tr)
    dc = contract["discrimination_checks"]

    ok_inv = "INV" not in outcomes
    ok_sep = outcomes.count("SEP") >= SEP_FRAC * len(outcomes)
    ok_std = std >= dc["min_std"]
    ok_mode = frac_mode <= dc["max_frac_at_mode"]
    ok_probemode = n_mode_detect == 0
    print(f"probes: {outcomes.count('SEP')}/{len(outcomes)} SEP, inverted={not ok_inv}, "
          f"probe-mode-detection={n_mode_detect}")
    print(f"train discrimination: std={std:.4f} (>={dc['min_std']}) "
          f"frac_at_mode={frac_mode:.3f} (<={dc['max_frac_at_mode']}) n={len(tr)} "
          f"scored={frac_scored:.2f} (>=0.90) in_range={ok_range}  [harness v2, train-only]")
    verdict = (ok_inv and ok_sep and ok_std and ok_mode and ok_probemode and g2_ok
               and ok_complete and ok_range)
    if verdict:
        print("CONTRACT PASS")
    elif not ok_probemode:
        print("CONTRACT FAIL (PROBE-MODE-DETECTION)")
    elif not (ok_complete and ok_range):
        print("CONTRACT FAIL (COMPLETENESS/RANGE)")
    elif not g2_ok:
        print("CONTRACT FAIL (G2 CONSTRUCT-VALIDITY CONTROLS)")
    else:
        print("CONTRACT FAIL")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
