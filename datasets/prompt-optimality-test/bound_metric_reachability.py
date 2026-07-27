"""CERTIFIED all-prompt cap #1: METRIC REACHABILITY (2026-07-24).

WHY. On deterministic-label benchmarks the all-prompt certified bound is 1.0 and PROVABLY
VACUOUS (a prompt may encode the answer key), so no information-theoretic cap can bind. But a
cap can still bind from the OTHER side: an item whose metric returns 0 even when handed an
IDEAL response is unreachable by every prompt. Then

    sup_p score(p)  <=  1 - (# certified-unreachable items) / n

which is a genuine, non-vacuous, all-prompt UPPER BOUND that needs no model, no search, and no
distributional assumption — only the metric's own source code, run on synthetic outputs.

SCOPE (declared, and the reason this is a certificate rather than an estimate): unreachability
is certified WITH RESPECT TO A DECLARED OUTPUT FAMILY F. We emit the gold answer in every
canonical surface form the family contains (bare, LaTeX, \\boxed, prose-wrapped, ...). If EVERY
member of F scores 0, the item is unreachable within F. Enlarging F can only shrink the count,
so the reported bound is CONSERVATIVE-BY-CONSTRUCTION with respect to F and is honest to quote
as "no prompt whose output lies in F can beat this".

Usage: python3 bound_metric_reachability.py aime --lm-tag Qwen3-8B
       (CPU only for exact-answer benches; no server, no GPU, no z.ai.)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dspy

import paperexact_arms as px

HERE = Path(__file__).parent


def output_forms(gold: str) -> dict[str, str]:
    """The declared output family F: canonical surface forms of one gold answer."""
    g = str(gold).strip()
    return {
        "bare": g,
        "latex_dollar": f"${g}$",
        "boxed": f"\\boxed{{{g}}}",
        "answer_prefix": f"Answer: {g}",
        "prose": f"The final answer is {g}.",
        "prose_boxed": f"Therefore the final answer is \\boxed{{{g}}}.",
        "with_reasoning": f"Let me work through this.\n\nAnswer: {g}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["aime", "hover", "hotpot", "ifbench", "livebench", "pupa"])
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--limit", type=int, default=0, help="0 = full test split")
    a = ap.parse_args()

    bench, program, metric, _ = px.load_bench(a.bench)
    test = list(bench.test_set)
    if a.limit:
        test = test[:a.limit]

    # the program's terminal output field is where an answer would land
    preds = program.named_predictors()
    out_fields = list(preds[-1][1].signature.output_fields.keys())
    print(f"[{a.bench}] n_test={len(test)} output_fields={out_fields}", flush=True)

    unreachable, reachable_by, errors = [], {}, 0
    for idx, ex in enumerate(test):
        labels = ex.labels().toDict() if hasattr(ex, "labels") else {}
        if not labels:
            print("no label fields — cannot probe this bench", flush=True)
            return
        gold = str(list(labels.values())[0])
        best, best_form = 0.0, None
        for form_name, text in output_forms(gold).items():
            kw = {f: text for f in out_fields}
            kw.update({k: v for k, v in labels.items() if k in out_fields})
            if len(out_fields) == 1:
                kw = {out_fields[0]: text}
            try:
                s = float(metric(ex, dspy.Prediction(**kw)))
            except Exception:
                errors += 1
                continue
            if s > best:
                best, best_form = s, form_name
            if best >= 1.0:
                break
        if best <= 0.0:
            unreachable.append({"idx": idx, "gold": gold[:120]})
        else:
            reachable_by[best_form] = reachable_by.get(best_form, 0) + 1

    n = len(test)
    n_unreach = len(unreachable)
    cap = 1.0 - n_unreach / n if n else float("nan")
    # VALIDITY GATE (2026-07-24): the probe is only meaningful where the terminal output field
    # IS the scored answer (exact-answer benches). On programs whose last field is a summary /
    # free response (hotpot, ifbench, hover, pupa) the synthetic Prediction is malformed, the
    # metric throws on every item, and the arithmetic would mint a spurious cap of 0.0. Refuse:
    # a certificate that fires on its own probe error is worse than no certificate.
    applicable = errors == 0 and n_unreach < n
    if not applicable:
        cap = None
    out = {
        "bench": a.bench, "n_test": n, "n_certified_unreachable": n_unreach,
        "probe_applicable": applicable,
        "certified_all_prompt_cap": cap,
        "invalid_reason": None if applicable else (
            f"probe not applicable: {errors} metric exceptions, "
            f"{n_unreach}/{n} scored zero on every declared form — the terminal output field "
            "is not the scored answer for this benchmark; NO CAP IS EMITTED"),
        "declared_output_family": list(output_forms("X").keys()),
        "first_reachable_form_counts": reachable_by,
        "metric_exceptions": errors,
        "unreachable_examples": unreachable[:25],
        "scope": ("cap is exact w.r.t. the declared output family F; enlarging F can only "
                  "lower n_unreachable, so the cap is conservative-by-construction"),
    }
    p = HERE / "runs" / f"bound_reachability_{a.bench}_{a.lm_tag}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    if applicable:
        print(f"unreachable {n_unreach}/{n} -> CERTIFIED ALL-PROMPT CAP = {cap:.4f}", flush=True)
    else:
        print(f"PROBE NOT APPLICABLE ({errors} metric exceptions, {n_unreach}/{n} all-zero) "
              "— no cap emitted", flush=True)
    print(f"reachable-by-form: {reachable_by}   metric exceptions: {errors}", flush=True)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
