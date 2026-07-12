"""Standing honest-numbers audit for §12.6 certificates + Face-2 grids. Enforces two disciplines
the 2026-07-02 audit surfaced, WITHOUT touching the shared verdict logic in value_certificate:

  D1 degenerate-target filter: a metric whose target entropy `H_M < floor` (default 0.15 bits ≈
     a near-constant readout) is VACUOUSLY certifiable — exclude it and flag any CODIFIABLE it
     produced as spurious. (Found 1 CW + 5 humor degenerate; 1 spurious humor CODIFIABLE.)

  D2 reference-executor exclusion: in a decompression grid the target M is one executor's reading;
     that SAME executor as a reader scores itself → self-consistency near-ceiling, not a data point.
     Cross-reader gaps must exclude the reference executor. (8B-vs-3B was contaminated; 3B-vs-1B and
     70B-vs-3B are clean when 8B is the reference.)

CPU, read-only. Usage:
  python -m methods.metric_implementer.experiments.audit_certificate --cert <cert.json> \
      [--grid-report <report.json> --ref-executor Llama-3.1-8B] [--floor 0.15]
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np


def audit_cert(rows, floor=0.15):
    """Return corrected verdict counts + the degenerate/spurious lists."""
    degen = [r for r in rows if (r.get("H_M") or 0.0) < floor]
    keep = [r for r in rows if (r.get("H_M") or 0.0) >= floor]
    spurious = [r for r in rows if r.get("verdict") == "CODIFIABLE" and (r.get("H_M") or 0.0) < floor]
    passg = sum(1 for r in keep if r.get("form_invariant") is True)
    return {
        "n_total": len(rows), "n_degenerate": len(degen), "n_keep": len(keep),
        "verdicts_raw": dict(collections.Counter(r.get("verdict") for r in rows)),
        "verdicts_kept": dict(collections.Counter(r.get("verdict") for r in keep)),
        "spurious_codifiable": [str(r.get("name"))[:50] for r in spurious],
        "form_gate_pass_kept": passg,
        "form_gate_pass_rate": round(passg / max(len(keep), 1), 3),
        "degenerate_names": [str(r.get("name"))[:50] for r in degen],
    }


def clean_reader_gaps(report, ref_executor):
    """Median cross-reader bal_acc gaps using ONLY readers whose tag does not contain
    `ref_executor` (the target-generating executor). Returns {rung: {pair: gap}} for adjacent
    reader-size pairs among the non-reference readers, plus which readers were excluded."""
    tags = list(report)
    non_ref = [t for t in tags if ref_executor not in t]
    excluded = [t for t in tags if ref_executor in t]
    # order readers by the size token if present (1B<3B<8B<70B), else lexical
    def size(t):
        for i, s in enumerate(["1B", "3B", "8B", "31B", "70B", "122B"]):
            if s in t:
                return i
        return 99
    non_ref.sort(key=size)
    rungs = ["name", "definition", "explanation", "full_rubric", "exemplars", "dossier"]
    out = {"excluded_reference_readers": excluded, "readers_used": non_ref, "gaps": {}}
    for a, b in zip(non_ref[1:], non_ref[:-1]):          # adjacent pairs, larger − smaller
        label = f"{a.split('-')[-1] if '-' in a else a}−{b.split('-')[-1] if '-' in b else b}"
        for r in rungs:
            gaps = [report[a][g][r]["bal_acc"] - report[b][g][r]["bal_acc"]
                    for g in report[a] if g in report[b]
                    and r in report[a].get(g, {}) and r in report[b].get(g, {})]
            if gaps:
                out["gaps"].setdefault(label, {})[r] = round(float(np.median(gaps)), 3)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cert", required=True)
    p.add_argument("--grid-report", default=None)
    p.add_argument("--ref-executor", default="8B",
                   help="substring identifying the target-generating executor's reader tag")
    p.add_argument("--floor", type=float, default=0.15)
    p.add_argument("--json", default=None)
    a = p.parse_args()

    result = {"cert": audit_cert(json.load(open(a.cert)), floor=a.floor)}
    c = result["cert"]
    print(f"D1 degenerate-target filter (H_M >= {a.floor} bits):")
    print(f"  {c['n_degenerate']}/{c['n_total']} degenerate -> keep {c['n_keep']}")
    print(f"  verdicts raw : {c['verdicts_raw']}")
    print(f"  verdicts kept: {c['verdicts_kept']}")
    if c["spurious_codifiable"]:
        print(f"  ⚠ spurious CODIFIABLE on degenerate target: {c['spurious_codifiable']}")
    print(f"  form-gate PASS (kept): {c['form_gate_pass_kept']}/{c['n_keep']} = {c['form_gate_pass_rate']:.0%}")

    if a.grid_report:
        result["grid"] = clean_reader_gaps(json.load(open(a.grid_report)), a.ref_executor)
        g = result["grid"]
        print(f"\nD2 reference-executor exclusion (ref='{a.ref_executor}'):")
        print(f"  excluded (self-referential) readers: {g['excluded_reference_readers']}")
        print(f"  clean cross-reader gaps: {json.dumps(g['gaps'], indent=2)}")

    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
