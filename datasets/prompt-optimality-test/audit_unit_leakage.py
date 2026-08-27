"""Answer-key leakage audit over the frozen unit pools (HB89/HB91b mandate, 2026-07-25).

WHY THIS IS MANDATORY BEFORE SUBMISSION. Our own vacuity theorem says that on deterministic-label
benchmarks `sup_p score(p) = 1.0` exactly, because a prompt MAY simply encode the answer key. That
cuts both ways: it is also the most dangerous latent objection to our own results. If any mined
unit carries item-specific answer content, then "recombination beats reflective search" reduces to
"recombination smuggled the labels in", and every certificate built on that pool inherits the flaw.

The units were mined by an LLM from optimizer trajectories, and those trajectories saw training
items — so leakage is a live possibility, not a hypothetical.

WHAT COUNTS AS LEAKAGE (graded, not binary):
  none      - generic strategy/format instruction ("end with 'Answer:'", "work tag by tag")
  domain    - general domain knowledge, the kind a textbook states ("the p-th roots of unity
              satisfy 1+x+...+x^(p-1)") - legitimate articulated knowledge, and the paper's
              actual subject matter
  item_hint - unusually specific content that would only help on a particular family of items
              (a named competition problem, an oddly specific constant)
  answer    - an actual answer, label, or answer key for a specific item
Only `answer` (and arguably `item_hint`) invalidates a unit. `domain` is what we WANT to find -
it is articulated knowledge, which is the paper's thesis, not a defect.

The audit is deliberately adversarial: the judge is told to hunt for leakage, and every unit is
judged independently with the benchmark's task described but WITHOUT the test items, so the judge
cannot itself be the leak.

Usage:
  python3 audit_unit_leakage.py --benches livebench aime hover hotpot ifbench
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import paperexact_arms as px

HERE = Path(__file__).parent

TASK_DESC = {
    "livebench": "LiveBench math: competition math problems, plus 'fill in the missing formula "
                 "tags' items where the model outputs a comma-separated list of expression ids.",
    "aime": "AIME 2025 competition math; the answer is a 3-digit integer 000-999.",
    "hover": "HoVer multi-hop claim verification over Wikipedia; label is SUPPORTED/NOT_SUPPORTED.",
    "hotpot": "HotpotQA multi-hop question answering; the answer is a short string.",
    "ifbench": "IFBench instruction-following; scored by programmatic constraint checkers.",
    "pupa": "PUPA privacy-preserving delegation; a judge scores task quality and PII leakage.",
}

PROMPT = """You are auditing prompt-optimization artifacts for ANSWER-KEY LEAKAGE.

Benchmark: {task}

Below is one instruction clause ("unit") that an optimizer mined and may append to a prompt.
Classify how much benchmark-specific information it leaks. Be adversarial: we are actively trying
to FIND leakage, and a false negative is far more costly to us than a false positive.

Categories:
- "none": generic strategy, formatting, or output-shape instruction. No benchmark content.
- "domain": general domain knowledge or a standard technique, the kind a textbook would state.
  Legitimate. Not leakage.
- "item_hint": unusually specific content that would only help on one particular item or a narrow
  family of items (a named problem, an oddly specific constant or edge case).
- "answer": contains an actual answer, label, or answer key for a specific item.

UNIT:
\"\"\"{unit}\"\"\"

Reply with STRICT JSON only, no prose, no code fence:
{{"category": "none|domain|item_hint|answer", "confidence": 0.0-1.0, "reason": "<one sentence>", "quote": "<the specific span that drove the call, or empty>"}}"""


def parse_json(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", nargs="+",
                    default=["livebench", "aime", "hover", "hotpot", "ifbench"])
    ap.add_argument("--lm-tag", default="Qwen3-8B")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--out", default="runs/unit_leakage_audit.json")
    a = ap.parse_args()

    judge = px.make_reflection_lm(a.model, patient=True)
    out_path = HERE / a.out
    out_path.parent.mkdir(exist_ok=True)
    all_results = {}
    if out_path.exists():                      # resume-by-id: never redo settled units
        all_results = json.loads(out_path.read_text())

    for bench in a.benches:
        pool_file = HERE / "pools" / f"{bench}_{a.lm_tag}_frozen.json"
        if not pool_file.exists():
            print(f"[{bench}] no frozen pool, skipping", flush=True)
            continue
        units = json.loads(pool_file.read_text())["units"]
        done = {r["unit"]: r for r in all_results.get(bench, {}).get("units", [])}
        recs = []
        print(f"[{bench}] auditing {len(units)} units ({len(done)} already done)", flush=True)
        for i, u in enumerate(units):
            text = u["unit"]
            if text in done and done[text].get("category"):
                recs.append(done[text])
                continue
            msg = PROMPT.format(task=TASK_DESC.get(bench, bench), unit=text)
            verdict = None
            for attempt in range(3):           # judge can emit prose; retry a different sample
                try:
                    raw = judge(messages=[{"role": "user", "content": msg}])
                    raw = raw[0] if isinstance(raw, list) else raw
                except Exception as e:
                    print(f"  unit {i}: judge error {type(e).__name__}", flush=True)
                    continue
                verdict = parse_json(raw)
                if verdict and verdict.get("category") in ("none", "domain", "item_hint", "answer"):
                    break
                verdict = None
            rec = {"unit": text, "module": u.get("module"), "source": u.get("source")}
            rec.update(verdict or {"category": None, "reason": "JUDGE_FAILED"})
            recs.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(units)}", flush=True)

        counts = {}
        for r in recs:
            counts[r.get("category") or "FAILED"] = counts.get(r.get("category") or "FAILED", 0) + 1
        flagged = [r for r in recs if r.get("category") in ("item_hint", "answer")]
        all_results[bench] = {"n_units": len(units), "counts": counts,
                              "n_flagged": len(flagged), "flagged": flagged, "units": recs}
        print(f"[{bench}] {counts}  FLAGGED={len(flagged)}", flush=True)
        for r in flagged:
            print(f"    !! {r['category']}: {r.get('quote','')[:100]}", flush=True)
        out_path.write_text(json.dumps(all_results, indent=1))

    print("\n=== SUMMARY ===", flush=True)
    for b, r in all_results.items():
        print(f"  {b:10s} n={r['n_units']:4d} {r['counts']}  flagged={r['n_flagged']}", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
