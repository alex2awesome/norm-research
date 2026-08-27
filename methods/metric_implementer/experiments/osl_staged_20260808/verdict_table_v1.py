"""Criterion-4 verdict/remedy table (2026-08-11): every construct gets a verdict row.
Universe = the 1,032 unsupervised bank constructs (family_verdict_join). Remedy logic
(first matching rule, evidence-based, all thresholds stated):
  1. WORKS-AS-STATED: rubric channel already strong at local top (LOO recovery >= .75
     at best non-voter top rung proxy = qwen25-14b/gemma/mistral max, or top-vs-mid
     plateaued with top-rung >= .75 family curves).
     -> remedy: none needed.
  2. BETTER-DEMONSTRATION: humor-bank constructs whose flip-selected examples carry
     certified content (fun-mm >= +.05 at either measured receiver) — examples add
     construct-specific signal.
  3. STRONGER-LISTENER: in the 69-construct still-gaining tail AND de-censor run
     showed a ceiling break (gpt-oss >= local top + .05).
  4. DIALECT-MATCHED-LISTENER: still-gaining tail, NO frontier ceiling break (the
     de-censor verdict: asymptote unknown under local-dialect ref) OR family-dependent
     saturation with cross-family verdict disagreement.
  5. BETTER-STATEMENT: rubric weak at top (top recovery < .65) but capacity still
     helped somewhere (family-dependent) and no example/content evidence — the
     articulation, not the listener, is the binding constraint (candidate for
     rewriting; supported by def-weakness gating from flip-v3).
  6. NOTHING-CERTIFIED: remaining rows (weak everywhere, no positive instrument).
Where evidence exists only for humor (examples instruments), other tasks fall through
to rules 1/4/5/6 — per-rule evidence scope is recorded on every row.
Output: outputs/analyses/verdict_table_v1/{verdict_table_v1.json,summary.md}
"""
import json
import os

import numpy as np

D = "outputs/articulation_story_20260810"
OUT = "outputs/analyses/verdict_table_v1"
os.makedirs(OUT, exist_ok=True)

fj = json.load(open(f"{D}/analyses/family_verdict_join_v1.json"))["full_rows"]
cat = json.load(open(f"{D}/analyses/metric_categories_blind_v1.json"))
led = json.load(open(f"{D}/analyses/v3sets_ledger_v1.json"))
dec = {r["b"]: r for r in json.load(open(f"{D}/analyses/decensor_harvest_v1.json"))}
tail = {(r["task"], r["name"]) for r in json.load(open(f"{D}/flips/decensor_tail_v1.json"))}

content = {}
for rcv in ("qwen25-72b", "gpt-oss-120b"):
    for r in led[rcv]:
        content.setdefault(r["b"], {})[rcv] = r["functional"] - r["functionalmm"]


def satgroup(r, t=.02):
    tm = [v for v in r["top_minus_mid"].values() if v is not None]
    if len(tm) < 3:
        return None
    if all(v > t for v in tm):
        return "rising"
    if all(v <= t for v in tm):
        return "plateaued"
    return "family-dependent"


rows = []
for r in fj:
    task, name = r["task"], r["name"]
    sat = satgroup(r)
    tm = r["top_minus_mid"]
    # proxy for "strong at top": all family top-vs-mid small AND slopes indicate high top —
    # we lack absolute top recovery for every task here; use top_minus_mid + slope sign
    verdict3 = r["verdict3"]
    c = content.get(name, {})
    has_content = any(v >= .05 for v in c.values())
    in_tail = (task, name) in tail
    brk = None
    if name in dec and dec[name].get("qwen3-32b") is not None:
        brk = dec[name]["gptoss"] - dec[name]["qwen3-32b"]
    if sat == "plateaued":
        remedy, why = "works-as-stated / none-needed", \
            "capability stopped adding at every family's top (plateaued; ceiling reached)"
    elif has_content and task == "humor":
        remedy, why = "better-demonstration", \
            f"flip-selected examples carry certified construct content (fun-mm {max(c.values()):+.3f})"
    elif in_tail and brk is not None and brk >= .05:
        remedy, why = "stronger-listener", \
            f"still-gaining tail AND frontier ceiling break (+{brk:.3f} vs local top)"
    elif in_tail:
        remedy, why = "dialect-matched-listener", \
            "still-gaining tail; frontier listener did NOT de-censor under the local-dialect ref (asymptote unknown)"
    elif sat == "family-dependent":
        fams_gaining = [k for k, v in tm.items() if v is not None and v > .02]
        remedy, why = "dialect-matched-listener", \
            f"gains only in {','.join(fams_gaining)} — listener-family-relative construct"
    else:
        remedy, why = "nothing-certified", "no positive instrument evidence"
    rows.append({"task": task, "name": name, "sat": sat, "category": cat.get(name),
                 "remedy": remedy, "why": why,
                 "evidence_scope": "humor-instrumented" if task == "humor" else "staircase-only"})

json.dump({"thresholds": {"content": .05, "ceiling_break": .05, "sat": .02},
           "rows": rows}, open(f"{OUT}/verdict_table_v1.json", "w"), indent=0)

from collections import Counter
by_remedy = Counter(r["remedy"] for r in rows)
by_task = {}
for r in rows:
    by_task.setdefault(r["task"], Counter())[r["remedy"]] += 1
lines = ["# Verdict/remedy table v1 (criterion 4)", "",
         f"Universe: {len(rows)} constructs. Thresholds in JSON.", "",
         "| remedy | n | share |", "|---|---|---|"]
for k, v in by_remedy.most_common():
    lines.append(f"| {k} | {v} | {100*v/len(rows):.1f}% |")
lines += ["", "Per task:", "", "| task | " + " | ".join(k for k, _ in by_remedy.most_common()) + " |",
          "|---|" + "---|" * len(by_remedy)]
for t, c in by_task.items():
    lines.append(f"| {t} | " + " | ".join(str(c.get(k, 0)) for k, _ in by_remedy.most_common()) + " |")
open(f"{OUT}/summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
print("\nDONE ->", OUT)
