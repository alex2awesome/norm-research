#!/usr/bin/env python3
"""Re-batch the UNANSWERED Phase-1 comparisons at a larger batch size.

Per-call latency (~560 s) rather than parallelism is the bottleneck, so fewer, bigger
calls is the only lever that helps. Completed work is preserved: already-answered
pair_ids are excluded, the old jobs stay in the manifest so their outputs keep counting,
and new tags cannot collide with old ones.
"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parent / "phase1"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 50

man = json.loads((R / "si_prompt_manifest.json").read_text())
comps = {c["pair_id"]: c for c in json.loads((R / "si_bt_comparisons.json").read_text())}
answered = set()
for f in (R / "out").glob("*.json"):
    try:
        for a in json.load(open(f)).get("answers", []):
            answered.add(str(a.get("pair_id", "")).strip())
    except Exception:
        pass
todo = [p for p in comps if p not in answered]
HOL = (R / "prompts" / man["jobs"][0]["tag"] + ".txt") if False else None
head = (R / "prompts" / (man["jobs"][0]["tag"] + ".txt")).read_text().split("PAIRS:\n\n")[0] + "PAIRS:\n\n"

jobs = [j for j in man["jobs"]]
new = 0
for b in range(0, len(todo), BATCH):
    ch = todo[b:b + BATCH]
    tag = f"rb_b{b//BATCH:03d}"
    body = "\n\n".join(
        f"--- pair_id={comps[p]['pair_id']} ---\nCONTEST PROMPT: {comps[p]['prompt']}\n"
        f"ENTRY A: {comps[p]['entry_A']}\nENTRY B: {comps[p]['entry_B']}" for p in ch)
    (R / "prompts" / f"{tag}.txt").write_text(head + body)
    jobs.append({"tag": tag, "question": "holistic", "n_pairs": len(ch),
                 "pair_ids": ch})
    new += 1
man["jobs"] = jobs
man["n_jobs"] = len(jobs)
(R / "si_prompt_manifest.json").write_text(json.dumps(man, indent=1))
print(f"answered {len(answered)} / {len(comps)}; {len(todo)} remaining -> {new} new jobs at batch {BATCH}")
