#!/usr/bin/env python3
"""Run a blind concept-identity judge over a `<tag>_bmerge<T>_packet.json` and emit the
verdict file `species_merge.py apply --verdicts` expects.

WHY THIS EXISTS.  The Sonnet-era merges (aops_curation, jokes_community, mathse_*) were
judged ad hoc -- the packet went to a subagent and the verdict JSON was saved by hand, so
there was no runner and no recorded prompt for the bmerge pass specifically.  The
Track-A certificate backfill (2026-08-11) has to judge ~8 campaigns reproducibly and on a
different judge family, so the pass is scripted here: one preamble, one parser, one
coverage gate, resume-by-output-file.

PROMPT PROVENANCE.  The preamble is the one already on disk from the cap_finalist r5 and
cap_crowd r4 bmerge passes (`<tag>_bmerge_prompt.txt`), reproduced verbatim so this
runner is the same instrument those rounds used.  The only addition is the pair block
renderer, which formats the packet exactly as those files show it.

JUDGE FAMILY CAVEAT.  The backfill runs gpt-5.6-sol + gpt-5.6-luna, i.e. TWO LEGS OF ONE
FAMILY.  That is the same hive-mind caveat the Sonnet-Sonnet merges already carry
(claude-sonnet-5 on both legs) and must be recorded symmetrically wherever either is
quoted: "both judges must say SAME" is a weaker independence claim when the judges share
a family.  Judge identity is written into every output file.

Usage:
  python run_bmerge_judges.py --packet <path>_bmerge<T>_packet.json \
      --model gpt-5.6-sol --out <path>_bmerge<T>_judge_sol.json [--chunk 0]

`--chunk 0` (default) sends every pair in one prompt, which is what the Sonnet legs saw.
Any pair the judge omits is automatically re-asked in a follow-up prompt containing only
the missing pairs; the output records how many passes were needed.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

PREAMBLE = (
    "You are a blind concept-identity judge. Below is a list of PAIRS of criterion "
    "descriptions. For each pair decide whether X and Y name the SAME underlying concept "
    "or DIFFERENT concepts, judging the two descriptions on their own text. You are not "
    "told who wrote anything and you must not try to guess.\n\n"
    "OUTPUT. Emit exactly one JSON object and nothing else:\n"
    '{"judge": "<your model name>", "verdicts": [{"pair_id": "...", "verdict": "SAME" or '
    '"DIFFERENT"}, ...]}\n'
    "One entry per pair, using the pair_id shown, covering EVERY pair listed.\n\n"
    "PAIRS:\n"
)


def render(items) -> str:
    out = []
    for it in items:
        out.append(
            f"\n--- pair_id={it['pair_id']} ---\n"
            f"X NAME: {it['X_name']}\nX: {it['X_desc']}\n"
            f"Y NAME: {it['Y_name']}\nY: {it['Y_desc']}\n"
        )
    return "".join(out)


def extract_json(raw: str):
    """Codex prints the answer then a `tokens used` trailer; take the last JSON object."""
    depth, start, best = 0, None, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                cand = raw[start:i + 1]
                if '"verdicts"' in cand:
                    best = cand
    if best is None:
        return None
    try:
        return json.loads(best)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", best))


def ask(model, prompt, wd: Path, effort="high", timeout=3600):
    wd.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--model", model, "-c", f"model_reasoning_effort={effort}",
           "-s", "read-only", "--skip-git-repo-check", "--cd", str(wd), "-"]
    t0 = time.time()
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    print(f"    [{model}] {time.time() - t0:.0f}s rc={p.returncode} "
          f"stdout={len(p.stdout)}B", flush=True)
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=0, help="0 = all pairs in one prompt")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--scratch", default="/private/tmp/claude-502/"
                    "-Users-spangher-Projects-stanford-research-norm-research/"
                    "4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/bmerge")
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists():
        print(f"{out.name} already exists, skip")
        return

    pk = json.loads(Path(a.packet).read_text())
    items = list(pk["items"]) + list(pk["anchors"])
    want = [it["pair_id"] for it in items]
    by_id = {it["pair_id"]: it for it in items}
    wd = Path(a.scratch) / out.stem
    verdicts, passes = {}, []

    todo = list(want)
    for attempt in range(4):
        if not todo:
            break
        batches = ([todo] if a.chunk <= 0
                   else [todo[i:i + a.chunk] for i in range(0, len(todo), a.chunk)])
        print(f"  pass {attempt + 1}: {len(todo)} pairs in {len(batches)} prompt(s)", flush=True)
        for b in batches:
            raw = ask(a.model, PREAMBLE + render([by_id[i] for i in b]), wd, a.effort)
            (wd / f"raw_pass{attempt + 1}_{b[0]}.txt").write_text(raw)
            obj = extract_json(raw)
            if not obj:
                print("    no JSON parsed from this prompt", flush=True)
                continue
            for v in obj.get("verdicts", []):
                pid, vd = v.get("pair_id"), str(v.get("verdict", "")).strip().upper()
                if pid in by_id and vd in ("SAME", "DIFFERENT"):
                    verdicts[pid] = {"pair_id": pid, "verdict": vd}
        passes.append({"pass": attempt + 1, "asked": len(todo), "have": len(verdicts)})
        todo = [p for p in want if p not in verdicts]

    if todo:
        raise SystemExit(f"COVERAGE FAIL: {len(todo)} pairs unjudged after 4 passes: {todo[:10]}")

    res = {"judge": a.model,
           "judge_family": "openai/codex",
           "single_family_pair_caveat": "the backfill's two legs (gpt-5.6-sol, gpt-5.6-luna) "
                                        "share a family, exactly as the Sonnet-Sonnet merges "
                                        "do; 'both judges SAME' is a weaker independence "
                                        "claim than a cross-family pair would give",
           "packet": Path(a.packet).name, "reasoning_effort": a.effort,
           "prompt_provenance": "verbatim cap_finalist_r5 / cap_crowd_r4 bmerge preamble",
           "n_pairs": len(want), "passes": passes,
           "verdicts": [verdicts[p] for p in want]}
    out.write_text(json.dumps(res, indent=1))
    n_same = sum(1 for v in res["verdicts"] if v["verdict"] == "SAME")
    print(f"wrote {out.name}: {len(want)} verdicts, {n_same} SAME ({n_same / len(want):.1%})")


if __name__ == "__main__":
    main()
