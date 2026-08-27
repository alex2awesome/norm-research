#!/usr/bin/env python3
"""Sonnet reference scoring for GEPA (label-free construct fidelity target).

For each of the 198 rubrics, ONE `claude -p --model sonnet` call scores all dev
comments (+3 anchors) against the ORIGINAL rubric construct on the 1.0/0.5/0.0/NA
protocol. The reference is frozen across GEPA rounds: it defines the construct;
Gemma-facing rubric text is optimized toward agreement with it.

Resumable: ref_scores.json is a {rubric_id: {item_id: score}} map, written after
every completed rubric. Concurrency via a small process pool of claude CLI calls.
"""
import json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE / "dev.jsonl"
RUBRICS = HERE.parent / "nc_rubrics.jsonl"
OUT = HERE / "ref_scores.json"
CONC = 3
TRUNC = 2000

ANCHORS = json.loads((HERE / "anchors.json").read_text()) if (HERE / "anchors.json").exists() else []

PROTO = ("Score how strongly each comment, on its own evidence, satisfies the criterion. "
         "Use EXACTLY one of: 1.0 (clearly satisfies), 0.5 (partially/borderline), "
         "0.0 (fails / cuts against), NA (no evidence bearing on this criterion). "
         "Judge the comment's quality as regulatory input, not whether you agree with its position.")

def build_prompt(rubric, items):
    parts = [
        "You are an expert regulatory analyst scoring PUBLIC COMMENTS on proposed federal rules "
        "against ONE quality criterion.",
        f"CRITERION: {rubric['name']}",
        f"DESCRIPTION: {rubric.get('description','')}",
        PROTO,
        "",
    ]
    for iid, txt in items:
        parts.append(f"=== COMMENT {iid} ===\n{txt[:TRUNC]}\n")
    parts.append(
        "Return ONLY a JSON object mapping every comment id to its score string, e.g. "
        '{"C01": "1.0", "C02": "NA", ...}. Include all ids. No other text.')
    return "\n".join(parts)

def call_sonnet(prompt, retries=3):
    for attempt in range(retries):
        try:
            r = subprocess.run(["claude", "-p", "--model", "sonnet"], input=prompt,
                               capture_output=True, text=True, timeout=600)
            m = re.search(r"\{.*\}", r.stdout, re.S)
            if m:
                return json.loads(m.group(0))
            print(f"  no-json (attempt {attempt}): {r.stdout[:120]!r}", flush=True)
            import time; time.sleep(10 * (attempt + 1))
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
    return None

def main():
    rubrics = [json.loads(l) for l in open(RUBRICS) if l.strip()]
    dev = [json.loads(l) for l in open(DEV)]
    items = [(f"C{i+1:02d}", r["text"]) for i, r in enumerate(dev)]
    id_map = {f"C{i+1:02d}": r["doc_id"] for i, r in enumerate(dev)}
    for j, a in enumerate(ANCHORS):
        iid = f"A{j+1:02d}"
        items.append((iid, a["text"]))
        id_map[iid] = a["doc_id"]

    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [r for r in rubrics if str(r["rubric_id"]) not in done]
    print(f"{len(todo)}/{len(rubrics)} rubrics to score, {len(items)} items each", flush=True)

    def work(r):
        res = call_sonnet(build_prompt(r, items))
        return r, res

    with ThreadPoolExecutor(CONC) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for k, f in enumerate(as_completed(futs)):
            r, res = f.result()
            if res is None:
                print(f"[{r['rubric_id']}] FAILED", flush=True)
                continue
            done[str(r["rubric_id"])] = {id_map[c]: v for c, v in res.items() if c in id_map}
            OUT.write_text(json.dumps(done))
            if (k + 1) % 10 == 0:
                print(f"progress {k+1}/{len(todo)}", flush=True)
    n_ok = len(done)
    print(f"REF_DONE {n_ok}/{len(rubrics)} -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
