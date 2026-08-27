"""Generate R1-level judge prompts (parallel to the R2-level run).

For each R1 metric × 5 paraphrases × N datapoint chunks, write a judge prompt.
Same SCORE_SYSTEM as R2 — just different rubric framing.

Output:
  runs/validity_full/<run>/judge_r1_prompts/<r1_metric_id>__p<i>__c<chunk>.txt
  runs/validity_full/<run>/judge_r1_manifest.json

Scope cap: --max-r1 (default 0 = all). Use --max-r1 50 for a subset test.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCORE_SYSTEM = """You score peer-review paper texts on a single evaluation rubric.

For each numbered TEXT below, output a single integer 0..10 indicating how well that text satisfies the rubric:
- 0 = the text clearly violates the rubric (or doesn't address it at all when it should)
- 5 = neutral / not applicable / cannot tell
- 10 = the text clearly satisfies the rubric

Be strict but fair. If a text doesn't naturally engage with the rubric's concern, score 5, not 0.

Output VALID JSON ONLY:
{"scores": [{"id": "d0", "score": <int>}, {"id": "d1", "score": <int>}, ...]}"""


def parse_paraphrases(p):
    raw = p.read_text().strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
    if m: raw = m.group(1).strip()
    try: obj = json.loads(raw)
    except json.JSONDecodeError:
        s, e = raw.find("{"), raw.rfind("}")
        obj = json.loads(raw[s:e + 1])
    return obj.get("paraphrases", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--max-r1", type=int, default=0, help="0 = all")
    ap.add_argument("--select-aspects", nargs="*", default=None,
                    help="Limit to R1 children of these aspect_ids (e.g., a102 a131 ...)")
    args = ap.parse_args()

    base = Path(f"runs/validity_full/{args.run_name}")
    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    if args.select_aspects:
        sel = set(args.select_aspects)
        r1_metrics = [m for m in r1_metrics if m["parent_aspect_id"] in sel]
        print(f"filtered to {len(r1_metrics)} R1 metrics by aspect")
    if args.max_r1 > 0:
        r1_metrics = r1_metrics[:args.max_r1]

    datapoints = json.loads((base / "datapoints.json").read_text())
    para_dir = base / "paraphrase_responses"
    out_dir = base / "judge_r1_prompts"
    out_dir.mkdir(exist_ok=True)
    (base / "judge_r1_responses").mkdir(exist_ok=True)

    chunks = [datapoints[i:i + args.chunk_size]
              for i in range(0, len(datapoints), args.chunk_size)]
    print(f"datapoints split into {len(chunks)} chunks of {args.chunk_size}")

    manifest = []
    n_skip = 0
    for m in r1_metrics:
        pr_path = para_dir / f"r1__{m['metric_id']}.json"
        if not pr_path.exists():
            n_skip += 1; continue
        try:
            paraphrases = parse_paraphrases(pr_path)
        except Exception:
            n_skip += 1; continue
        for pi, phr in enumerate(paraphrases[:5]):
            for ci, chunk in enumerate(chunks):
                texts_block = "\n\n".join(
                    f"--- {d['datapoint_id']} ---\n{d['text']}"
                    for d in chunk)
                user = (f'RUBRIC: "{phr}"\n\nTEXTS to score:\n{texts_block}\n\n'
                        f'Output the JSON now.')
                key = f"{m['metric_id']}__p{pi}__c{ci}"
                prompt = SCORE_SYSTEM + "\n\n=== USER ===\n" + user
                (out_dir / f"{key}.txt").write_text(prompt)
                manifest.append({
                    "key": key,
                    "metric_id": m["metric_id"],
                    "parent_aspect_id": m["parent_aspect_id"],
                    "paraphrase_idx": pi,
                    "chunk_idx": ci,
                    "rubric": phr,
                    "datapoint_ids": [d["datapoint_id"] for d in chunk],
                })

    (base / "judge_r1_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} R1 judge prompts -> {out_dir}/")
    print(f"  ({n_skip} R1 metrics skipped)")


if __name__ == "__main__":
    main()
