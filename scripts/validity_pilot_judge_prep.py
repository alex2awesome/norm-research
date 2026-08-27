"""Generate judge-pipeline subagent prompt files.

Phase 1: paraphrase generation. For each (metric, level), generate 4 NL
         paraphrases of the rubric (so we can test convergent validity).
Phase 2: scoring. For each (metric, level, paraphrase, datapoint-chunk),
         score every datapoint in the chunk 0-10. Single prompt scores
         many datapoints at once.

Output:
  runs/validity_pilot/<run>/judge/paraphrase_prompts/<metric>__<level>.txt
  runs/validity_pilot/<run>/judge/paraphrase_responses/   (subagent writes)
  runs/validity_pilot/<run>/judge/score_prompts/<metric>__<level>__p{0..3}__chunk{i}.txt   (after paraphrases land)
  runs/validity_pilot/<run>/judge/score_responses/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PARAPHRASE_SYSTEM = """You rewrite an evaluation rubric in 4 different ways while preserving its EXACT meaning. Each rewrite must be substantively interchangeable with the original — a competent reviewer would judge them as the same rubric.

Constraints:
- Same scope (don't broaden or narrow the rule).
- Different wording, sentence structure, and word choice.
- Each rewrite should stand alone (don't reference "the original").
- Output one JSON object only — no commentary."""


PARAPHRASE_USER_TEMPLATE = """Original rubric: "{rubric}"
{context}
Output VALID JSON ONLY:
{{"paraphrases": ["...", "...", "...", "..."]}}"""


SCORE_SYSTEM = """You score peer-review paper texts on a single evaluation rubric.

For each numbered TEXT below, output a single integer 0..10 indicating how well that text satisfies the rubric:
- 0 = the text clearly violates the rubric (or doesn't address it at all when it should)
- 5 = neutral / not applicable / cannot tell
- 10 = the text clearly satisfies the rubric

Be strict but fair. If a text doesn't naturally engage with the rubric's concern (e.g., a methods paper for a "data quality" rubric), score 5, not 0.

Output VALID JSON ONLY:
{"scores": [{"id": "d0", "score": <int>}, {"id": "d1", "score": <int>}, ...]}"""


SCORE_USER_TEMPLATE = """RUBRIC: "{rubric}"

TEXTS to score:
{texts}

Output the JSON now."""


def r1_rubric_text(m):
    return m["r1_focal_name"] + ". " + m["r1_focal_description"]


def r2_rubric_text(m):
    sub = "; ".join(m["r2_r1_member_names"][:6])
    return (m["r2_aspect_name"] + " — " + m["r2_aspect_description"]
            + f" (covers: {sub})")


def r1_paraphrase_context(m):
    samples = "\n".join(f"  - {s[:140]}" for s in m["r1_focal_samples"])
    return f"\nFor context, equivalent statements of this rubric:\n{samples}\n"


def r2_paraphrase_context(m):
    sub = "\n".join(f"  - {n[:140]}" for n in m["r2_r1_member_names"])
    return f"\nFor context, this aspect covers these specific sub-rules:\n{sub}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--phase", choices=["paraphrase", "score"], default="paraphrase")
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="Datapoints per scoring prompt (10 fits well in subagent context)")
    args = ap.parse_args()

    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())
    datapoints = json.loads((base / "datapoints.json").read_text())
    j = base / "judge"
    j.mkdir(exist_ok=True)

    if args.phase == "paraphrase":
        (j / "paraphrase_prompts").mkdir(exist_ok=True)
        (j / "paraphrase_responses").mkdir(exist_ok=True)
        manifest = []
        for m in metrics:
            for level in ("r1", "r2"):
                rubric = r1_rubric_text(m) if level == "r1" else r2_rubric_text(m)
                context = (r1_paraphrase_context(m) if level == "r1"
                           else r2_paraphrase_context(m))
                user = PARAPHRASE_USER_TEMPLATE.format(rubric=rubric,
                                                       context=context)
                key = f"{m['metric_id']}__{level}"
                prompt = PARAPHRASE_SYSTEM + "\n\n=== USER ===\n" + user
                (j / "paraphrase_prompts" / f"{key}.txt").write_text(prompt)
                manifest.append({"key": key, "metric_id": m["metric_id"],
                                 "level": level, "rubric_text": rubric})
        (j / "paraphrase_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"phase=paraphrase: wrote {len(manifest)} prompts to {j}/paraphrase_prompts/")
        return

    # phase == score: requires paraphrase responses to exist
    (j / "score_prompts").mkdir(exist_ok=True)
    (j / "score_responses").mkdir(exist_ok=True)
    manifest = []
    para_man = json.loads((j / "paraphrase_manifest.json").read_text())
    para_by_key = {x["key"]: x for x in para_man}

    chunks = [datapoints[i:i + args.chunk_size]
              for i in range(0, len(datapoints), args.chunk_size)]
    print(f"datapoints split into {len(chunks)} chunks of {args.chunk_size}")

    for m in metrics:
        for level in ("r1", "r2"):
            key = f"{m['metric_id']}__{level}"
            resp_path = j / "paraphrase_responses" / f"{key}.json"
            if not resp_path.exists():
                print(f"  SKIP {key}: paraphrase response missing")
                continue
            try:
                raw = resp_path.read_text().strip()
                m_md = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.S)
                if m_md: raw = m_md.group(1).strip()
                try: obj = json.loads(raw)
                except json.JSONDecodeError:
                    s, e = raw.find("{"), raw.rfind("}")
                    obj = json.loads(raw[s:e + 1])
                paraphrases = obj.get("paraphrases", [])
            except Exception as ex:
                print(f"  SKIP {key}: parse fail {ex}")
                continue

            # Include original + 4 paraphrases (= 5 total framings if all parse)
            rubric_orig = (r1_rubric_text(m) if level == "r1"
                           else r2_rubric_text(m))
            all_phrasings = [rubric_orig] + list(paraphrases)
            for pi, phrasing in enumerate(all_phrasings[:5]):
                for ci, chunk in enumerate(chunks):
                    texts_block = "\n\n".join(
                        f"--- {d['datapoint_id']} ---\n{d['text']}"
                        for d in chunk)
                    user = SCORE_USER_TEMPLATE.format(rubric=phrasing,
                                                      texts=texts_block)
                    prompt = SCORE_SYSTEM + "\n\n=== USER ===\n" + user
                    pkey = f"{m['metric_id']}__{level}__p{pi}__chunk{ci}"
                    (j / "score_prompts" / f"{pkey}.txt").write_text(prompt)
                    manifest.append({
                        "key": pkey, "metric_id": m["metric_id"],
                        "level": level, "paraphrase_idx": pi,
                        "chunk_idx": ci, "rubric": phrasing,
                        "datapoint_ids": [d["datapoint_id"] for d in chunk],
                    })
    (j / "score_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"phase=score: wrote {len(manifest)} score prompts to {j}/score_prompts/")


if __name__ == "__main__":
    main()
