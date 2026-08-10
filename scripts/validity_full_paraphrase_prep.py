"""Generate paraphrase prompts for R1 metrics and R2 aspects.

For each rubric statement, ask LLM to write 5 NL paraphrases (same meaning,
different wording). Used downstream as separate rubric framings for both
code-gen (different prompt) and judging (different LLM-judge prompt).

Output:
  runs/validity_full/<run>/paraphrase_prompts/<metric_id>.txt
  runs/validity_full/<run>/paraphrase_responses/   (Llama writes here)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PARAPHRASE_SYSTEM = """You rewrite an evaluation rubric in 5 different ways while preserving its EXACT meaning. Each rewrite must be substantively interchangeable — a competent reviewer would judge them as the same rubric.

Constraints:
- Same scope (don't broaden or narrow).
- Different wording, sentence structure, and word choice.
- Each rewrite must stand alone (don't reference "the original").

Output VALID JSON ONLY:
{"paraphrases": ["...", "...", "...", "...", "..."]}"""


def r1_user(metric):
    samples_block = "\n".join(f"  - {s[:140]}" for s in metric.get("samples", []))
    ctx = f"\nContext — equivalent statements of this rubric:\n{samples_block}\n" if metric.get("samples") else ""
    return (f'Original rubric: "{metric["name"]}. {metric.get("description","")}"\n{ctx}\n'
            f'Generate 5 paraphrases as VALID JSON.')


def r2_user(aspect):
    return (f'Original rubric (thematic aspect): "{aspect["name"]} — {aspect.get("description","")}"\n\n'
            f'Generate 5 paraphrases as VALID JSON.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    args = ap.parse_args()
    base = Path(f"runs/validity_full/{args.run_name}")
    (base / "paraphrase_prompts").mkdir(exist_ok=True)
    (base / "paraphrase_responses").mkdir(exist_ok=True)

    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    r2_aspects = json.loads((base / "r2_aspects.json").read_text())

    manifest = []
    for m in r1_metrics:
        prompt = PARAPHRASE_SYSTEM + "\n\n=== USER ===\n" + r1_user(m)
        key = f"r1__{m['metric_id']}"
        (base / "paraphrase_prompts" / f"{key}.txt").write_text(prompt)
        manifest.append({"key": key, "kind": "r1", "metric_id": m["metric_id"]})

    for a in r2_aspects:
        prompt = PARAPHRASE_SYSTEM + "\n\n=== USER ===\n" + r2_user(a)
        key = f"r2__{a['aspect_id']}"
        (base / "paraphrase_prompts" / f"{key}.txt").write_text(prompt)
        manifest.append({"key": key, "kind": "r2", "aspect_id": a["aspect_id"]})

    (base / "paraphrase_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} paraphrase prompts ({len(r1_metrics)} R1 + {len(r2_aspects)} R2)")


if __name__ == "__main__":
    main()
