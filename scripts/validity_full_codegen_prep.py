"""Generate code-gen prompts for all R1 metrics × 5 paraphrases.

Loads:
  runs/validity_full/<run>/r1_metrics.json
  runs/validity_full/<run>/paraphrase_responses/r1__<metric_id>.json
    {"paraphrases": ["...", "...", "...", "...", "..."]}

Writes:
  runs/validity_full/<run>/codegen_prompts/<metric_id>__p<i>.txt
  runs/validity_full/<run>/codegen_manifest.json

(Note: paraphrase[0..4] gives 5 variants. The ORIGINAL rubric is the 6th option.
For simplicity we use only the 5 paraphrases — paraphrase index = framing index.)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CODEGEN_SYSTEM = """You write a single Python function `score(text: str) -> float` returning a value in [0.0, 1.0] indicating how well `text` satisfies a single evaluation rubric for peer-review papers.

Strict requirements:
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, intermediate for partial. Return 0.5 if the rubric is not applicable to this text or you cannot tell.
- Use ONLY the Python standard library. Do NOT import third-party packages (no numpy, regex, requests, nltk, etc.).
- The function must never raise. Wrap any risky logic in try/except and return 0.5 on failure.
- Handle empty / very short text gracefully — return 0.5 if text is implausibly short.
- The function must run deterministically without network or filesystem access.

Output ONLY the Python code. No markdown fences, no commentary. Start with `def score` or with the `import` lines you need."""


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
    args = ap.parse_args()
    base = Path(f"runs/validity_full/{args.run_name}")

    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    para_dir = base / "paraphrase_responses"
    out_dir = base / "codegen_prompts"
    out_dir.mkdir(exist_ok=True)

    manifest = []
    n_skip = 0
    for m in r1_metrics:
        pr_path = para_dir / f"r1__{m['metric_id']}.json"
        if not pr_path.exists():
            n_skip += 1
            continue
        try:
            paraphrases = parse_paraphrases(pr_path)
        except Exception as e:
            print(f"  skip {m['metric_id']}: {e}")
            n_skip += 1
            continue
        # Use paraphrases as the rubric framings
        for pi, phr in enumerate(paraphrases[:5]):
            user = (f'Rubric: "{phr}"\n\nWrite the score(text: str) function.')
            key = f"{m['metric_id']}__p{pi}"
            prompt = CODEGEN_SYSTEM + "\n\n=== USER ===\n" + user
            (out_dir / f"{key}.txt").write_text(prompt)
            manifest.append({
                "key": key,
                "metric_id": m["metric_id"],
                "parent_aspect_id": m["parent_aspect_id"],
                "paraphrase_idx": pi,
                "rubric": phr,
            })

    (base / "codegen_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} code-gen prompts -> {out_dir}/")
    print(f"  ({n_skip} R1 metrics skipped — paraphrase response missing)")


if __name__ == "__main__":
    main()
