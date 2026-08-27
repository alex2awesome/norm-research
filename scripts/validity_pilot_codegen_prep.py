"""Generate code-gen subagent prompt files.

For each metric × level (R1, R2) × trial, emit one subagent prompt that asks
for `def score(text: str) -> float in [0,1]` using stdlib only. Each model
will produce its own variant per trial.

Output:
  runs/validity_pilot/<run>/codegen/prompts/<metric>__<level>__<model>__t<trial>.txt
  runs/validity_pilot/<run>/codegen/responses/   (subagent writes here)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CODEGEN_SYSTEM = """You write a single Python function `score(text: str) -> float` returning a value in [0.0, 1.0] indicating how well `text` satisfies a single evaluation rubric for peer-review papers.

Strict requirements:
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, intermediate for partial. Return 0.5 if the rubric is not applicable to this text or you cannot tell.
- Use ONLY the Python standard library. Do NOT import third-party packages (no numpy, regex, requests, nltk, etc.).
- The function must never raise. Wrap any risky logic in try/except and return 0.5 on failure.
- Handle empty / very short text gracefully — return 0.5 if text is implausibly short.
- The function must run deterministically without network or filesystem access.

Output ONLY the Python code. No markdown fences, no commentary. Start with `def score` or with the `import` lines you need."""


def build_r1_user(metric):
    """User message framing the metric at R1 (specific rule) level."""
    samples_block = "\n".join(f"  - {s[:140]}" for s in metric["r1_focal_samples"])
    return (
        f"Rubric NAME: \"{metric['r1_focal_name']}\"\n"
        f"DESCRIPTION: {metric['r1_focal_description']}\n\n"
        f"Examples of equivalent rubric statements (same underlying rule, "
        f"different wording):\n{samples_block}\n\n"
        f"Write the score(text: str) function.")


def build_r2_user(metric):
    """User message framing the metric at R2 (aspect) level — includes
    names of sub-R1 families to clarify what the aspect covers."""
    sub_block = "\n".join(f"  - {n[:140]}" for n in metric["r2_r1_member_names"])
    return (
        f"Rubric ASPECT NAME: \"{metric['r2_aspect_name']}\"\n"
        f"ASPECT DESCRIPTION: {metric['r2_aspect_description']}\n\n"
        f"This aspect covers the following specific sub-rules:\n{sub_block}\n\n"
        f"Write a score(text: str) function that returns 1.0 only if the text "
        f"substantively addresses this aspect (i.e., one or more of these "
        f"sub-rules is well-satisfied).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="smoke")
    ap.add_argument("--models", nargs="+", default=["claude"],
                    help="claude / llama / qwen (currently only claude is "
                         "wired via subagents; llama/qwen need sk3 launch).")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    base = Path(f"runs/validity_pilot/{args.run_name}")
    metrics = json.loads((base / "metrics.json").read_text())
    cg = base / "codegen"
    (cg / "prompts").mkdir(parents=True, exist_ok=True)
    (cg / "responses").mkdir(parents=True, exist_ok=True)

    n_written = 0
    manifest = []
    for m in metrics:
        for level in ("r1", "r2"):
            user = build_r1_user(m) if level == "r1" else build_r2_user(m)
            for model in args.models:
                for t in range(args.trials):
                    key = f"{m['metric_id']}__{level}__{model}__t{t}"
                    prompt = (CODEGEN_SYSTEM
                              + "\n\n=== USER ===\n" + user)
                    (cg / "prompts" / f"{key}.txt").write_text(prompt)
                    manifest.append({
                        "key": key,
                        "metric_id": m["metric_id"],
                        "level": level,
                        "model": model,
                        "trial": t,
                        "metric_name": (m["r1_focal_name"] if level == "r1"
                                        else m["r2_aspect_name"]),
                    })
                    n_written += 1

    (cg / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {n_written} code-gen prompts -> {cg}/prompts/")
    print(f"manifest -> {cg}/manifest.json")


if __name__ == "__main__":
    main()
