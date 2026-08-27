"""Run code-gen via Qwen-Coder on OpenRouter for cross-model subset.

Picks K R2 aspects (mix of high-|label correlation| and middle), top-1 R1 per
aspect, all 5 paraphrases = ~K × 5 prompts. Calls Qwen3-Coder async.

Output:
  runs/validity_full/<run>/codegen_responses_qwen/<key>.py
  runs/validity_full/<run>/qwen_codegen_manifest.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from openrouter import chat, make_client


CODEGEN_SYSTEM = """You write a single Python function `score(text: str) -> float` returning a value in [0.0, 1.0] indicating how well `text` satisfies a single evaluation rubric for peer-review papers.

Strict requirements:
- Return 1.0 if the rubric is fully satisfied, 0.0 if clearly violated, intermediate for partial. Return 0.5 if the rubric is not applicable to this text or you cannot tell.
- Use ONLY the Python standard library. Do NOT import third-party packages.
- The function must never raise. Wrap risky logic in try/except and return 0.5 on failure.
- Handle empty / very short text gracefully — return 0.5 if text is implausibly short.
- The function must run deterministically without network or filesystem access.

Output ONLY the Python code. No markdown fences, no commentary. Start with `def score` or with the `import` lines you need."""


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


async def gen_one(client, sem, model, entry, prompt_text):
    async with sem:
        try:
            user = prompt_text.split("\n=== USER ===\n", 1)[1].strip()
            raw = await chat(client, model, CODEGEN_SYSTEM, user, max_tokens=2500)
            code = extract_code(raw)
            return entry["key"], code, None
        except Exception as e:
            return entry["key"], "", str(e)[:200]


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="full_v1")
    ap.add_argument("--n-aspects", type=int, default=10)
    ap.add_argument("--model", default="qwen/qwen3-coder")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    base = Path(f"runs/validity_full/{args.run_name}")
    # Pick aspects: 5 top by |code-label| + 5 middle
    per_aspect = json.loads((base / "per_aspect_scores.json").read_text())
    per_aspect = [r for r in per_aspect if r["n_dp_code"] >= 10]
    per_aspect.sort(key=lambda r: -abs(r["rho_code_label"]))
    top = per_aspect[:5]
    middle = per_aspect[len(per_aspect)//2 - 2 : len(per_aspect)//2 + 3]
    chosen = top + middle
    chosen_ids = {r["aspect_id"] for r in chosen}
    print(f"chose {len(chosen)} aspects: top 5 |ρ| + 5 middle")

    # Map aspect -> first R1 metric
    aspects = json.loads((base / "r2_aspects.json").read_text())
    r1_metrics = json.loads((base / "r1_metrics.json").read_text())
    r1_by_id = {m["metric_id"]: m for m in r1_metrics}
    metric_for_aspect = {}
    for a in aspects:
        if a["aspect_id"] in chosen_ids:
            if a["r1_metric_ids"]:
                metric_for_aspect[a["aspect_id"]] = a["r1_metric_ids"][0]

    # Load codegen manifest to find paraphrase prompts
    cg_manifest = json.loads((base / "codegen_manifest.json").read_text())
    cg_by_key = {e["key"]: e for e in cg_manifest}

    # Pick all 5 paraphrases of the chosen R1s
    jobs = []
    for aid, mid in metric_for_aspect.items():
        for pi in range(5):
            key = f"{mid}__p{pi}"
            if key in cg_by_key:
                entry = cg_by_key[key]
                pf = base / "codegen_prompts" / f"{key}.txt"
                if pf.exists():
                    jobs.append((entry, pf.read_text()))
    print(f"jobs: {len(jobs)}")

    out_dir = base / "codegen_responses_qwen"
    out_dir.mkdir(exist_ok=True)
    manifest_out = []
    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[
        gen_one(client, sem, args.model, e, pt) for e, pt in jobs
    ])
    for key, code, err in results:
        if err:
            print(f"  ERR {key}: {err}")
        if code:
            (out_dir / f"{key}.py").write_text(code)
        manifest_out.append({"key": key, "model": args.model, "error": err})

    (base / "qwen_codegen_manifest.json").write_text(json.dumps(manifest_out, indent=1))
    n_ok = sum(1 for r in manifest_out if not r["error"])
    print(f"DONE: {n_ok}/{len(jobs)} OK")


if __name__ == "__main__":
    asyncio.run(amain())
