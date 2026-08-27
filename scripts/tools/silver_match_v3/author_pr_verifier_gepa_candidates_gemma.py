#!/usr/bin/env python3
"""Author three optimize-only PR verifier prompt candidates with Gemma-4.

This is a bounded GEPA mutation step, not prompt selection.  The input freeze is
written before model initialization; detailed examples are sampled exclusively
from the frozen optimize-role packet and stripped of identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


THEMES = {
    "predicate_first": (
        "Make explicit-normative-predicate detection a hard fail-closed gate. "
        "Optimize precision on factual/topic-only negatives without discarding clear breaches."
    ),
    "leaf_contrast": (
        "Make proposal-versus-strongest-alternative operational-nucleus contrast the core. "
        "Optimize exact-leaf precision and order robustness."
    ),
    "proof_obligations": (
        "Use compact sequential proof obligations for explicit criterion, exact proposal entailment, "
        "and unique victory over alternatives. Optimize calibrated typed abstention."
    ),
}

REQUIRED_DECISIONS = (
    "CONFIRM_MATCH",
    "AMBIGUOUS_MATCH",
    "BETTER_CANDIDATE",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rank(seed: int, uid: str) -> str:
    return hashlib.sha256(f"{seed}\0{uid}".encode()).hexdigest()


def _extract_prompt(raw: str) -> str:
    text = str(raw or "").strip()
    if "<PROMPT>" in text and "</PROMPT>" in text:
        text = text.split("<PROMPT>", 1)[1].split("</PROMPT>", 1)[0].strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    lower = text.lower()
    if (
        not 900 <= len(text) <= 16000
        or any(value not in text for value in REQUIRED_DECISIONS)
        or '"high"' not in text
        or '"medium"' not in text
        or '"low"' not in text
        or "24 words" not in lower
        or "json" not in lower
        or "proposal" not in lower
        or "strongest" not in lower
    ):
        raise ValueError("Gemma-authored prompt violates the verifier interface")
    return text.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--training-examples", required=True)
    parser.add_argument("--aggregate-taxonomy", required=True)
    parser.add_argument("--base-prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=2026071333)
    parser.add_argument("--examples-per-class", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "training_report",
            "training_examples",
            "aggregate_taxonomy",
            "base_prompt",
        )
    }
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(paths["training_report"].read_text(encoding="utf-8"))
    examples = list(read_jsonl(paths["training_examples"]))
    taxonomy = json.loads(paths["aggregate_taxonomy"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "FROZEN_OPTIMIZE_ONLY_AUTHORSHIP_EVIDENCE"
        or report.get("task") != "press-releases"
        or int(report.get("count", -1)) != len(examples)
        or (report.get("outputs") or {}).get("examples", {}).get("sha256")
        != sha256_file(paths["training_examples"])
        or taxonomy.get("status")
        != "FROZEN_IDENTITY_FREE_AGGREGATES_FOR_OPTIMIZE_ONLY_PROMPT_AUTHORING"
        or (taxonomy.get("contracts") or {}).get("consumed_dev_text_included") is not False
    ):
        raise ValueError("invalid optimize author evidence or aggregate taxonomy")

    by_target: dict[str, list[dict[str, Any]]] = {"CONFIRM_MATCH": [], "REJECT": []}
    for row in examples:
        target = str(row.get("target") or "")
        use = row.get("use_contract") or {}
        if (
            target not in by_target
            or row.get("gepa_role") != "optimize"
            or row.get("predeclared_split") != "train"
            or use.get("verifier_prompt_authorship_or_gepa_optimize_only") is not True
            or use.get("verifier_selection") is not False
        ):
            raise ValueError("author example is not optimize-only")
        by_target[target].append(row)
    if any(len(rows) < args.examples_per_class for rows in by_target.values()):
        raise ValueError("insufficient examples per class")

    selected: list[dict[str, Any]] = []
    for target, rows in sorted(by_target.items()):
        chosen = sorted(
            rows, key=lambda row: (_rank(args.seed, str(row["norm_uid"])), row["norm_uid"])
        )[: args.examples_per_class]
        for row in chosen:
            selected.append(
                {
                    "norm": row.get("norm"),
                    "context": row.get("context"),
                    "proposal": row.get("proposal"),
                    "gold": row.get("gold"),
                    "target": target,
                    "metric_cards": row.get("metric_cards"),
                }
            )

    output.mkdir(parents=True)
    sanitized_path = output / "SANITIZED_OPTIMIZE_EXAMPLES.json"
    _write_json(
        sanitized_path,
        {
            "identity_fields_removed": ["norm_uid", "source_group"],
            "count": len(selected),
            "target_counts": {
                target: sum(row["target"] == target for row in selected)
                for target in sorted(by_target)
            },
            "examples": selected,
        },
    )
    freeze = {
        "schema_version": "silver-match-v3-pr-gemma-verifier-gepa-author-freeze-v1",
        "status": "FROZEN_BEFORE_GEMMA4_PROMPT_MUTATION",
        "task": "press-releases",
        "model": args.model,
        "seed": args.seed,
        "variant_themes": THEMES,
        "variant_count": len(THEMES),
        "sampled_optimize_examples": len(selected),
        "contracts": {
            "detailed_examples_are_optimize_only": True,
            "consumed_dev_evidence_is_aggregate_identity_free_only": True,
            "fresh_test_not_drawn_or_read": True,
            "candidates_are_not_selected_by_this_step": True,
            "codex_judges_candidates_after_optimize_scoring": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "sanitized_examples": {
            "path": str(sanitized_path),
            "sha256": sha256_file(sanitized_path),
        },
        "inference": {
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "temperature": 0.65,
            "top_p": 0.90,
        },
    }
    freeze_path = output / "INPUT_FREEZE.json"
    _write_json(freeze_path, freeze)

    base_prompt = paths["base_prompt"].read_text(encoding="utf-8")
    common = (
        "You are performing one bounded GEPA prompt-mutation round for a precision-first "
        "press-release norm-to-metric verifier. Author a complete replacement SYSTEM PROMPT, "
        "not an analysis. The downstream item renderer supplies a HUMAN STATEMENT, evidence "
        "passage, one PROPOSED METRIC card, and STRONGEST ALTERNATIVES. Preserve exactly this "
        "output interface: one JSON object with exactly decision, metric_id, confidence, reason. "
        "Allowed decisions are " + ", ".join(REQUIRED_DECISIONS) + ". confidence must be exactly "
        '"high", "medium", or "low". reason is at most 24 words. CONFIRM_MATCH metric_id equals '
        "the proposal; BETTER_CANDIDATE metric_id is a supplied alternative; otherwise null. "
        "Precision dominates yield. Do not mention training examples, GEPA, labels, or this task. "
        "Return only <PROMPT>complete prompt text</PROMPT>.\n\n"
        "IDENTITY-FREE AGGREGATE FAILURE TAXONOMY:\n"
        + json.dumps(taxonomy, ensure_ascii=False, sort_keys=True)
        + "\n\nOPTIMIZE-ONLY LABELED EXAMPLES:\n"
        + json.dumps(selected, ensure_ascii=False, sort_keys=True)
        + "\n\nCURRENT BASE PROMPT TO IMPROVE:\n"
        + base_prompt
    )
    prompts = []
    for name, theme in THEMES.items():
        prompts.append(
            "Variant theme: " + name + ". " + theme + "\n\n" + common
        )

    from vllm import LLM, SamplingParams  # imported only after the input freeze

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    rendered = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You author rigorous classifier prompts."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    values = llm.generate(
        rendered,
        SamplingParams(
            temperature=0.65,
            top_p=0.90,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )
    candidates: dict[str, Any] = {}
    for name, result in zip(THEMES, values, strict=True):
        raw = result.outputs[0].text
        raw_path = output / f"{name}.raw.txt"
        raw_path.write_text(raw, encoding="utf-8")
        prompt_text = _extract_prompt(raw)
        prompt_path = output / f"{name}.prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")
        candidates[name] = {
            "theme": THEMES[name],
            "raw": {"path": str(raw_path), "sha256": sha256_file(raw_path)},
            "prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
        }
    result = {
        "schema_version": "silver-match-v3-pr-gemma-verifier-gepa-author-output-v1",
        "status": "FROZEN_THREE_GEMMA4_CANDIDATES_OPTIMIZE_ONLY",
        "input_freeze": {
            "path": str(freeze_path),
            "sha256": sha256_file(freeze_path),
        },
        "candidates": candidates,
        "fresh_test_drawn_or_read": False,
    }
    result_path = output / "OUTPUT_FREEZE.json"
    _write_json(result_path, result)
    print(json.dumps({**result, "output_freeze_sha256": sha256_file(result_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
