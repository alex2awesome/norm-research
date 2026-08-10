#!/usr/bin/env python3
"""Batch-score all frozen PR Gemma verifier prompt variants on optimize92.

This is the batch-equivalent of six ``verify_gemma`` runs (three prompts by
original/hashed order), but initializes Gemma once.  It freezes every input and
inference setting before importing vLLM and reports scores without selecting a
prompt.  Select, test, MI, and outcome artifacts are not accepted as inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .adjudicate_gemma import load_inputs, ordered_candidates, prompt_sha256, scan_candidate_input
from .common import read_jsonl, sha256_file, write_jsonl
from .score_verifier_calibration import safe_rate, wilson_interval
from .verify_gemma import build_verification_prompt, parse_response


ORDERS = ("original", "hashed")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    values = {str(row.get("norm_uid") or ""): row for row in rows}
    if not rows or "" in values or len(values) != len(rows):
        raise ValueError(f"invalid UID coverage: {path}")
    return values


def _score(
    targets: dict[str, dict[str, Any]],
    primary: dict[str, dict[str, Any]],
    original: dict[str, dict[str, Any]],
    hashed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    retained = retained_true = decision_agree = exact_agree = 0
    positives = sum(row.get("target") == "CONFIRM_MATCH" for row in targets.values())
    negatives = len(targets) - positives
    for uid, target in targets.items():
        proposal = str(primary[uid]["metric_id"])
        left, right = original[uid], hashed[uid]
        decision_agree += left.get("decision") == right.get("decision")
        exact_agree += (left.get("decision"), left.get("metric_id")) == (
            right.get("decision"),
            right.get("metric_id"),
        )
        keep = (
            left.get("decision") == right.get("decision") == "CONFIRM_MATCH"
            and str(left.get("metric_id")) == str(right.get("metric_id")) == proposal
            and left.get("confidence") == right.get("confidence") == "high"
            and left.get("parse_error") is None
            and right.get("parse_error") is None
        )
        retained += keep
        retained_true += keep and target.get("target") == "CONFIRM_MATCH"
    false_retained = retained - retained_true
    return {
        "n": len(targets),
        "target_counts": {"CONFIRM_MATCH": positives, "REJECT": negatives},
        "policy": "two_order_exact_high",
        "retained": retained,
        "retained_true": retained_true,
        "false_retained": false_retained,
        "retained_precision": safe_rate(retained_true, retained),
        "retained_precision_wilson_95": wilson_interval(retained_true, retained),
        "retained_recall_of_correct_proposals": safe_rate(retained_true, positives),
        "wrong_proposal_rejection_rate": safe_rate(negatives - false_retained, negatives),
        "wrong_proposal_rejection_wilson_95": wilson_interval(
            negatives - false_retained, negatives
        ),
        "order_stability": {
            "decision_agreement": safe_rate(decision_agree, len(targets)),
            "exact_decision_and_id_agreement": safe_rate(exact_agree, len(targets)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--balanced-report", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--primary", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--author-output-freeze", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-alternatives", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    args = parser.parse_args()

    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in (
            "manifest",
            "balanced_report",
            "candidates",
            "primary",
            "targets",
            "truth",
            "author_output_freeze",
        )
    }
    if any(
        token in str(path).lower()
        for path in paths.values()
        for token in ("/select", "/test", "/mi_", "/outcome")
    ):
        raise ValueError("select/test/MI/outcome paths are forbidden")
    output = Path(args.output_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    balanced = json.loads(paths["balanced_report"].read_text(encoding="utf-8"))
    author = json.loads(paths["author_output_freeze"].read_text(encoding="utf-8"))
    if (
        balanced.get("schema_version") != "silver-match-v3-balanced-verifier-gepa-train-v1"
        or int(balanced.get("count") or -1) != 92
        or int(balanced.get("positive_count") or -1) != 46
        or int(balanced.get("negative_count") or -1) != 46
        or author.get("status") != "FROZEN_THREE_GEMMA4_CANDIDATES_OPTIMIZE_ONLY"
        or author.get("fresh_test_drawn_or_read") is not False
        or set(author.get("candidates") or {})
        != {"predicate_first", "leaf_contrast", "proof_obligations"}
    ):
        raise ValueError("invalid optimize92 or Gemma author freeze")
    for name in ("candidates", "primary", "targets", "truth"):
        if (balanced.get("output_hashes") or {}).get(name) != sha256_file(paths[name]):
            raise ValueError(f"balanced optimize output drift: {name}")

    targets = _index(paths["targets"])
    primary = _index(paths["primary"])
    truth = _index(paths["truth"])
    if set(targets) != set(primary) or set(targets) != set(truth) or len(targets) != 92:
        raise ValueError("optimize92 target/primary/truth coverage drift")
    if any(
        row.get("target") not in {"CONFIRM_MATCH", "REJECT"}
        for row in targets.values()
    ):
        raise ValueError("invalid paired verifier target")
    if any(
        row.get("gepa_role") != "optimize"
        or row.get("split") != "train"
        or row.get("task") != "press-releases"
        or row.get("prompt_gradient_eligible") is not True
        for row in truth.values()
    ):
        raise ValueError("scoring truth is not frozen optimize-only evidence")
    if any(row.get("decision") != "MATCH" for row in primary.values()):
        raise ValueError("every verifier proposal must be MATCH")

    prompt_paths = {
        name: Path(meta["prompt"]["path"]).resolve()
        for name, meta in sorted((author.get("candidates") or {}).items())
    }
    if any(
        not path.is_file()
        or sha256_file(path) != author["candidates"][name]["prompt"]["sha256"]
        for name, path in prompt_paths.items()
    ):
        raise ValueError("Gemma-authored prompt artifact drift")
    prompts = {name: path.read_text(encoding="utf-8") for name, path in prompt_paths.items()}

    candidate_rows = _index(paths["candidates"])
    if set(candidate_rows) != set(targets):
        raise ValueError("candidate slates do not exactly cover optimize92")
    corpora, _ = scan_candidate_input(paths["candidates"], done=set(), shard_id=0, num_shards=1)
    manifest, norms_by_corpus, banks = load_inputs(paths["manifest"], corpora)
    work: list[dict[str, Any]] = []
    for variant, system_prompt in sorted(prompts.items()):
        for order in ORDERS:
            for uid in sorted(targets):
                candidate = candidate_rows[uid]
                proposal = primary[uid]
                corpus = str(candidate["corpus"])
                norm = norms_by_corpus[corpus][uid]
                bank = banks[norm["task"]]
                proposal_id = str(proposal["metric_id"])
                values = [str(row["metric_id"]) for row in candidate.get("candidates") or []]
                if proposal_id not in values:
                    raise ValueError(f"proposal absent from candidate slate: {uid}")
                alternatives = [
                    row
                    for row in ordered_candidates(candidate["candidates"], order, uid)
                    if str(row["metric_id"]) != proposal_id
                ][: args.max_alternatives]
                item_prompt = build_verification_prompt(
                    system_prompt,
                    norm,
                    bank[proposal_id],
                    alternatives,
                    bank,
                    context_chars=args.context_chars,
                    description_chars=args.description_chars,
                    example_chars=args.example_chars,
                    max_examples=args.max_examples,
                )
                work.append(
                    {
                        "variant": variant,
                        "order": order,
                        "uid": uid,
                        "candidate": candidate,
                        "primary": proposal,
                        "norm": norm,
                        "alternative_ids": [str(row["metric_id"]) for row in alternatives],
                        "prompt": item_prompt,
                    }
                )

    output.mkdir(parents=True, exist_ok=False)
    inference = {
        "max_alternatives": args.max_alternatives,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "temperature": 0.0,
        "context_chars": args.context_chars,
        "description_chars": args.description_chars,
        "example_chars": args.example_chars,
        "max_examples": args.max_examples,
    }
    freeze = {
        "schema_version": "silver-match-v3-pr-verifier-gepa-batch-score-freeze-v1",
        "status": "FROZEN_BEFORE_OPTIMIZE92_BATCH_SCORING",
        "task": "press-releases",
        "role": "optimize",
        "variant_count": len(prompts),
        "orders": list(ORDERS),
        "paired_count": len(targets),
        "total_prompt_count": len(work),
        "model": args.model,
        "inference": inference,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "prompts": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in prompt_paths.items()},
        "contracts": {
            "optimize_only": True,
            "select_test_mi_outcomes_opened": False,
            "all_variants_and_orders_scored": True,
            "fixed_policy": "two_order_exact_high",
            "this_step_selects_prompt": False,
        },
    }
    freeze_path = output / "SCORING_INPUT_FREEZE.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    from vllm import LLM, SamplingParams  # imported after pre-inference freeze

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    parsed_work: list[tuple[dict[str, Any], dict[str, Any], str | None, str]] = []
    started = time.time()
    for start in range(0, len(work), args.batch_size):
        batch = work[start : start + args.batch_size]
        conversations = [[{"role": "user", "content": row["prompt"]}] for row in batch]
        generated = llm.chat(conversations, sampling, use_tqdm=False)
        first = []
        retries = []
        for index, (row, result) in enumerate(zip(batch, generated, strict=True)):
            raw = result.outputs[0].text if result.outputs else ""
            parsed, error = parse_response(
                raw,
                str(row["primary"]["metric_id"]),
                set(row["alternative_ids"]),
            )
            first.append((parsed, error, raw))
            if parsed is None:
                retries.append(index)
        if retries:
            retry_conversations = [
                [
                    {"role": "user", "content": batch[index]["prompt"]},
                    {"role": "assistant", "content": first[index][2]},
                    {
                        "role": "user",
                        "content": (
                            "Your prior answer violated the JSON contract. Return only a valid "
                            "object. CONFIRM_MATCH must repeat the proposal ID; BETTER_CANDIDATE "
                            "must use an alternative ID; every abstention must use metric_id null."
                        ),
                    },
                ]
                for index in retries
            ]
            regenerated = llm.chat(retry_conversations, sampling, use_tqdm=False)
            for index, result in zip(retries, regenerated, strict=True):
                raw = result.outputs[0].text if result.outputs else ""
                parsed, error = parse_response(
                    raw,
                    str(batch[index]["primary"]["metric_id"]),
                    set(batch[index]["alternative_ids"]),
                )
                first[index] = (parsed, error, raw)
        for row, (parsed, error, raw) in zip(batch, first, strict=True):
            if parsed is None:
                parsed = {
                    "decision": "INVALID_OUTPUT",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": error,
                }
            parsed_work.append((row, parsed, error, raw))
        print(f"verified={len(parsed_work)}/{len(work)}", flush=True)

    outputs: dict[str, dict[str, Path]] = {name: {} for name in prompts}
    predictions: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        name: {order: {} for order in ORDERS} for name in prompts
    }
    for variant in sorted(prompts):
        for order in ORDERS:
            rows = []
            for item, parsed, error, raw in parsed_work:
                if item["variant"] != variant or item["order"] != order:
                    continue
                norm, proposal, candidate = item["norm"], item["primary"], item["candidate"]
                row = {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": norm["norm_uid"],
                    "corpus": norm["corpus"],
                    "task": norm["task"],
                    "row": norm["row"],
                    "primary_metric_id": proposal["metric_id"],
                    "decision": parsed["decision"],
                    "metric_id": parsed["metric_id"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "alternative_ids": item["alternative_ids"],
                    "candidate_bank_source_sha256": candidate["bank_source_sha256"],
                    "primary_prompt_sha256": proposal.get("prompt_sha256"),
                    "prompt_sha256": prompt_sha256(prompts[variant]),
                    "model": args.model,
                    "order_mode": order,
                    "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "raw_response": raw if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "item_prompt_sha256": prompt_sha256(item["prompt"]),
                }
                rows.append(row)
                predictions[variant][order][str(norm["norm_uid"])] = row
            path = output / variant / f"{order}.jsonl"
            write_jsonl(path, rows)
            meta = {
                "schema_version": manifest["schema_version"],
                "output": str(path),
                "output_sha256": sha256_file(path),
                "prompt": str(prompt_paths[variant]),
                "prompt_sha256": prompt_sha256(prompts[variant]),
                "model": args.model,
                "order_mode": order,
                "eligible_count": len(rows),
                "invalid_count": sum(row["parse_error"] is not None for row in rows),
                **inference,
            }
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            outputs[variant][order] = path

    scores = {
        variant: _score(
            targets,
            primary,
            predictions[variant]["original"],
            predictions[variant]["hashed"],
        )
        for variant in sorted(prompts)
    }
    report = {
        "schema_version": "silver-match-v3-pr-verifier-gepa-optimize92-batch-score-v1",
        "status": "SCORED_ALL_CANDIDATES_OPTIMIZE_ONLY_NOT_SELECTED",
        "task": "press-releases",
        "role": "optimize",
        "policy": "two_order_exact_high",
        "scores": scores,
        "outputs": {
            variant: {
                order: {"path": str(path), "sha256": sha256_file(path)}
                for order, path in values.items()
            }
            for variant, values in outputs.items()
        },
        "input_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "elapsed_seconds": time.time() - started,
        "selection_performed": False,
        "select_test_mi_outcomes_opened": False,
    }
    report_path = output / "SCORE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
