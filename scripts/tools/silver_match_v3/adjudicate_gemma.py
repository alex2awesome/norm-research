#!/usr/bin/env python3
"""Independent one-norm Gemma-4 adjudication over retrieved candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from .common import read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT, GEMMA4
from .retrieve import stable_shard


DECISIONS = {
    "MATCH",
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}
CONFIDENCES = {"high", "medium", "low"}


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def truncate(text: str, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def ordered_candidates(
    candidate_rows: list[dict[str, Any]], mode: str, norm_uid: str
) -> list[dict[str, Any]]:
    rows = list(candidate_rows)
    if mode == "reverse":
        return list(reversed(rows))
    if mode == "hashed":
        return sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{norm_uid}\x1f{row['metric_id']}".encode()
            ).hexdigest(),
        )
    return rows


def prompt_equivalence_groups(
    batch: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]],
) -> tuple[list[int], list[int], dict[int, int]]:
    """Collapse only byte-identical rendered inference requests.

    The rendered prompt already contains the task, human evidence and every
    ordered metric card.  We additionally include the ordered IDs in the key
    as a defensive invariant.  Broadcasting one deterministic temperature-0
    response within such a group changes no model input and preserves a
    representative UID for auditability.
    """

    representative_indices: list[int] = []
    representative_for: list[int] = []
    group_sizes: dict[int, int] = {}
    seen: dict[tuple[str, tuple[str, ...]], int] = {}
    for index, (_, _, candidates, prompt) in enumerate(batch):
        key = (prompt, tuple(str(row["metric_id"]) for row in candidates))
        representative = seen.get(key)
        if representative is None:
            representative = index
            seen[key] = index
            representative_indices.append(index)
            group_sizes[index] = 0
        representative_for.append(representative)
        group_sizes[representative] += 1
    return representative_indices, representative_for, group_sizes


def build_item_prompt(
    system_prompt: str,
    norm: dict[str, Any],
    candidates: list[dict[str, Any]],
    metric_by_id: dict[str, dict[str, Any]],
    *,
    context_chars: int = 1400,
    description_chars: int = 520,
    example_chars: int = 180,
    max_examples: int = 2,
    rescue_context: str = "",
) -> str:
    lines = [system_prompt.rstrip(), "", f"TASK BANK: {norm['task']}"]
    lines.append(f'HUMAN STATEMENT: "{norm["norm"]}"')
    context = truncate(norm.get("context"), context_chars)
    if context and context != str(norm["norm"]).strip():
        lines.append(f'EVIDENCE PASSAGE FROM THE HUMAN FEEDBACK: "{context}"')
    if norm.get("aspect"):
        lines.append(f"EXTRACTION ASPECT HINT (weak evidence only): {norm['aspect']}")
    if norm.get("polarity"):
        lines.append(f"EXTRACTED POLARITY (does not determine metric): {norm['polarity']}")
    lines.extend(["", "CANDIDATE METRIC CARDS:"])
    for row in candidates:
        metric = metric_by_id[row["metric_id"]]
        lines.append(f"[{metric['metric_id']}] {metric['name']}")
        lines.append(f"  Definition: {truncate(metric['description'], description_chars)}")
        examples = metric.get("examples") or []
        if examples:
            rendered = "; ".join(
                truncate(example, example_chars) for example in examples[:max_examples]
            )
            lines.append(f"  Bank examples: {rendered}")
    if rescue_context:
        lines.extend(
            [
                "",
                "EXHAUSTIVE RESCUE CONTEXT (provenance, not a verdict):",
                truncate(rescue_context, 1400),
            ]
        )
    lines.extend(["", "Return the JSON decision now."])
    return "\n".join(lines)


def parse_response(raw: str, candidate_ids: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    # Regex cannot parse JSON: braces inside a quoted reason (often source
    # code such as ``{}``) truncated otherwise valid model outputs.  Try a
    # real decoder from every object start and keep the last valid object.
    decoder = json.JSONDecoder()
    objects = []
    for start, char in enumerate(raw or ""):
        if char != "{":
            continue
        suffix = (raw or "")[start:]
        try:
            value, _ = decoder.raw_decode(suffix)
        except (json.JSONDecodeError, TypeError):
            # Models occasionally emit a literal LaTeX/code backslash in an
            # otherwise valid JSON string (for example ``"`\|`"``). JSON
            # permits only the escapes in ["\\/bfnrtu]. Escape any other
            # backslash and retry without changing the semantic answer.
            repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", suffix)
            if repaired == suffix:
                continue
            try:
                value, _ = decoder.raw_decode(repaired)
            except (json.JSONDecodeError, TypeError):
                continue
        if isinstance(value, dict) and "decision" in value:
            objects.append(value)
    if not objects:
        return None, "no_json"
    parsed = objects[-1]
    decision = str(parsed.get("decision") or "").strip().upper()
    metric_id = parsed.get("metric_id")
    metric_id = None if metric_id in (None, "", "null", "None") else str(metric_id).strip()
    raw_confidence = parsed.get("confidence")
    confidence_raw = None
    if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
        confidence_raw = float(raw_confidence)
        if not 0.0 <= confidence_raw <= 1.0:
            return None, "unknown_confidence"
        confidence = (
            "high"
            if confidence_raw >= 0.8
            else "medium"
            if confidence_raw >= 0.5
            else "low"
        )
    else:
        confidence = str(raw_confidence or "").strip().lower()
    reason = str(parsed.get("reason") or "").strip()
    if decision not in DECISIONS:
        return None, "unknown_decision"
    if confidence not in CONFIDENCES:
        return None, "unknown_confidence"
    if decision == "MATCH":
        if metric_id not in candidate_ids:
            return None, "metric_not_in_candidates"
    elif metric_id is not None:
        return None, "metric_on_abstention"
    if not reason:
        return None, "missing_reason"
    result = {
        "decision": decision,
        "metric_id": metric_id,
        "confidence": confidence,
        "reason": reason,
    }
    # A frozen prompt may request calibrated [0,1] confidence while the
    # downstream row contract remains categorical.  Preserve the raw value in
    # the parsed object; inference runs that need byte-level provenance also
    # retain ``raw_response`` with ``--keep-raw``.
    if confidence_raw is not None:
        result["confidence_raw"] = confidence_raw
    return result, None


def scan_candidate_input(
    candidates_path: Path,
    *,
    done: set[str],
    shard_id: int,
    num_shards: int,
) -> tuple[set[str], int]:
    corpora: set[str] = set()
    eligible = 0
    for row in read_jsonl(candidates_path):
        corpora.add(str(row["corpus"]))
        uid = str(row["norm_uid"])
        if uid not in done and stable_shard(uid, num_shards) == shard_id:
            eligible += 1
    return corpora, eligible


def load_inputs(manifest_path: Path, corpora: set[str] | Path | str):
    """Load canonical norms/banks, retaining the legacy verifier interface.

    New streaming adjudication passes a corpus set and receives three values.
    Existing verifier/audit callers pass a candidate path and receive the
    historical four-tuple including materialized candidate rows.
    """

    legacy_candidate_rows: list[dict[str, Any]] | None = None
    if isinstance(corpora, (str, Path)):
        legacy_candidate_rows = list(read_jsonl(Path(corpora)))
        corpus_names = {str(row["corpus"]) for row in legacy_candidate_rows}
    else:
        corpus_names = set(corpora)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_corpus: dict[str, dict[str, Any]] = {}
    banks: dict[str, dict[str, Any]] = {}
    for corpus in corpus_names:
        meta = manifest["corpora"][corpus]
        by_corpus[corpus] = {
            row["norm_uid"]: row for row in read_jsonl(Path(meta["path"]))
        }
        task = meta["task"]
        if task not in banks:
            bank = json.loads(
                Path(manifest["banks"][task]["path"]).read_text(encoding="utf-8")
            )["metrics"]
            banks[task] = {metric["metric_id"]: metric for metric in bank}
    if legacy_candidate_rows is not None:
        return manifest, legacy_candidate_rows, by_corpus, banks
    return manifest, by_corpus, banks


def iter_work_items(
    candidates_path: Path,
    *,
    done: set[str],
    shard_id: int,
    num_shards: int,
    max_candidates: int,
    order_mode: str,
    system_prompt: str,
    norms_by_corpus: dict[str, dict[str, Any]],
    banks: dict[str, dict[str, Any]],
    context_chars: int,
    description_chars: int,
    example_chars: int,
    max_examples: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]]:
    for candidate_row in read_jsonl(candidates_path):
        uid = str(candidate_row["norm_uid"])
        if uid in done or stable_shard(uid, num_shards) != shard_id:
            continue
        corpus = str(candidate_row["corpus"])
        norm = norms_by_corpus[corpus].get(uid)
        if norm is None:
            raise KeyError(f"candidate norm_uid missing from manifest: {uid}")
        task_bank = banks[norm["task"]]
        candidates = ordered_candidates(
            candidate_row["candidates"][:max_candidates], order_mode, uid
        )
        unknown = [
            row["metric_id"] for row in candidates if row["metric_id"] not in task_bank
        ]
        if unknown:
            raise KeyError(f"unknown candidate IDs for {uid}: {unknown}")
        prompt = build_item_prompt(
            system_prompt,
            norm,
            candidates,
            task_bank,
            context_chars=context_chars,
            description_chars=description_chars,
            example_chars=example_chars,
            max_examples=max_examples,
            rescue_context=str(candidate_row.get("rescue_context") or ""),
        )
        yield candidate_row, norm, candidates, prompt


def batched_work(
    items: Iterator[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]],
    batch_size: int,
) -> Iterator[list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str]]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    candidates_path = Path(args.candidates)
    output_path = Path(args.output)
    prompt_paths = [Path(args.prompt), *[Path(path) for path in args.prompt_addon]]
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    prompt_hash = prompt_sha256(system_prompt)
    done = (
        {row["norm_uid"] for row in read_jsonl(output_path)}
        if args.resume and output_path.exists()
        else set()
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")
    corpora, eligible_count = scan_candidate_input(
        candidates_path,
        done=done,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )
    manifest, norms_by_corpus, banks = load_inputs(manifest_path, corpora)
    work = iter_work_items(
        candidates_path,
        done=done,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        max_candidates=args.max_candidates,
        order_mode=args.order_mode,
        system_prompt=system_prompt,
        norms_by_corpus=norms_by_corpus,
        banks=banks,
        context_chars=args.context_chars,
        description_chars=args.description_chars,
        example_chars=args.example_chars,
        max_examples=args.max_examples,
    )

    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    # sk3's account HOME is on read-only AFS in non-interactive jobs.  Newer
    # FlashInfer versions otherwise try to create JIT locks there.
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/lfs/skampere3/0/alexspan")
    os.environ.setdefault("VLLM_USE_FLASHINFER_MOE_FP8", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)
    started = time.time()
    written = invalid = unique_prompt_inferences = retry_prompt_inferences = 0
    for batch in batched_work(work, args.batch_size):
        representatives, representative_for, group_sizes = prompt_equivalence_groups(batch)
        inference_batch = [batch[index] for index in representatives]
        unique_prompt_inferences += len(inference_batch)
        conversations = [
            [{"role": "user", "content": item[3]}] for item in inference_batch
        ]
        outputs = llm.chat(conversations, sampling, use_tqdm=False)
        rows = []
        retry_indices = []
        representative_values: dict[
            int, tuple[dict[str, Any] | None, str | None, str]
        ] = {}
        for representative, item, output in zip(
            representatives, inference_batch, outputs
        ):
            raw = output.outputs[0].text if output.outputs else ""
            candidate_ids = {row["metric_id"] for row in item[2]}
            parsed, error = parse_response(raw, candidate_ids)
            representative_values[representative] = (parsed, error, raw)
            if parsed is None:
                retry_indices.append(representative)

        if retry_indices:
            retry_prompt_inferences += len(retry_indices)
            retry_conversations = []
            for representative in retry_indices:
                original = batch[representative][3]
                raw = representative_values[representative][2]
                retry_conversations.append(
                    [
                        {"role": "user", "content": original},
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your prior answer violated the JSON contract. Return only a valid "
                                "object. MATCH must use an ID from this item's cards; every abstention "
                                "must use metric_id null."
                            ),
                        },
                    ]
                )
            retry_outputs = llm.chat(retry_conversations, sampling, use_tqdm=False)
            for representative, output in zip(retry_indices, retry_outputs):
                raw = output.outputs[0].text if output.outputs else ""
                candidate_ids = {
                    row["metric_id"] for row in batch[representative][2]
                }
                parsed, error = parse_response(raw, candidate_ids)
                representative_values[representative] = (parsed, error, raw)

        parsed_values = [
            representative_values[representative]
            for representative in representative_for
        ]

        for item_index, (item, (parsed, error, raw)) in enumerate(
            zip(batch, parsed_values)
        ):
            candidate_row, norm, candidates, _ = item
            representative = representative_for[item_index]
            if parsed is None:
                invalid += 1
                parsed = {
                    "decision": "INVALID_OUTPUT",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": error,
                }
            rows.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": norm["norm_uid"],
                    "corpus": norm["corpus"],
                    "task": norm["task"],
                    "row": norm["row"],
                    "decision": parsed["decision"],
                    "metric_id": parsed["metric_id"],
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "candidate_ids": [row["metric_id"] for row in candidates],
                    "candidate_bank_source_sha256": candidate_row["bank_source_sha256"],
                    "prompt_sha256": prompt_hash,
                    "model": args.model,
                    "order_mode": args.order_mode,
                    "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "raw_response": raw if args.keep_raw or parsed["decision"] == "INVALID_OUTPUT" else None,
                    "item_prompt_sha256": prompt_sha256(item[3]),
                    "inference_representative_norm_uid": batch[representative][1][
                        "norm_uid"
                    ],
                    "inference_equivalence_size": group_sizes[representative],
                    # Optional provenance used by exhaustive complementary-slate
                    # abstention rescue.  Ordinary candidate rows omit it.
                    "rescue_trial": candidate_row.get("rescue_trial"),
                    "rescue_lane": candidate_row.get("rescue_lane"),
                    "rescue_system": candidate_row.get("rescue_system"),
                    "rescue_coverage_before": candidate_row.get("rescue_coverage_before"),
                    "rescue_coverage_after": candidate_row.get("rescue_coverage_after"),
                    "rescue_bank_count": candidate_row.get("rescue_bank_count"),
                }
            )
        append_rows(output_path, rows)
        written += len(rows)
        print(
            f"adjudicated={written}/{eligible_count} written={written} "
            f"invalid={invalid} elapsed={time.time() - started:.0f}s",
            flush=True,
        )

    meta = {
        "schema_version": manifest["schema_version"],
        "input_candidates": str(candidates_path),
        "input_candidates_sha256": sha256_file(candidates_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "prompt": str(args.prompt),
        "prompt_addons": [str(path) for path in args.prompt_addon],
        "prompt_component_sha256": {
            str(path): sha256_file(path) for path in prompt_paths
        },
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "python_executable": str(Path(sys.executable).resolve()),
        "order_mode": args.order_mode,
        "max_candidates": args.max_candidates,
        "prompt_rendering": {
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
        },
        "runtime": {
            "temperature": 0.0,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "keep_raw": args.keep_raw,
            "resume": args.resume,
        },
        "new_count": written,
        "eligible_count": eligible_count,
        "unique_prompt_inferences": unique_prompt_inferences,
        "deduplicated_prompt_count": written - unique_prompt_inferences,
        "retry_prompt_inferences": retry_prompt_inferences,
        "invalid_count": invalid,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "elapsed_seconds": time.time() - started,
    }
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).with_name("prompts") / "gepa_round0.txt"),
    )
    parser.add_argument(
        "--prompt-addon",
        action="append",
        default=[],
        help="append a task-specific GEPA instruction file (repeatable)",
    )
    parser.add_argument("--model", default=GEMMA4)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--order-mode", choices=("original", "reverse", "hashed"), default="original")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards)")
    if args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if min(args.context_chars, args.description_chars, args.example_chars) < 1:
        parser.error("prompt truncation lengths must be positive")
    if args.max_examples < 0:
        parser.error("--max-examples must be nonnegative")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
