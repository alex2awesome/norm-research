#!/usr/bin/env python3
"""Independently verify typed abstentions after exhaustive bank rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES, append_rows, prompt_sha256, truncate
from .common import read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT, GEMMA4
from .retrieve import stable_shard


TYPED_DECISIONS = {
    "MATCH_FAMILY_ONLY",
    "NO_EXPLICIT_CRITERION",
    "CONTEXT_NEEDED",
    "GENERIC_VERDICT",
    "NO_CANDIDATE_FITS",
    "NOISE",
}
DECISIONS = TYPED_DECISIONS | {"POSSIBLE_EXACT_BANK_MATCH"}


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    decoder = json.JSONDecoder()
    objects = []
    for start, char in enumerate(raw or ""):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode((raw or "")[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict) and "decision" in value:
            objects.append(value)
    if not objects:
        return None, "no_json"
    value = objects[-1]
    decision = str(value.get("decision") or "").strip().upper()
    metric_id = value.get("metric_id")
    confidence = str(value.get("confidence") or "").strip().lower()
    reason = str(value.get("reason") or "").strip()
    if decision not in DECISIONS:
        return None, "unknown_decision"
    if metric_id not in (None, "", "null", "None"):
        return None, "metric_on_abstention_verification"
    if confidence not in CONFIDENCES:
        return None, "unknown_confidence"
    if not reason:
        return None, "missing_reason"
    return {
        "decision": decision,
        "metric_id": None,
        "confidence": confidence,
        "reason": reason,
    }, None


def build_prompt(
    system_prompt: str,
    norm: dict[str, Any],
    audit: dict[str, Any],
    *,
    order_mode: str = "original",
) -> str:
    trial_results = list(audit.get("trial_results") or [])
    if order_mode == "reverse":
        trial_results.reverse()
    elif order_mode == "hashed":
        uid = str(audit.get("norm_uid") or norm.get("norm_uid") or "")
        trial_results.sort(
            key=lambda row: hashlib.sha256(
                f"{uid}|{row.get('trial')}".encode("utf-8")
            ).hexdigest()
        )
    elif order_mode != "original":
        raise ValueError(f"unknown order_mode: {order_mode}")
    trial_lines = []
    for row in trial_results:
        trial_lines.append(
            f"trial {row.get('trial')}: {row.get('decision')} ({row.get('confidence')}) — "
            f"{truncate(row.get('reason'), 260)}"
        )
    lines = [
        system_prompt.rstrip(),
        "",
        f"TASK BANK: {norm['task']}",
        f"FROZEN BANK SIZE: {audit.get('rescue_bank_count')}",
        (
            "BANK COVERAGE: exhaustive; every metric appeared in "
            f"{audit.get('rescue_coverage_repeats', 1)} independent rescue captures "
            f"(primary metrics re-included={audit.get('rescue_reincludes_primary', False)})"
        ),
        f'HUMAN STATEMENT: "{norm.get("norm")}"',
    ]
    context = truncate(norm.get("context"), 1800)
    if context and context != str(norm.get("norm") or "").strip():
        lines.append(f'EVIDENCE PASSAGE: "{context}"')
    lines.extend(
        [
            f"EXTRACTION KIND/POLARITY: {norm.get('kind')} / {norm.get('polarity')}",
            f"PROVISIONAL TYPE: {audit.get('provisional_decision')}",
            f"WEIGHTED VOTES: {json.dumps(audit.get('vote_counts') or {}, sort_keys=True)}",
            "TRIAL SUMMARIES:",
            *(trial_lines or ["(none)"]),
            "",
            "Return the JSON verification now.",
        ]
    )
    return "\n".join(lines)


def _load_norms(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    needed_by_corpus: dict[str, set[str]] = {}
    for row in rows:
        needed_by_corpus.setdefault(str(row["corpus"]), set()).add(str(row["norm_uid"]))
    norms = {}
    for corpus, needed in needed_by_corpus.items():
        meta = manifest["corpora"][corpus]
        for norm in read_jsonl(Path(meta["path"])):
            if norm["norm_uid"] in needed:
                norms[norm["norm_uid"]] = norm
    missing = {row["norm_uid"] for row in rows} - set(norms)
    if missing:
        raise ValueError(f"audit UIDs absent from canonical manifest: {sorted(missing)[:3]}")
    return norms


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    audits_path = Path(args.audits)
    output_path = Path(args.output)
    prompt_paths = [Path(args.prompt), *map(Path, args.prompt_addon)]
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    prompt_hash = prompt_sha256(system_prompt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit_rows = list(read_jsonl(audits_path))
    audit_by_uid = {row["norm_uid"]: row for row in audit_rows}
    if len(audit_by_uid) != len(audit_rows):
        raise ValueError("duplicate norm_uid in abstention audits")
    norms = _load_norms(manifest, audit_rows)
    done = (
        {row["norm_uid"] for row in read_jsonl(output_path)}
        if args.resume and output_path.exists()
        else set()
    )
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --resume")
    work = []
    for uid, audit in sorted(audit_by_uid.items()):
        if uid in done or stable_shard(uid, args.num_shards) != args.shard_id:
            continue
        if not audit.get("rescue_exhaustive"):
            raise ValueError(f"non-exhaustive row sent to final abstention verifier: {uid}")
        if audit.get("provisional_decision") not in TYPED_DECISIONS:
            raise ValueError(f"invalid provisional abstention for {uid}")
        norm = norms[uid]
        if norm["task"] != audit["task"] or norm["corpus"] != audit["corpus"]:
            raise ValueError(f"routing mismatch for {uid}")
        work.append(
            (
                audit,
                norm,
                build_prompt(
                    system_prompt, norm, audit, order_mode=args.order_mode
                ),
            )
        )

    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
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
    written = invalid = possible = 0
    for start in range(0, len(work), args.batch_size):
        batch = work[start : start + args.batch_size]
        conversations = [[{"role": "user", "content": item[2]}] for item in batch]
        outputs = llm.chat(conversations, sampling, use_tqdm=False)
        parsed_values = []
        retry = []
        for index, output in enumerate(outputs):
            raw = output.outputs[0].text if output.outputs else ""
            parsed, error = parse_response(raw)
            parsed_values.append((parsed, error, raw))
            if parsed is None:
                retry.append(index)
        if retry:
            retry_outputs = llm.chat(
                [
                    [
                        {"role": "user", "content": batch[index][2]},
                        {"role": "assistant", "content": parsed_values[index][2]},
                        {
                            "role": "user",
                            "content": "Your answer violated the JSON contract. Return only one valid object with metric_id null.",
                        },
                    ]
                    for index in retry
                ],
                sampling,
                use_tqdm=False,
            )
            for index, output in zip(retry, retry_outputs):
                raw = output.outputs[0].text if output.outputs else ""
                parsed, error = parse_response(raw)
                parsed_values[index] = (parsed, error, raw)
        rows = []
        for (audit, norm, _), (parsed, error, raw) in zip(batch, parsed_values):
            if parsed is None:
                invalid += 1
                parsed = {
                    "decision": "INVALID_OUTPUT",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": error,
                }
            possible += int(parsed["decision"] == "POSSIBLE_EXACT_BANK_MATCH")
            rows.append(
                {
                    "schema_version": manifest["schema_version"],
                    "norm_uid": norm["norm_uid"],
                    "corpus": norm["corpus"],
                    "task": norm["task"],
                    "row": norm["row"],
                    "provisional_decision": audit["provisional_decision"],
                    "decision": parsed["decision"],
                    "confirmed_decision": (
                        parsed["decision"] if parsed["decision"] in TYPED_DECISIONS else None
                    ),
                    "possible_exact_bank_match": parsed["decision"] == "POSSIBLE_EXACT_BANK_MATCH",
                    "metric_id": None,
                    "confidence": parsed["confidence"],
                    "reason": parsed["reason"],
                    "rescue_bank_count": audit["rescue_bank_count"],
                    "rescue_coverage_repeats": audit.get(
                        "rescue_coverage_repeats", 1
                    ),
                    "rescue_reincludes_primary": audit.get(
                        "rescue_reincludes_primary", False
                    ),
                    "rescue_vote_counts": audit.get("vote_counts"),
                    "bank_source_sha256": audit["bank_source_sha256"],
                    "prompt_sha256": prompt_hash,
                    "model": args.model,
                    "order_mode": args.order_mode,
                    "parse_error": error if parsed["decision"] == "INVALID_OUTPUT" else None,
                    "raw_response": raw if args.keep_raw or parsed["decision"] == "INVALID_OUTPUT" else None,
                }
            )
        append_rows(output_path, rows)
        written += len(rows)
        print(
            f"abstention_verified={start + len(batch)}/{len(work)} written={written} "
            f"possible_match={possible} invalid={invalid} elapsed={time.time()-started:.0f}s",
            flush=True,
        )
    meta = {
        "schema_version": "silver-match-v3-abstention-verification-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "audits": str(audits_path),
        "audits_sha256": sha256_file(audits_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "prompt_components": {str(path): sha256_file(path) for path in prompt_paths},
        "prompt_sha256": prompt_hash,
        "model": args.model,
        "order_mode": args.order_mode,
        "new_count": written,
        "possible_exact_bank_match_count": possible,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--audits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prompt",
        default=str(Path(__file__).with_name("prompts") / "verify_abstention_v1.txt"),
    )
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--model", default=GEMMA4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--order-mode", choices=("original", "reverse", "hashed"), default="original"
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    if not (0 <= args.shard_id < args.num_shards):
        parser.error("--shard-id must be in [0, --num-shards)")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
