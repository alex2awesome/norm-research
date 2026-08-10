#!/usr/bin/env python3
"""Apply a frozen clean task cross-encoder to v3 candidate slates.

This is a scalable proposal/reranking stage, never a final silver decision.
The selected dev gate may emit a provisional MATCH proposal; every proposal
still remains subject to the task's frozen verifier and blind release gate.
Rows below the gate remain provisional abstentions and proceed to the normal
multi-system rescue path.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .common import metric_card, norm_query, read_jsonl, sha256_file
from .config import DEFAULT_OUTPUT_ROOT
from .retrieve import append_rows


SCHEMA_VERSION = "silver-match-v3.cross-encoder-proposals.1"


def _resolve(path: str, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected_dev") or {}
    for key in ("score_threshold", "margin_threshold"):
        if key not in selected:
            raise ValueError(f"training report lacks selected_dev.{key}: {path}")
    return report


def _verify_model(report: dict[str, Any], report_path: Path) -> Path:
    model_dir = Path(str(report["model_dir"]))
    if not model_dir.is_absolute():
        model_dir = (report_path.parent / model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    expected = report.get("model_hashes") or {}
    if not expected:
        raise ValueError("training report has no frozen model_hashes")
    observed = {
        str(path.relative_to(model_dir)): sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    if observed != expected:
        raise ValueError("cross-encoder model hashes differ from training report")
    return model_dir


def rerank_candidates(
    candidates: Sequence[dict[str, Any]],
    scores: Sequence[float],
    *,
    score_threshold: float,
    margin_threshold: float,
    output_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank one slate and return a non-final gated proposal."""
    if len(candidates) != len(scores):
        raise ValueError("candidate/score length mismatch")
    if not candidates:
        raise ValueError("empty candidate slate")
    enriched = []
    for original_index, (candidate, score) in enumerate(zip(candidates, scores)):
        row = dict(candidate)
        row["ce_score"] = float(score)
        row["ce_original_index"] = original_index
        enriched.append(row)
    enriched.sort(
        key=lambda row: (
            -float(row["ce_score"]),
            int(row["ce_original_index"]),
            str(row["metric_id"]),
        )
    )
    for rank, row in enumerate(enriched, 1):
        row["ce_rank"] = rank
    top_score = float(enriched[0]["ce_score"])
    second_score = float(enriched[1]["ce_score"]) if len(enriched) > 1 else 0.0
    margin = top_score - second_score
    passed = top_score >= score_threshold and margin >= margin_threshold
    proposal = {
        "decision": "PROVISIONAL_MATCH" if passed else "PROVISIONAL_ABSTAIN",
        "metric_id": str(enriched[0]["metric_id"]) if passed else None,
        "top_metric_id": str(enriched[0]["metric_id"]),
        "top_score": top_score,
        "second_metric_id": str(enriched[1]["metric_id"]) if len(enriched) > 1 else None,
        "second_score": second_score,
        "margin": margin,
        "score_threshold": float(score_threshold),
        "margin_threshold": float(margin_threshold),
    }
    return enriched[: min(output_k, len(enriched))], proposal


def _candidate_rows(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from read_jsonl(path)


def _existing_uids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    uids = []
    for row in read_jsonl(path):
        uids.append(str(row["norm_uid"]))
    if len(uids) != len(set(uids)):
        raise ValueError(f"duplicate norm_uid in resume output: {path}")
    return set(uids)


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HOME", "/lfs/skampere3/0/shared_hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.task not in manifest.get("banks", {}):
        raise KeyError(args.task)
    bank_meta = manifest["banks"][args.task]
    bank_path = _resolve(str(bank_meta["path"]), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))["metrics"]
    bank_by_id = {str(metric["metric_id"]): metric for metric in bank}

    report_path = Path(args.training_report).resolve()
    report = _load_report(report_path)
    if report.get("task") != args.task:
        raise ValueError("training-report task mismatch")
    if report.get("bank_source_sha256") != bank_meta["source_sha256"]:
        raise ValueError("training-report bank hash is stale")
    if report.get("status") not in {"DEV_PROMOTABLE_PENDING_BLIND", "PROMOTABLE"}:
        raise ValueError(f"cross-encoder report is not eligible: {report.get('status')}")
    if report.get("dev_promotable") is not True:
        raise ValueError("cross-encoder did not clear its frozen dev promotion gate")
    model_dir = _verify_model(report, report_path)
    selected = report["selected_dev"]
    score_threshold = float(selected["score_threshold"])
    margin_threshold = float(selected["margin_threshold"])

    candidate_paths = [Path(path).resolve() for path in args.candidates]
    if not candidate_paths:
        raise ValueError("at least one candidate file is required")
    for path in candidate_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_hashes = {str(path): sha256_file(path) for path in candidate_paths}

    norms: dict[str, dict[str, Any]] = {}
    expected_uids: set[str] = set()
    input_rows = 0
    corpus_counts: Counter[str] = Counter()
    for row in _candidate_rows(candidate_paths):
        uid = str(row["norm_uid"])
        if uid in expected_uids:
            raise ValueError(f"duplicate candidate norm_uid: {uid}")
        expected_uids.add(uid)
        input_rows += 1
        corpus_counts[str(row["corpus"])] += 1
        if row.get("task") != args.task:
            raise ValueError(f"candidate task mismatch for {uid}")
        if row.get("bank_source_sha256") != bank_meta["source_sha256"]:
            raise ValueError(f"candidate bank hash mismatch for {uid}")
        ids = [str(item["metric_id"]) for item in row.get("candidates") or []]
        if not ids or len(ids) != len(set(ids)) or any(value not in bank_by_id for value in ids):
            raise ValueError(f"invalid candidate slate for {uid}")
    if args.expected_rows is not None and input_rows != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} candidate rows, found {input_rows}")

    for corpus, corpus_meta in manifest["corpora"].items():
        if corpus_meta["task"] != args.task or corpus not in corpus_counts:
            continue
        for norm in read_jsonl(_resolve(str(corpus_meta["path"]), manifest_path)):
            uid = str(norm["norm_uid"])
            if uid in expected_uids:
                norms[uid] = norm
    missing_norms = expected_uids - set(norms)
    if missing_norms:
        raise ValueError(f"candidate UIDs absent from manifest norms: {sorted(missing_norms)[:5]}")

    output = Path(args.output).resolve()
    meta_path = Path(str(output) + ".meta.json")
    pin = {
        "schema_version": SCHEMA_VERSION,
        "task": args.task,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bank": str(bank_path),
        "bank_source_sha256": bank_meta["source_sha256"],
        "training_report": str(report_path),
        "training_report_sha256": sha256_file(report_path),
        "model_dir": str(model_dir),
        "model_hashes": report["model_hashes"],
        "candidate_inputs": candidate_hashes,
        "input_rows": input_rows,
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "score_threshold": score_threshold,
        "margin_threshold": margin_threshold,
        "output_k": args.output_k,
    }
    completed: set[str]
    if args.resume:
        if not meta_path.is_file():
            raise FileNotFoundError(f"resume requires metadata: {meta_path}")
        prior = json.loads(meta_path.read_text(encoding="utf-8"))
        if prior.get("pins") != pin:
            raise ValueError("resume metadata pins differ from current inputs")
        completed = _existing_uids(output)
    else:
        if output.exists() or meta_path.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {meta_path}")
        completed = set()
        _write_meta(meta_path, {"status": "RUNNING", "pins": pin, "written": 0})

    from sentence_transformers import CrossEncoder
    import torch

    model = CrossEncoder(str(model_dir), device=args.device)
    pending: list[dict[str, Any]] = []
    proposal_counts: Counter[str] = Counter()
    written = len(completed)

    def flush(rows: list[dict[str, Any]]) -> None:
        nonlocal written
        pairs: list[list[str]] = []
        lengths: list[int] = []
        for row in rows:
            query = norm_query(norms[str(row["norm_uid"])])
            slate = row["candidates"]
            lengths.append(len(slate))
            pairs.extend(
                [query, metric_card(bank_by_id[str(candidate["metric_id"])])]
                for candidate in slate
            )
        predicted = np.asarray(
            model.predict(
                pairs,
                batch_size=args.pair_batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Sigmoid(),
            ),
            dtype=np.float32,
        )
        offset = 0
        output_rows = []
        for row, length in zip(rows, lengths):
            scores = predicted[offset : offset + length]
            offset += length
            reranked, proposal = rerank_candidates(
                row["candidates"],
                scores,
                score_threshold=score_threshold,
                margin_threshold=margin_threshold,
                output_k=args.output_k,
            )
            proposal_counts[proposal["decision"]] += 1
            output_rows.append(
                {
                    **{key: value for key, value in row.items() if key != "candidates"},
                    "schema_version": SCHEMA_VERSION,
                    "candidates": reranked,
                    "ce_proposal": proposal,
                    "ce_training_report_sha256": pin["training_report_sha256"],
                }
            )
        append_rows(output, output_rows)
        written += len(output_rows)
        _write_meta(
            meta_path,
            {
                "status": "RUNNING",
                "pins": pin,
                "written": written,
                "proposal_counts_current_process": dict(sorted(proposal_counts.items())),
            },
        )

    for row in _candidate_rows(candidate_paths):
        if str(row["norm_uid"]) in completed:
            continue
        pending.append(row)
        if len(pending) >= args.row_batch_size:
            flush(pending)
            pending = []
    if pending:
        flush(pending)

    observed = _existing_uids(output)
    if observed != expected_uids:
        raise ValueError(
            f"incomplete CE output: expected {len(expected_uids)}, observed {len(observed)}"
        )
    final_counts = Counter(
        str(row["ce_proposal"]["decision"]) for row in read_jsonl(output)
    )
    final = {
        "status": "COMPLETE",
        "pins": pin,
        "written": len(observed),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "proposal_counts": dict(sorted(final_counts.items())),
    }
    _write_meta(meta_path, final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_OUTPUT_ROOT / "manifest.json"))
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--candidates", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--row-batch-size", type=int, default=128)
    parser.add_argument("--pair-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_k < 2:
        parser.error("--output-k must be at least 2")
    if args.row_batch_size < 1 or args.pair_batch_size < 1:
        parser.error("batch sizes must be positive")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
