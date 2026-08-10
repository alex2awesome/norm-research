#!/usr/bin/env python3
"""Materialize truth-blind task-local Nemotron retrieval candidates.

The identity input deliberately contains no labels.  This keeps candidate
generation usable for a consumed select panel: truth is opened only by a
later scorer, after this candidate file and its hash have been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import metric_card, norm_query, read_jsonl, sha256_file
from .train_nemotron_lora import (
    DEFAULT_NEMOTRON,
    _encode,
    format_query,
    load_nemotron_adapter,
    stable_rank,
)


FORBIDDEN_TRUTH_FIELDS = {
    "decision",
    "metric_id",
    "confidence",
    "reason",
    "label",
    "gold",
}


def _resolve(path: str | Path, manifest_path: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else manifest_path.parent / value


def load_truth_blind_inputs(
    manifest_path: Path,
    identities_path: Path,
    task: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities = list(read_jsonl(identities_path))
    if not identities:
        raise ValueError("identity panel is empty")
    leaked = sorted(
        {
            field
            for row in identities
            for field in FORBIDDEN_TRUTH_FIELDS
            if field in row
        }
    )
    if leaked:
        raise ValueError(f"identity panel contains forbidden truth fields: {leaked}")
    if any(row.get("task") != task for row in identities):
        raise ValueError("identity panel contains a foreign task")
    if any(row.get("gepa_role") != "select" for row in identities):
        raise ValueError("identity panel is not a pure select-role projection")
    uids = [str(row["norm_uid"]) for row in identities]
    if len(uids) != len(set(uids)):
        raise ValueError("identity panel contains duplicate norm_uid values")

    bank_meta = manifest["banks"][task]
    bank_path = _resolve(bank_meta["path"], manifest_path)
    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank = list(bank_payload["metrics"])
    bank_sha = str(bank_meta["source_sha256"])
    if not bank or len({str(row["metric_id"]) for row in bank}) != len(bank):
        raise ValueError("bank is empty or contains duplicate metric IDs")

    wanted = set(uids)
    norms: dict[str, dict[str, Any]] = {}
    corpora = sorted({str(row["corpus"]) for row in identities})
    for corpus in corpora:
        meta = manifest["corpora"].get(corpus)
        if not meta or meta.get("task") != task:
            raise KeyError(f"manifest lacks task-compatible corpus: {corpus}")
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in wanted:
                if uid in norms:
                    raise ValueError(f"duplicate selected norm across manifest: {uid}")
                norms[uid] = row
    missing = sorted(wanted - set(norms))
    if missing:
        raise KeyError(f"selected UIDs absent from manifest: {missing[:5]}")
    ordered_norms = [norms[uid] for uid in uids]
    for identity, norm in zip(identities, ordered_norms):
        if (
            norm.get("corpus") != identity.get("corpus")
            or norm.get("source_group") != identity.get("source_group")
            or norm.get("task") != task
        ):
            raise ValueError(f"identity/manifest mismatch: {identity['norm_uid']}")
    return manifest, ordered_norms, bank, bank_sha


def ranking_rows(
    scores: np.ndarray,
    norms: Sequence[Mapping[str, Any]],
    bank: Sequence[Mapping[str, Any]],
    bank_sha: str,
    top_k: int,
    model: str,
    adapter: Path,
) -> list[dict[str, Any]]:
    if scores.shape != (len(norms), len(bank)):
        raise ValueError(
            f"score shape mismatch: {scores.shape} != {(len(norms), len(bank))}"
        )
    if not 1 <= top_k <= len(bank):
        raise ValueError("top_k must fall within the bank size")
    bank_ids = [str(metric["metric_id"]) for metric in bank]
    output: list[dict[str, Any]] = []
    for row_index, norm in enumerate(norms):
        order = stable_rank(scores[row_index])[:top_k]
        candidates = [
            {
                "metric_id": bank_ids[int(metric_index)],
                "rank": rank,
                "score": float(scores[row_index, int(metric_index)]),
                "retrieval_lane": "task_nemotron_lora",
            }
            for rank, metric_index in enumerate(order, 1)
        ]
        output.append(
            {
                "schema_version": "silver-match-v3-truth-blind-nemotron-candidates-v1",
                "norm_uid": str(norm["norm_uid"]),
                "corpus": str(norm["corpus"]),
                "task": str(norm["task"]),
                "row": norm.get("row"),
                "source_group": str(norm["source_group"]),
                "bank_source_sha256": bank_sha,
                "retriever_model": model,
                "retriever_adapter": str(adapter),
                "top_k": top_k,
                "truth_fields_read": False,
                "candidates": candidates,
            }
        )
    return output


def write_jsonl_new(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidates: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--identities", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", default=DEFAULT_NEMOTRON)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    identities_path = Path(args.identities).resolve()
    adapter_path = Path(args.adapter).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidates: {output_path}")
    manifest, norms, bank, bank_sha = load_truth_blind_inputs(
        manifest_path, identities_path, args.task
    )

    os.environ.setdefault(
        "HF_MODULES_CACHE", "/lfs/skampere3/0/alexspan/.cache/huggingface/modules"
    )
    model = load_nemotron_adapter(
        args.model,
        adapter_path,
        device=args.device,
        attention=args.attention,
        max_seq_length=args.max_seq_length,
    )
    queries = [format_query(norm_query(dict(norm))) for norm in norms]
    cards = [metric_card(dict(metric)) for metric in bank]
    bank_embeddings = _encode(model, cards, args.batch_size)
    query_embeddings = _encode(model, queries, args.batch_size)
    scores = np.asarray(query_embeddings) @ np.asarray(bank_embeddings).T
    rows = ranking_rows(
        scores,
        norms,
        bank,
        bank_sha,
        args.top_k,
        args.model,
        adapter_path,
    )
    write_jsonl_new(output_path, rows)
    report_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen report: {report_path}")
    report = {
        "schema_version": "silver-match-v3-truth-blind-nemotron-candidates-meta-v1",
        "status": "COMPLETE_TRUTH_BLIND_CANDIDATES",
        "task": args.task,
        "count": len(rows),
        "top_k": args.top_k,
        "bank_count": len(bank),
        "bank_source_sha256": bank_sha,
        "truth_fields_read": False,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "identities": {"path": str(identities_path), "sha256": sha256_file(identities_path)},
            "generator_script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "adapter": {
                path.name: sha256_file(path)
                for path in sorted(adapter_path.iterdir())
                if path.is_file()
            },
        },
        "model": args.model,
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
