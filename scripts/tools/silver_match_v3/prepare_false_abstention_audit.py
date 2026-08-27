#!/usr/bin/env python3
"""Draw deterministic blind uniform samples from final matches or abstentions."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _resolve(path: str | Path, anchor: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else anchor.parent / value


def _rank(seed: str, scope: str, uid: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\x1f{scope}\x1f{uid}".encode()).digest(), "big"
    )


def _offer(
    heap: list[tuple[int, str, dict[str, Any]]],
    row: dict[str, Any],
    *,
    limit: int,
    score: int,
) -> None:
    if limit < 1:
        return
    item = (-score, str(row["norm_uid"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def _ordered(
    heap: list[tuple[int, str, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    return sorted(
        [(-negative, row) for negative, _, row in heap], key=lambda value: value[0]
    )


def _write_task_label_pack(
    *,
    output_root: Path,
    scope_name: str,
    scope: str,
    task: str,
    values: list[tuple[int, dict[str, Any]]],
    norms: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
    blind_path: Path,
    key_path: Path,
    seed: str,
    chunk_size: int = 25,
) -> dict[str, Any]:
    """Create a standard, transcript-auditable full-bank pack for one task scope."""

    pack_root = output_root / f"{scope_name}.label_pack"
    pack_root.mkdir(parents=True, exist_ok=False)
    items = []
    for _, prediction in values:
        uid = str(prediction["norm_uid"])
        source = norms[uid]
        if str(source.get("task") or "") != task:
            raise ValueError(f"mixed task in task-scoped final audit: {scope}/{uid}")
        group = split_group_for(source)
        items.append(
            {
                "schema_version": source.get("schema_version")
                or manifest.get("schema_version")
                or "silver-match-v3.0",
                "norm_uid": uid,
                "corpus": source["corpus"],
                "task": task,
                "row": source["row"],
                "norm": source.get("norm"),
                "context": source.get("context"),
                "aspect": source.get("aspect"),
                "kind": source.get("kind"),
                "polarity": source.get("polarity"),
                "split_group": group,
                "source_group": group,
                "split": split_for(group),
                "boundary_stratum": "production_final_blind_risk_audit",
                "sample_scope": scope,
                "permanently_excluded_from_gradients": True,
            }
        )

    source_bank = _resolve(manifest["banks"][task]["path"], manifest_path)
    source_payload = json.loads(source_bank.read_text(encoding="utf-8"))
    metrics = list(source_payload.get("metrics") or [])
    if len(metrics) != int(manifest["banks"][task].get("count", len(metrics))):
        raise ValueError(f"canonical bank count mismatch for {task}")
    metrics.sort(
        key=lambda row: (
            _rank(seed, f"label-bank:{scope}", str(row["metric_id"])),
            str(row["metric_id"]),
        )
    )
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    write_jsonl(items_path, items)
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": source_payload.get("schema_version")
                or manifest.get("schema_version")
                or "silver-match-v3.0",
                "task": task,
                "source_sha256": manifest["banks"][task]["source_sha256"],
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    chunks: dict[str, str] = {}
    for start in range(0, len(items), chunk_size):
        chunk = pack_root / "chunks" / f"part-{start // chunk_size:03d}.jsonl"
        write_jsonl(chunk, items[start : start + chunk_size])
        chunks[str(chunk)] = sha256_file(chunk)
    validation = {
        "schema_version": "silver-match-v3-final-risk-label-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "sample_scope": scope,
        "count": len(items),
        "unique_source_groups": len({str(row["source_group"]) for row in items}),
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "bank_metric_count": len(metrics),
        "bank_source_sha256": manifest["banks"][task]["source_sha256"],
        "truth_hidden": True,
        "system_decisions_hidden": True,
        "system_key_excluded_from_label_pack": True,
        "permanently_excluded_from_gradients": True,
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "blind_sample": {
                "path": str(blind_path),
                "sha256": sha256_file(blind_path),
            },
            "system_key": {
                "path": str(key_path),
                "sha256": sha256_file(key_path),
                "excluded_from_pack": True,
            },
            "canonical_bank": {
                "path": str(source_bank),
                "sha256": sha256_file(source_bank),
                "source_sha256": manifest["banks"][task]["source_sha256"],
            },
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
            "chunks": chunks,
        },
    }
    validation_path = pack_root / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "path": str(validation_path),
        "sha256": sha256_file(validation_path),
        "pack_root": str(pack_root),
        "count": len(items),
    }


def prepare(
    *,
    manifest_path: Path,
    final_paths: list[Path],
    output_root: Path,
    global_n: int,
    per_task_n: int,
    seed: str,
    sample_kind: str = "abstention",
    exclude_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if min(global_n, per_task_n) < 1:
        raise ValueError("sample sizes must be positive")
    if sample_kind not in {"match", "abstention"}:
        raise ValueError("sample_kind must be match or abstention")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty audit root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exclude_paths = exclude_paths or []
    excluded_uids: set[str] = set()
    exclusion_inputs = {}
    for path in exclude_paths:
        exclusion_inputs[str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid:
                raise ValueError(f"analysis exclusion lacks norm_uid: {path}")
            excluded_uids.add(uid)
    global_heap: list[tuple[int, str, dict[str, Any]]] = []
    task_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    population_by_task: Counter[str] = Counter()
    population_by_decision: Counter[str] = Counter()
    seen: set[str] = set()
    for path in final_paths:
        for row in read_jsonl(path):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen:
                raise ValueError(f"missing/duplicate final UID: {uid!r}")
            seen.add(uid)
            if uid in excluded_uids:
                continue
            is_match = row.get("decision") == "MATCH"
            if (sample_kind == "match") != is_match:
                continue
            task = str(row.get("task") or "")
            if task not in manifest.get("banks", {}):
                raise ValueError(f"unknown task in final decision sample: {task}")
            population_by_task[task] += 1
            population_by_decision[str(row.get("decision") or "MISSING")] += 1
            _offer(
                global_heap,
                row,
                limit=global_n,
                score=_rank(seed, "global", uid),
            )
            _offer(
                task_heaps[task],
                row,
                limit=per_task_n,
                score=_rank(seed, f"task:{task}", uid),
            )
    if not global_heap:
        raise ValueError(f"final outputs contain no {sample_kind} decisions to audit")

    samples: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "global": _ordered(global_heap),
        **{f"task:{task}": _ordered(heap) for task, heap in sorted(task_heaps.items())},
    }
    selected_uids = {
        str(row["norm_uid"]) for values in samples.values() for _, row in values
    }
    norms: dict[str, dict[str, Any]] = {}
    for _, meta in manifest["corpora"].items():
        for row in read_jsonl(_resolve(meta["path"], manifest_path)):
            uid = str(row["norm_uid"])
            if uid in selected_uids:
                if uid in norms:
                    raise ValueError(
                        f"sample UID occurs in multiple canonical corpora: {uid}"
                    )
                norms[uid] = row
    missing_norms = selected_uids - set(norms)
    if missing_norms:
        raise ValueError(
            f"sampled UIDs absent from canonical norms: {sorted(missing_norms)[:3]}"
        )

    bank_dir = output_root / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    bank_outputs = {}
    for task in sorted({str(norms[uid]["task"]) for uid in selected_uids}):
        source = _resolve(manifest["banks"][task]["path"], manifest_path)
        destination = bank_dir / f"{task}.json"
        shutil.copyfile(source, destination)
        bank_outputs[task] = {
            "path": str(destination),
            "sha256": sha256_file(destination),
            "source_sha256": manifest["banks"][task]["source_sha256"],
        }

    outputs = {}
    for scope, values in samples.items():
        scope_name = scope.replace(":", "__")
        blind_path = output_root / f"{scope_name}.blind.jsonl"
        key_path = output_root / f"{scope_name}.key.jsonl"
        population = (
            sum(population_by_task.values())
            if scope == "global"
            else population_by_task[scope.split(":", 1)[1]]
        )
        blind, key = [], []
        for sample_rank, (_, prediction) in enumerate(values, 1):
            uid = str(prediction["norm_uid"])
            norm = norms[uid]
            blind.append(
                {
                    "norm_uid": uid,
                    "corpus": norm["corpus"],
                    "task": norm["task"],
                    "row": norm["row"],
                    "norm": norm["norm"],
                    "context": norm.get("context"),
                    "aspect": norm.get("aspect"),
                    "kind": norm.get("kind"),
                    "polarity": norm.get("polarity"),
                    "bank_file": bank_outputs[norm["task"]]["path"],
                    "bank_source_sha256": manifest["banks"][norm["task"]][
                        "source_sha256"
                    ],
                    "decision": None,
                    "metric_id": None,
                    "confidence": None,
                    "reason": None,
                }
            )
            key.append(
                {
                    "norm_uid": uid,
                    "sample_scope": scope,
                    "sample_rank": sample_rank,
                    "scope_population_decisions": population,
                    "scope_sample_n": len(values),
                    "uniform_inclusion_probability": min(1.0, len(values) / population),
                    "system_decision": prediction.get("decision"),
                    "system_confidence": prediction.get("confidence"),
                    "system_reason": prediction.get("reason"),
                    "system_metric_id": prediction.get("metric_id"),
                }
            )
        write_jsonl(blind_path, blind)
        write_jsonl(key_path, key)
        label_pack_validation = None
        if scope.startswith("task:"):
            task = scope.split(":", 1)[1]
            label_pack_validation = _write_task_label_pack(
                output_root=output_root,
                scope_name=scope_name,
                scope=scope,
                task=task,
                values=values,
                norms=norms,
                manifest=manifest,
                manifest_path=manifest_path,
                blind_path=blind_path,
                key_path=key_path,
                seed=seed,
            )
        outputs[scope] = {
            "population_decisions": population,
            "sample_n": len(values),
            "blind": {"path": str(blind_path), "sha256": sha256_file(blind_path)},
            "key": {"path": str(key_path), "sha256": sha256_file(key_path)},
            "label_pack_validation": label_pack_validation,
        }
    report = {
        "schema_version": "silver-match-v3-final-decision-sample-v2",
        "sample_kind": sample_kind,
        "sampling": "uniform_without_replacement_by_lowest_stable_sha256_rank",
        "seed": seed,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "final_inputs": {str(path): sha256_file(path) for path in final_paths},
        "unique_final_rows_seen": len(seen),
        "analysis_exclusions": {
            "inputs": dict(sorted(exclusion_inputs.items())),
            "count": len(excluded_uids),
            "excluded_final_rows_seen": len(seen & excluded_uids),
            "eligible_final_rows_seen": len(seen - excluded_uids),
        },
        "population_decisions": sum(population_by_task.values()),
        "population_by_task": dict(sorted(population_by_task.items())),
        "population_by_decision": dict(sorted(population_by_decision.items())),
        "bank_outputs": bank_outputs,
        "labeling_guide": {
            "path": str(
                Path(__file__).with_name("FINAL_DECISION_LABELING.md").resolve()
            ),
            "sha256": sha256_file(
                Path(__file__).with_name("FINAL_DECISION_LABELING.md").resolve()
            ),
        },
        "outputs": outputs,
        "blindness": (
            "auditors receive .blind files only; system decisions and sampling "
            "provenance remain in separately joined .key files"
        ),
    }
    report_path = output_root / "sample_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--final", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--global-n", type=int, default=400)
    parser.add_argument("--per-task-n", type=int, default=100)
    parser.add_argument("--seed", default="silver-match-v3-false-abstention-audit-1")
    parser.add_argument(
        "--sample-kind", choices=("match", "abstention"), default="abstention"
    )
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    report = prepare(
        manifest_path=Path(args.manifest).resolve(),
        final_paths=[Path(path).resolve() for path in args.final],
        output_root=Path(args.output_root).resolve(),
        global_n=args.global_n,
        per_task_n=args.per_task_n,
        seed=args.seed,
        sample_kind=args.sample_kind,
        exclude_paths=[Path(path).resolve() for path in args.exclude],
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
