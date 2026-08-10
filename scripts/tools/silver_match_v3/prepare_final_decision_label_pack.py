#!/usr/bin/env python3
"""Turn one truth-hidden final-decision sample into an immutable label pack.

The sampler deliberately writes system decisions to a separate ``.key`` file.
This command reads only the sample report, its declared blind file, the frozen
bank, and canonical norms.  It never opens the key, final predictions, or any
prior audit label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _resolve(value: str | Path, anchor: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else anchor.parent / path


def _stable_key(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def prepare(
    *,
    sample_report_path: Path,
    scope: str,
    output_root: Path,
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    if chunk_size < 1 or chunk_size > 25:
        raise ValueError("chunk_size must be in [1, 25]")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty pack: {output_root}")
    report = json.loads(sample_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "silver-match-v3-final-decision-sample-v1":
        raise ValueError("not a final-decision sample report")
    if report.get("sample_kind") not in {"match", "abstention"}:
        raise ValueError("unknown final-decision sample kind")
    sample_meta = (report.get("outputs") or {}).get(scope)
    if not sample_meta:
        raise KeyError(f"scope absent from sample report: {scope}")
    blind_path = _resolve(sample_meta["blind"]["path"], sample_report_path)
    if sha256_file(blind_path) != sample_meta["blind"]["sha256"]:
        raise ValueError("blind sample hash mismatch")
    rows = list(read_jsonl(blind_path))
    if len(rows) != int(sample_meta["sample_n"]):
        raise ValueError("blind sample count mismatch")
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("blind sample has missing/duplicate UIDs")
    hidden_fields = ("decision", "metric_id", "confidence", "reason")
    for row in rows:
        if any(row.get(field) is not None for field in hidden_fields):
            raise ValueError(f"blind row leaks a label field: {row.get('norm_uid')}")
    tasks = {str(row.get("task") or "") for row in rows}
    if len(tasks) != 1 or "" in tasks:
        raise ValueError("one label pack must contain exactly one task")
    task = next(iter(tasks))
    bank_meta = (report.get("bank_outputs") or {}).get(task)
    if not bank_meta:
        raise KeyError(f"sample report lacks bank for {task}")
    bank_path = _resolve(bank_meta["path"], sample_report_path)
    if sha256_file(bank_path) != bank_meta["sha256"]:
        raise ValueError("sample bank hash mismatch")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if str(bank.get("source_sha256") or "") != str(bank_meta["source_sha256"]):
        raise ValueError("sample bank source identity mismatch")
    metric_ids = [str(metric["metric_id"]) for metric in bank.get("metrics") or []]
    if not metric_ids or len(metric_ids) != len(set(metric_ids)):
        raise ValueError("sample bank has missing/duplicate metric IDs")

    manifest_path = _resolve(report["manifest"], sample_report_path)
    if sha256_file(manifest_path) != report["manifest_sha256"]:
        raise ValueError("sample manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wanted = set(uids)
    canonical: dict[str, dict[str, Any]] = {}
    for _, corpus_meta in manifest["corpora"].items():
        if corpus_meta.get("task") != task:
            continue
        corpus_path = _resolve(corpus_meta["path"], manifest_path)
        for norm in read_jsonl(corpus_path):
            uid = str(norm["norm_uid"])
            if uid in wanted:
                if uid in canonical:
                    raise ValueError(f"sample UID duplicated across corpora: {uid}")
                canonical[uid] = norm
    missing = wanted - set(canonical)
    if missing:
        raise ValueError(f"sample UIDs absent from canonical norms: {sorted(missing)[:3]}")

    items = []
    for blind in rows:
        uid = str(blind["norm_uid"])
        norm = canonical[uid]
        for field in ("corpus", "task", "row", "norm"):
            if blind.get(field) != norm.get(field):
                raise ValueError(f"blind/canonical provenance mismatch for {field}: {uid}")
        group = split_group_for(norm)
        items.append(
            {
                **blind,
                "split_group": group,
                "split": split_for(group),
                "source_group": group,
                "boundary_stratum": f"final_{report['sample_kind']}_uniform",
                "final_audit_scope": scope,
                "permanently_excluded_from_gradients": True,
            }
        )

    permuted_bank = {
        **bank,
        "metrics": sorted(
            bank["metrics"],
            key=lambda metric: (
                _stable_key(seed, "bank", str(metric["metric_id"])),
                str(metric["metric_id"]),
            ),
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    items_path = output_root / "items.jsonl"
    bank_output = output_root / "bank.json"
    write_jsonl(items_path, items)
    bank_output.write_text(
        json.dumps(permuted_bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    chunks = []
    for start in range(0, len(items), chunk_size):
        path = output_root / "chunks" / f"part-{start // chunk_size:03d}.jsonl"
        write_jsonl(path, items[start : start + chunk_size])
        chunks.append(path)
    validation = {
        "schema_version": "silver-match-v3-final-decision-label-pack-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "count": len(items),
        "unique_source_groups": len({row["source_group"] for row in items}),
        "chunk_size": chunk_size,
        "chunk_count": len(chunks),
        "bank_metric_count": len(metric_ids),
        "bank_source_sha256": bank_meta["source_sha256"],
        "sample_kind": report["sample_kind"],
        "sample_scope": scope,
        "truth_hidden": True,
        "system_decisions_hidden": True,
        "system_key_not_read": True,
        "permanently_excluded_from_gradients": True,
        "inputs": {
            "sample_report": {
                "path": str(sample_report_path),
                "sha256": sha256_file(sample_report_path),
            },
            "blind_sample": {"path": str(blind_path), "sha256": sha256_file(blind_path)},
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        },
        "outputs": {
            "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
            "bank": {"path": str(bank_output), "sha256": sha256_file(bank_output)},
            "chunks": {str(path): sha256_file(path) for path in chunks},
        },
    }
    validation_path = output_root / "validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-report", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=57721)
    args = parser.parse_args()
    report = prepare(
        sample_report_path=Path(args.sample_report).resolve(),
        scope=args.scope,
        output_root=Path(args.output_root).resolve(),
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
