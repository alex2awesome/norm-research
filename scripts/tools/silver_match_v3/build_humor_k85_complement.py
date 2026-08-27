#!/usr/bin/env python3
"""Build the exact 85-metric complement of frozen Humor K200 production pairs.

The builder streams the immutable 77,378 x 200 pair rectangle, excludes the
22,090 hash-bound truth UIDs, and emits every bank metric absent from each
remaining norm's exact K200 identity set.  It never reads labels beyond the UID
exclusion set and creates no synthetic gold relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .common import metric_card, sha256_file


SCHEMA = "silver-match-v3-humor-k85-exact-complement-v1"
REPORT_SCHEMA = "silver-match-v3-humor-k85-exact-complement-report-v1"


def _load_truth_uids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in result:
                raise ValueError(f"missing/duplicate truth UID at line {number}")
            result.add(uid)
    return result


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(args: argparse.Namespace) -> dict[str, Any]:
    source, truth, bank, output, report_path = map(
        lambda raw: Path(raw).resolve(),
        (args.k200_pairs, args.excluded_truth, args.bank, args.output, args.report_output),
    )
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite K85 complement artifacts")
    if sha256_file(truth) != args.expected_truth_sha256:
        raise ValueError("excluded truth SHA differs")
    if sha256_file(bank) != args.expected_bank_sha256:
        raise ValueError("bank SHA differs")
    truth_uids = _load_truth_uids(truth)
    if len(truth_uids) != args.expected_truth_uids:
        raise ValueError("excluded truth UID count differs")
    bank_doc = json.loads(bank.read_text(encoding="utf-8"))
    metrics = bank_doc.get("metrics") or []
    bank_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        len(bank_ids) != args.expected_bank_metrics
        or "" in bank_ids
        or len(set(bank_ids)) != len(bank_ids)
    ):
        raise ValueError("bank metric identity contract differs")
    bank_source_sha = str(bank_doc.get("source_sha256") or "")
    cards = {str(row["metric_id"]): metric_card(row) for row in metrics}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    input_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    input_rows = output_rows = input_norms = output_norms = 0
    current_uid: str | None = None
    group: list[dict[str, Any]] = []
    truth_seen: set[str] = set()
    source_counts: Counter[str] = Counter()

    def emit(rows: list[dict[str, Any]], handle: Any) -> None:
        nonlocal input_norms, output_norms, output_rows
        if not rows:
            return
        input_norms += 1
        uid = str(rows[0].get("norm_uid") or "")
        if len(rows) != args.expected_k200:
            raise ValueError(f"K200 depth differs for {uid}: {len(rows)}")
        ids = [str(row.get("metric_id") or "") for row in rows]
        if len(set(ids)) != args.expected_k200 or not set(ids) <= set(bank_ids):
            raise ValueError(f"K200 identity set differs for {uid}")
        contracts = {
            (str(row.get("source_group") or ""), str(row.get("split") or ""),
             str(row.get("query") or ""), str(row.get("corpus") or ""),
             str(row.get("task") or ""), str(row.get("current_bank_source_sha256") or ""))
            for row in rows
        }
        if len(contracts) != 1:
            raise ValueError(f"K200 group contract differs for {uid}")
        source_group, split, query, corpus, task, supplied_bank_sha = next(iter(contracts))
        if split != "production" or task != "humor" or supplied_bank_sha != bank_source_sha:
            raise ValueError(f"production routing differs for {uid}")
        if uid in truth_uids:
            truth_seen.add(uid)
            return
        complement = [metric_id for metric_id in bank_ids if metric_id not in set(ids)]
        if len(complement) != args.expected_k85:
            raise ValueError(f"K85 complement depth differs for {uid}: {len(complement)}")
        output_norms += 1
        source_counts[source_group] += 1
        for complement_index, metric_id in enumerate(complement, 1):
            record = {
                "schema_version": SCHEMA,
                "task": task,
                "corpus": corpus,
                "norm_uid": uid,
                "source_group": source_group,
                "split": split,
                "query": query,
                "metric_id": metric_id,
                "metric_card": cards[metric_id],
                "full_bank_complement_index": complement_index,
                "complement_of_exact_k": args.expected_k200,
                "current_bank_source_sha256": bank_source_sha,
            }
            raw = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            handle.write(raw)
            output_digest.update(raw)
            output_rows += 1

    try:
        with source.open("rb") as source_handle, temporary.open("xb") as destination:
            for line_number, raw in enumerate(source_handle, 1):
                input_digest.update(raw)
                input_rows += 1
                row = json.loads(raw)
                uid = str(row.get("norm_uid") or "")
                if not uid:
                    raise ValueError(f"pair lacks norm_uid at line {line_number}")
                if current_uid is not None and uid != current_uid:
                    emit(group, destination)
                    group = []
                current_uid = uid
                group.append(row)
            emit(group, destination)
            destination.flush()
            os.fsync(destination.fileno())

        input_sha = input_digest.hexdigest()
        if input_sha != args.expected_k200_sha256:
            raise ValueError(f"K200 input SHA differs: {input_sha}")
        if input_rows != args.expected_input_rows or input_norms != args.expected_input_uids:
            raise ValueError("K200 input rectangle cardinality differs")
        if truth_seen != truth_uids:
            raise ValueError(f"truth exclusion is not a subset: missing={len(truth_uids-truth_seen)}")
        if output_norms != args.expected_output_uids or output_rows != args.expected_output_rows:
            raise ValueError("K85 output rectangle cardinality differs")
        os.replace(temporary, output)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "COMPLETE_EXACT_K85_COMPLEMENT",
            "task": "humor",
            "input_k200": {"path": str(source), "sha256": input_sha,
                           "rows": input_rows, "norm_uids": input_norms, "k": args.expected_k200},
            "excluded_truth": {"path": str(truth), "sha256": args.expected_truth_sha256,
                               "norm_uids": len(truth_uids), "labels_read": False},
            "bank": {"path": str(bank), "sha256": args.expected_bank_sha256,
                     "source_sha256": bank_source_sha, "metrics": len(bank_ids)},
            "output": {"path": str(output), "sha256": output_digest.hexdigest(),
                       "rows": output_rows, "norm_uids": output_norms, "k": args.expected_k85},
            "identity_proof": {"per_norm_k200_intersection_k85": 0,
                               "per_norm_k200_union_k85_equals_bank285": True,
                               "truth_uids_in_output": 0,
                               "source_group_counts": dict(sorted(source_counts.items()))},
            "safety": {"gold_relations_created": False, "test_or_blind_rows_read": 0,
                       "create_only": True},
        }
        _write_report(report_path, report)
        return report
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k200-pairs", required=True)
    parser.add_argument("--expected-k200-sha256", required=True)
    parser.add_argument("--excluded-truth", required=True)
    parser.add_argument("--expected-truth-sha256", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--expected-bank-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--expected-input-rows", type=int, default=15_475_600)
    parser.add_argument("--expected-input-uids", type=int, default=77_378)
    parser.add_argument("--expected-truth-uids", type=int, default=22_090)
    parser.add_argument("--expected-output-uids", type=int, default=55_288)
    parser.add_argument("--expected-output-rows", type=int, default=4_699_480)
    parser.add_argument("--expected-bank-metrics", type=int, default=285)
    parser.add_argument("--expected-k200", type=int, default=200)
    parser.add_argument("--expected-k85", type=int, default=85)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
