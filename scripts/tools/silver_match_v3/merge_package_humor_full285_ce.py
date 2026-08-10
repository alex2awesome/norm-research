#!/usr/bin/env python3
"""Exact-merge Humor K200+K85 CE scores and freeze paired c2 prompts.

This deployment-only step consumes no truth labels.  It proves that each of
55,288 production norms has one score for every frozen bank metric, writes the
complete 285-metric surface, and packages a deterministic slate consisting of
CE top-16 plus every candidate above the checkpoint's frozen positive
threshold.  Original and hashed-order compact prompts are emitted together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Iterator, Mapping

from .common import normalize_space, read_jsonl, sha256_file
from .run_nemotron_ce import SCORE_META_SCHEMA, SCORE_SCHEMA


FULL_SCHEMA = "silver-match-v3-humor-ce-full285-surface-v1"
PACKAGE_SCHEMA = "silver-match-v3-humor-ce-top16-plus-positives-v1"
PROMPT_SCHEMA = "silver-match-v3-humor-c2-production-paired-prompt-v1"
REPORT_SCHEMA = "silver-match-v3-humor-full285-ce-package-report-v1"
EXPECTED_UIDS = 55_288
EXPECTED_K200 = 200
EXPECTED_K85 = 85
EXPECTED_BANK = 285
EXPECTED_K200_ROWS = EXPECTED_UIDS * EXPECTED_K200
EXPECTED_K85_ROWS = EXPECTED_UIDS * EXPECTED_K85
EXPECTED_FULL_ROWS = EXPECTED_UIDS * EXPECTED_BANK
EXPECTED_BANK_SOURCE_SHA = "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
EXPECTED_CHECKPOINT_SHA = "76a58ba823fc3895a292b71d9cbee8a1e81314dfbf9762aa111ea3b4ea1d98d2"
EXPECTED_POSITIVE_THRESHOLD = 0.9960545301437378

INSTRUCTION = """# Humor norm-to-metric adjudication

Label the explicit human criterion against only the candidate metric cards.
Use context only to resolve referent or evaluative force; do not infer a criterion
from topic alone. Polarity never changes metric identity.

Decisions: MATCH = one uniquely best leaf; MATCH_FAMILY_ONLY = the construct is
clear but leaves remain indistinguishable; NO_EXPLICIT_CRITERION = no evaluative,
prescriptive, omission, comparison, or violation force; CONTEXT_NEEDED = criterion
cannot be resolved even with context; GENERIC_VERDICT = praise/blame names no
quality dimension; NO_CANDIDATE_FITS = a specific criterion is explicit but absent
from the slate; NOISE = garbled or meaningless extraction.

For MATCH return an exact candidate metric_id. Every abstention uses metric_id null.
Prefer abstention over a merely related leaf. The reason must briefly name the
explicit criterion and the decisive sibling contrast. Return only the typed JSON.
"""


def artifact(path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, **extra}


def _meta(path: Path, expected_rows: int) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    contract = payload.get("checkpoint_contract") or {}
    if (
        payload.get("schema_version") != SCORE_META_SCHEMA
        or payload.get("output_sha256") != sha256_file(path)
        or int(payload.get("row_count", -1)) != expected_rows
        or int(payload.get("norm_group_count", -1)) != EXPECTED_UIDS
        or payload.get("classification_mode") != "binary"
        or list(payload.get("score_labels") or []) != ["REJECT", "EXACT"]
        or contract.get("checkpoint_metadata_sha256") != EXPECTED_CHECKPOINT_SHA
        or contract.get("threshold_provenance") != "checkpoint.dev"
        or float(contract.get("score_threshold", -1)) != EXPECTED_POSITIVE_THRESHOLD
    ):
        raise ValueError(f"frozen CE score metadata differs: {path}")
    return payload


def _groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    uid: str | None = None
    group: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        current = str(row.get("norm_uid") or "")
        if not current:
            raise ValueError(f"row lacks norm_uid: {path}")
        if uid is not None and current != uid:
            if uid in seen:
                raise ValueError(f"noncontiguous duplicate UID: {path}/{uid}")
            seen.add(uid)
            yield uid, group
            group = []
        uid = current
        group.append(row)
    if uid is not None:
        if uid in seen:
            raise ValueError(f"noncontiguous duplicate UID: {path}/{uid}")
        yield uid, group


def _clip(value: Any, limit: int) -> str:
    text = normalize_space(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _statement_context(query: str) -> tuple[str, str]:
    marker = "Human evaluative statement: "
    if marker not in query:
        raise ValueError("production query lacks human-statement marker")
    value = query.split(marker, 1)[1]
    evidence_marker = ". Evidence passage: "
    hint_marker = ". Weak extraction aspect hint: "
    if evidence_marker in value:
        statement, context = value.split(evidence_marker, 1)
        if hint_marker in context:
            context = context.split(hint_marker, 1)[0]
    else:
        statement = value.split(hint_marker, 1)[0]
        context = statement
    statement, context = normalize_space(statement), normalize_space(context)
    if not statement:
        raise ValueError("empty production human statement")
    return statement, context


def _prompt(statement: str, context: str, ids: list[str], bank: Mapping[str, Mapping[str, Any]]) -> str:
    cards = "\n".join(
        f"[{metric_id}] {normalize_space(bank[metric_id]['name'])} — {_clip(bank[metric_id].get('description'), 140)}"
        for metric_id in ids
    )
    return (
        f"{INSTRUCTION}\nTASK BANK: humor\n"
        f"HUMAN STATEMENT (verbatim):\n{statement}\n"
        f"CONTEXT (capped at 600 characters):\n{_clip(context, 600)}\n\n"
        f"CANDIDATE METRIC CARDS (no examples):\n{cards}\n\n"
        "Return the JSON decision now."
    )


def _hashed_order(uid: str, ids: list[str]) -> list[str]:
    result = sorted(ids, key=lambda metric: hashlib.sha256(f"{uid}\0{metric}".encode()).hexdigest())
    if len(result) > 1 and result == ids:
        result = result[1:] + result[:1]
    return result


class AtomicJsonl:
    def __init__(self, path: Path):
        self.path = path
        self.temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        self.handle: Any = None
        self.digest = hashlib.sha256()
        self.rows = 0

    def __enter__(self) -> "AtomicJsonl":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temp.open("xb")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        raw = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self.handle.write(raw); self.digest.update(raw); self.rows += 1

    def __exit__(self, typ: Any, value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.flush(); os.fsync(self.handle.fileno()); self.handle.close()
        if typ is None:
            os.replace(self.temp, self.path)
        else:
            self.temp.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    k200, k85, k85_pairs, bank_path, root = (
        Path(args.k200_scores).resolve(), Path(args.k85_scores).resolve(),
        Path(args.k85_pairs).resolve(), Path(args.bank).resolve(), Path(args.output_root).resolve(),
    )
    if root.exists():
        raise FileExistsError(root)
    _meta(k200, EXPECTED_K200_ROWS); _meta(k85, EXPECTED_K85_ROWS)
    bank_doc = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = bank_doc.get("metrics") or []
    bank_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        bank_doc.get("source_sha256") != EXPECTED_BANK_SOURCE_SHA
        or len(bank_ids) != EXPECTED_BANK or "" in bank_ids or len(set(bank_ids)) != EXPECTED_BANK
    ):
        raise ValueError("frozen Humor bank differs")
    bank = {str(row["metric_id"]): row for row in metrics}
    root.mkdir(parents=True, exist_ok=False)
    full_path, package_path, prompt_path = (
        root / "scores.full285.jsonl", root / "candidates.top16-plus-positives.jsonl",
        root / "paired_order.prompts.jsonl",
    )
    slates, positives, top_metrics = [], [], Counter()
    unique_uids: set[str] = set()
    source_groups: set[str] = set()
    iterator = zip_longest(_groups(k200), _groups(k85), _groups(k85_pairs))
    with AtomicJsonl(full_path) as full, AtomicJsonl(package_path) as package, AtomicJsonl(prompt_path) as prompts:
        for index, bundle in enumerate(iterator):
            if None in bundle:
                raise ValueError("K200/K85/pair group coverage differs")
            (uid200, rows200), (uid85, rows85), (uidp, pair_rows) = bundle  # type: ignore[misc]
            if uid200 != uid85 or uid200 != uidp or uid200 in unique_uids:
                raise ValueError(f"K200/K85 UID order differs: {uid200}/{uid85}/{uidp}")
            uid = uid200; unique_uids.add(uid)
            if len(rows200) != EXPECTED_K200 or len(rows85) != EXPECTED_K85 or len(pair_rows) != EXPECTED_K85:
                raise ValueError(f"surface depth differs: {uid}")
            by_id: dict[str, tuple[dict[str, Any], str]] = {}
            for source, rows in (("k200", rows200), ("k85", rows85)):
                for row in rows:
                    metric_id = str(row.get("metric_id") or "")
                    probabilities = row.get("probabilities") or {}
                    if (
                        row.get("schema_version") != SCORE_SCHEMA
                        or set(probabilities) != {"EXACT", "REJECT"}
                        or metric_id not in bank or metric_id in by_id
                        or str(row.get("split")) != "production"
                        or "gold_relation" in row
                    ):
                        raise ValueError(f"invalid/duplicate production score: {uid}/{metric_id}")
                    by_id[metric_id] = (row, source)
            if set(by_id) != set(bank_ids):
                raise ValueError(f"K200+K85 does not equal bank285: {uid}")
            pair_ids = {str(row.get("metric_id") or "") for row in pair_rows}
            if len(pair_ids) != EXPECTED_K85 or pair_ids != {m for m, (_, src) in by_id.items() if src == "k85"}:
                raise ValueError(f"K85 pair/score identities differ: {uid}")
            query = str(pair_rows[0].get("query") or "")
            # Pair construction preserves the original unit-separator source key,
            # while CE serialization normalizes control whitespace to spaces.
            # Norm UID and metric identity are the join keys; compare the source
            # provenance only after the same whitespace normalization.
            source_group = normalize_space(pair_rows[0].get("source_group"))
            if not query or not source_group or any(
                normalize_space(row.get("source_group")) != source_group
                for row, _ in by_id.values()
            ):
                raise ValueError(f"query/source-group contract differs: {uid}")
            source_groups.add(source_group)
            ranked = sorted(
                ((metric_id, float(row["probabilities"]["EXACT"])) for metric_id, (row, _) in by_id.items()),
                key=lambda value: (-value[1], value[0]),
            )
            positive_ids = [metric for metric, score in ranked if score >= EXPECTED_POSITIVE_THRESHOLD]
            slate_ids = list(dict.fromkeys([metric for metric, _ in ranked[: args.ce_top]] + positive_ids))
            reordered = _hashed_order(uid, slate_ids)
            positives.append(len(positive_ids)); slates.append(len(slate_ids)); top_metrics[ranked[0][0]] += 1
            for bank_index, metric_id in enumerate(bank_ids):
                row, source = by_id[metric_id]
                full.write({
                    "schema_version": FULL_SCHEMA, "task": "humor", "corpus": "humor_multi",
                    "norm_uid": uid, "source_group": source_group, "split": "production",
                    "metric_id": metric_id, "bank_index": bank_index,
                    "ce_surface_source": source, "predicted_relation": row["predicted_relation"],
                    "probabilities": row["probabilities"],
                })
            package.write({
                "schema_version": PACKAGE_SCHEMA, "task": "humor", "corpus": "humor_multi",
                "row": index, "norm_uid": uid, "source_group": source_group, "split": "production",
                "bank_source_sha256": EXPECTED_BANK_SOURCE_SHA,
                "ce_positive_threshold": EXPECTED_POSITIVE_THRESHOLD,
                "ce_positive_threshold_provenance": "checkpoint.dev",
                "ce_top1_metric_id": ranked[0][0], "ce_top1_exact_probability": ranked[0][1],
                "ce_top1_surface_source": by_id[ranked[0][0]][1],
                "ce_top2_exact_probability": ranked[1][1], "ce_top_margin": ranked[0][1] - ranked[1][1],
                "ce_positive_metric_ids": positive_ids, "candidate_metric_ids": slate_ids,
                "candidates": [
                    {"metric_id": metric, "ce_rank": rank + 1, "ce_exact_probability": score,
                     "ce_surface_source": by_id[metric][1],
                     "above_frozen_positive_threshold": metric in set(positive_ids)}
                    for rank, (metric, score) in enumerate(ranked) if metric in set(slate_ids)
                ],
            })
            statement, context = _statement_context(query)
            for order, ids in (("original", slate_ids), ("reordered", reordered)):
                prompts.write({
                    "schema_version": PROMPT_SCHEMA, "task": "humor", "corpus": "humor_multi",
                    "norm_uid": uid, "source_group": source_group, "split": "production",
                    "order_mode": order, "candidate_metric_ids": ids,
                    "conversation": [{"role": "user", "content": _prompt(statement, context, ids, bank)}],
                })
    if len(unique_uids) != EXPECTED_UIDS or full.rows != EXPECTED_FULL_ROWS or package.rows != EXPECTED_UIDS or prompts.rows != 2 * EXPECTED_UIDS:
        raise ValueError("final full-surface/package/prompt cardinality differs")
    report = {
        "schema_version": REPORT_SCHEMA, "status": "COMPLETE_EXACT_FULL285_AND_PAIRED_PROMPTS",
        "role": "production_deployment_no_truth", "test_or_blind_rows_read": 0,
        "inputs": {"k200": artifact(k200, rows=EXPECTED_K200_ROWS), "k85": artifact(k85, rows=EXPECTED_K85_ROWS),
                   "k85_pairs": artifact(k85_pairs, rows=EXPECTED_K85_ROWS), "bank": artifact(bank_path, metrics=EXPECTED_BANK)},
        "identity_audit": {"norm_uids": len(unique_uids), "rows_per_norm": EXPECTED_BANK,
                           "full_rows": full.rows, "k200_k85_disjoint_per_norm": True,
                           "k200_union_k85_equals_bank285_per_norm": True, "gold_fields_read": 0,
                           "source_groups": len(source_groups)},
        "candidate_policy": {"ce_top": args.ce_top, "plus_all_frozen_ce_positives": True,
                             "positive_threshold": EXPECTED_POSITIVE_THRESHOLD,
                             "threshold_provenance": "checkpoint.dev",
                             "slate_depth": {"min": min(slates), "median": median(slates), "mean": mean(slates), "max": max(slates)},
                             "positive_count": {"min": min(positives), "median": median(positives), "mean": mean(positives), "max": max(positives)},
                             "unique_ce_top1_metrics": len(top_metrics), "largest_ce_top1_metric_share": max(top_metrics.values()) / EXPECTED_UIDS},
        "outputs": {"full_surface": artifact(full_path, rows=full.rows), "candidate_package": artifact(package_path, rows=package.rows),
                    "paired_prompts": artifact(prompt_path, rows=prompts.rows, norm_uids=EXPECTED_UIDS)},
        "deployment_claim": "DEV_FROZEN_DEPLOYMENT_BLIND_P855",
    }
    report_path = root / "REPORT.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    print(json.dumps({"status": report["status"], "identity_audit": report["identity_audit"],
                      "candidate_policy": report["candidate_policy"], "outputs": report["outputs"]}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k200-scores", required=True); parser.add_argument("--k85-scores", required=True)
    parser.add_argument("--k85-pairs", required=True); parser.add_argument("--bank", required=True)
    parser.add_argument("--output-root", required=True); parser.add_argument("--ce-top", type=int, default=16)
    args = parser.parse_args()
    if args.ce_top != 16:
        raise ValueError("deployment contract fixes CE top depth at 16")
    build(args)


if __name__ == "__main__":
    main()
