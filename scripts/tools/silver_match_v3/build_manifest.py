#!/usr/bin/env python3
"""Freeze canonical norm and metric universes with stable per-norm IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .common import (
    extract_norm,
    extract_source_id,
    normalize_name,
    normalize_space,
    schema_record,
    sha256_file,
    stable_uid,
    write_jsonl,
)
from .config import (
    CANONICAL_SOURCES,
    CORPUS_TO_TASK,
    CORPUS_ALIASES,
    DEFAULT_HOME,
    DEFAULT_DATA_ROOT,
    DEFAULT_HIERARCHY_ROOT,
    DEFAULT_OUTPUT_ROOT,
    TASK_TO_HIERARCHY,
)


def load_bank(task: str, hierarchy_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    groups = payload.get("merged_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"no merged_groups in {hierarchy_path}")

    bank = []
    name_counts = Counter(
        normalize_name(group.get("merged_name")) for group in groups
    )
    for idx, group in enumerate(groups):
        name = normalize_space(group.get("merged_name"))
        description = normalize_space(group.get("merged_description"))
        if not name or not description:
            raise ValueError(f"missing name/description at {hierarchy_path} metric {idx}")
        name_key = normalize_name(name)
        leaves = group.get("all_leaves") or []
        examples = []
        for leaf in leaves:
            if isinstance(leaf, dict):
                example = normalize_space(leaf.get("name"))
                if example and normalize_name(example) != name_key and example not in examples:
                    examples.append(example)
            if len(examples) >= 8:
                break
        bank.append(
            schema_record(
                task=task,
                metric_id=f"a{idx}",
                metric_index=idx,
                name=name,
                name_key=name_key,
                # Some current banks intentionally contain same-name leaves
                # with distinct descriptions. IDs, never names, are canonical.
                name_ambiguous=name_counts[name_key] > 1,
                description=description,
                examples=examples,
                leaf_count=int(group.get("total_leaf_rubrics") or len(leaves)),
                source_r2_cluster_ids=group.get("source_r2_cluster_ids") or [],
            )
        )
    return bank


def canonical_norms(corpus: str, task: str, source_path: Path):
    with source_path.open("r", encoding="utf-8", errors="replace") as handle:
        for physical_row, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source_path}:{physical_row + 1}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object at {source_path}:{physical_row + 1}")
            norm = extract_norm(record)
            if not norm:
                raise ValueError(f"missing norm text at {source_path}:{physical_row + 1}")
            source_id = extract_source_id(record, physical_row)
            yield schema_record(
                # Canonical identity is per extracted row, not the historical
                # document ID (which repeats and caused the v1 overwrite bug).
                # Keeping the exact quote in the hash catches row drift.
                norm_uid=stable_uid(corpus, physical_row, norm),
                corpus=corpus,
                task=task,
                row=physical_row,
                source_id=source_id,
                norm=norm,
                aspect=normalize_space(record.get("aspect")) or None,
                polarity=normalize_space(record.get("polarity")) or None,
                kind=normalize_space(record.get("kind")) or None,
                paper_id=normalize_space(record.get("paper_id")) or None,
            )


def _nonempty_records(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for physical_row, line in enumerate(handle):
            line = line.strip()
            if line:
                yield physical_row, json.loads(line)


def _segment_record_counts(deploy_path: Path, score_path: Path) -> tuple[int, int]:
    """Return signal-bearing deploy and nonempty score counts.

    Tail judges append their scores while they run.  Merely observing both
    files is therefore not evidence that an optional segment is complete.
    Counting first lets manifest builds proceed safely during a live judge:
    the partial tail is reported as incomplete and excluded, rather than
    being mistaken for a complete segment or leaving a half-built manifest.
    The full positional/content checks still run in ``scored_deploy_norms``
    once the counts agree.
    """
    deploy_count = sum(
        1 for _, record in _nonempty_records(deploy_path) if record.get("signals")
    )
    score_count = sum(1 for _ in _nonempty_records(score_path))
    return deploy_count, score_count


def _source_id(record: dict[str, Any]) -> str:
    for key in (
        "unit_id",
        "thread_id",
        "pair_id",
        "source_id",
        "id",
        "review_id",
    ):
        if record.get(key) is not None:
            return normalize_space(record[key])
    return ""


def scored_deploy_norms(
    corpus: str,
    task: str,
    segments: list[tuple[Path, Path]],
):
    """Reconstruct accepted GEPA signals without losing deploy metadata."""
    output_row = 0
    for segment_index, (deploy_path, score_path) in enumerate(segments):
        deploy_iter = (
            (physical_row, record)
            for physical_row, record in _nonempty_records(deploy_path)
            if record.get("signals")
        )
        score_iter = _nonempty_records(score_path)
        paired = 0
        for (score_row, score), (deploy_row, deploy) in zip(score_iter, deploy_iter):
            paired += 1
            score_id, deploy_id = _source_id(score), _source_id(deploy)
            if not score_id or score_id != deploy_id:
                raise ValueError(
                    f"source mismatch {score_path}:{score_row + 1} / "
                    f"{deploy_path}:{deploy_row + 1}: {score_id!r} != {deploy_id!r}"
                )
            judged = score.get("scored") or []
            raw_signals = deploy.get("signals") or []
            if len(judged) != len(raw_signals):
                raise ValueError(
                    f"signal count mismatch for {corpus}/{score_id}: "
                    f"{len(judged)} != {len(raw_signals)}"
                )
            for signal_index, (judge, raw) in enumerate(zip(judged, raw_signals)):
                norm = normalize_space(raw.get("signal_text"))
                judged_norm = normalize_space(judge.get("signal_text"))
                raw_passage = str(raw.get("passage_text") or "")
                judged_passage = str(judge.get("passage_text") or "")
                if norm != judged_norm or judged_passage != raw_passage[:200]:
                    raise ValueError(
                        f"candidate mismatch for {corpus}/{score_id}/{signal_index}"
                    )
                faithful = int(judge.get("faithful") or 0)
                valid = int(judge.get("valid") or 0)
                # The extraction judge's old ``valid`` rubric was narrower
                # than the user's silver doctrine: it rejected explicit
                # clarity, evidence, logic, scope, and editing norms as
                # "not substantive legal reasoning."  Independent audit found
                # 66.5% positives in a balanced rejected-tail pack.  Preserve
                # every grounded human statement (faithful=1) and defer norm
                # typing/bank fit to the abstention-aware matcher.  Hallucinated
                # or ungrounded extractions (faithful=0) remain excluded.
                if faithful != 1:
                    continue
                yield schema_record(
                    norm_uid=stable_uid(
                        corpus, segment_index, score_row, score_id, signal_index, norm
                    ),
                    corpus=corpus,
                    task=task,
                    row=output_row,
                    source_id=score_id,
                    source_segment=segment_index,
                    source_record_row=score_row,
                    source_signal_index=signal_index,
                    norm=norm,
                    context=normalize_space(raw_passage) or None,
                    aspect=None,
                    polarity=normalize_space(raw.get("polarity")) or None,
                    kind=normalize_space(raw.get("signal_type")) or None,
                    paper_id=None,
                    extraction_judge_reason=normalize_space(judge.get("reason")) or None,
                    extraction_faithful=faithful,
                    extraction_valid=valid,
                    extraction_inclusion_policy="faithful_grounding; norm validity deferred to matcher",
                )
                output_row += 1
        # Detect a silently truncated side of the positional join.
        try:
            next(score_iter)
        except StopIteration:
            pass
        else:
            raise ValueError(f"score has extra records after {paired} pairs: {score_path}")
        try:
            next(deploy_iter)
        except StopIteration:
            pass
        else:
            raise ValueError(f"deploy has extra signal records after {paired} pairs: {deploy_path}")


def _context_window(text: str, quote: str, radius: int = 700) -> str:
    # Extractors normalize whitespace in signal_text.  Ground against the same
    # normalized representation so doubled spaces/newlines in the source do
    # not turn a verbatim quote into a false missing-evidence error.
    text = normalize_space(text)
    quote = normalize_space(quote)
    start = text.find(quote)
    if start < 0:
        return ""
    lo = max(0, start - radius)
    hi = min(len(text), start + len(quote) + radius)
    return normalize_space(text[lo:hi])


def review_feedback_norms(
    corpus: str,
    task: str,
    signals_path: Path,
    reviews_path: Path,
):
    reviews: dict[str, str] = {}
    for _, record in _nonempty_records(reviews_path):
        review_id = normalize_space(record.get("review_id"))
        text = str(record.get("review_text") or record.get("text") or "")
        if review_id and text:
            reviews[review_id] = text
    for physical_row, record in _nonempty_records(signals_path):
        norm = extract_norm(record)
        source_id = normalize_space(record.get("source_id"))
        review = reviews.get(source_id)
        if review is None:
            raise KeyError(f"review source missing for {corpus}:{physical_row}: {source_id}")
        context = _context_window(review, norm)
        if not context:
            raise ValueError(
                f"peer-review quote is not grounded for {corpus}:{physical_row}: {source_id}"
            )
        yield schema_record(
            norm_uid=stable_uid(corpus, physical_row, source_id, norm),
            corpus=corpus,
            task=task,
            row=physical_row,
            source_id=source_id,
            source_segment=0,
            source_record_row=physical_row,
            source_signal_index=0,
            norm=norm,
            context=context,
            aspect=normalize_space(record.get("aspect")) or None,
            polarity=normalize_space(record.get("polarity")) or None,
            kind=normalize_space(record.get("kind")) or None,
            paper_id=normalize_space(record.get("paper_id")) or None,
            extraction_judge_reason="certified Gemma-4+GEPA peer-review extraction",
            extraction_faithful=1,
            extraction_valid=1,
            extraction_inclusion_policy="certified peer-review extraction; norm validity deferred to matcher",
            evidence_grounding="normalized_exact_substring",
        )


def build(args: argparse.Namespace) -> dict[str, Any]:
    home = Path(args.home)
    data_root = Path(args.data_root)
    hierarchy_root = Path(args.hierarchy_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = sorted(set(CORPUS_TO_TASK.values()))
    bank_meta: dict[str, Any] = {}
    for task in tasks:
        hierarchy_path = hierarchy_root / TASK_TO_HIERARCHY[task]
        if not hierarchy_path.exists():
            raise FileNotFoundError(hierarchy_path)
        bank = load_bank(task, hierarchy_path)
        out_path = output_root / "banks" / f"{task}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "task": task,
                    "source_path": str(hierarchy_path),
                    "source_sha256": sha256_file(hierarchy_path),
                    "metrics": bank,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        bank_meta[task] = {
            "path": str(out_path),
            "count": len(bank),
            "source_path": str(hierarchy_path),
            "source_sha256": sha256_file(hierarchy_path),
        }

    corpus_meta: dict[str, Any] = {}
    all_uids: set[str] = set()
    if args.source_mode == "legacy_signals":
        source_specs = {
            corpus: {"task": task, "type": "legacy_signals"}
            for corpus, task in CORPUS_TO_TASK.items()
        }
    else:
        source_specs = CANONICAL_SOURCES

    for corpus, spec in sorted(source_specs.items()):
        task = spec["task"]
        source_paths: list[Path] = []
        coverage_complete = True
        missing_optional: list[dict[str, str]] = []
        if spec.get("type") == "review_feedback":
            signals_path = home / spec["signals"]
            reviews_path = home / spec["reviews"]
            for path in (signals_path, reviews_path):
                if not path.exists():
                    raise FileNotFoundError(path)
            source_paths = [signals_path, reviews_path]
            row_iter = review_feedback_norms(
                corpus, task, signals_path, reviews_path
            )
        elif spec.get("type") == "legacy_signals":
            source_path = data_root / corpus / f"signals_{corpus}.jsonl"
            if not source_path.exists():
                raise FileNotFoundError(source_path)
            source_paths = [source_path]
            row_iter = canonical_norms(corpus, task, source_path)
        else:
            gepa_dir = home / "data" / spec["gepa"] / "gepa"
            segments = [(gepa_dir / spec["deploy"], gepa_dir / spec["score"])]
            for optional in spec.get("optional_segments", []):
                pair = (gepa_dir / optional["deploy"], gepa_dir / optional["score"])
                optional_status: dict[str, Any] = {
                    "deploy": str(pair[0]),
                    "score": str(pair[1]),
                }
                if all(path.exists() for path in pair):
                    deploy_count, score_count = _segment_record_counts(*pair)
                    optional_status.update(
                        deploy_signal_records=deploy_count,
                        score_records=score_count,
                    )
                    if deploy_count == score_count:
                        segments.append(pair)
                        continue
                    optional_status["reason"] = "score_in_progress_or_truncated"
                else:
                    optional_status["reason"] = "file_missing"
                if pair not in segments:
                    coverage_complete = False
                    missing_optional.append(optional_status)
            for deploy_path, score_path in segments:
                if not deploy_path.exists():
                    raise FileNotFoundError(deploy_path)
                if not score_path.exists():
                    raise FileNotFoundError(score_path)
                source_paths.extend((deploy_path, score_path))
            row_iter = scored_deploy_norms(corpus, task, segments)
        out_path = output_root / "norms" / f"{corpus}.jsonl"
        rows = list(row_iter)
        duplicate_source_ids = sum(
            count > 1 for count in Counter(row["source_id"] for row in rows).values()
        )
        for row in rows:
            if row["norm_uid"] in all_uids:
                raise ValueError(f"duplicate norm_uid: {row['norm_uid']}")
            all_uids.add(row["norm_uid"])
        write_jsonl(out_path, rows)
        corpus_meta[corpus] = {
            "task": task,
            "path": str(out_path),
            "count": len(rows),
            "source_paths": [str(path) for path in source_paths],
            "source_sha256": {
                str(path): sha256_file(path) for path in source_paths
            },
            "coverage_complete": coverage_complete,
            "missing_optional_segments": missing_optional,
            "expected_unjudged_candidates": (
                int(spec.get("expected_unjudged_candidates") or 0)
                if not coverage_complete
                else 0
            ),
            "duplicated_source_id_groups": duplicate_source_ids,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_mode": args.source_mode,
        "extraction_inclusion_policy": (
            "faithful_grounding; norm validity deferred to abstention-aware matcher"
            if args.source_mode == "canonical"
            else "legacy signal calibration only"
        ),
        "routing": {
            corpus: spec["task"] for corpus, spec in sorted(source_specs.items())
        },
        "aliases": CORPUS_ALIASES if args.source_mode == "canonical" else {},
        "total_norms": sum(v["count"] for v in corpus_meta.values()),
        "total_corpora": len(corpus_meta),
        "total_tasks": len(bank_meta),
        "banks": bank_meta,
        "corpora": corpus_meta,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=str(DEFAULT_HOME))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--hierarchy-root", default=str(DEFAULT_HIERARCHY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--source-mode",
        choices=("canonical", "legacy_signals"),
        default="canonical",
    )
    return parser.parse_args()


def main() -> None:
    manifest = build(parse_args())
    print(
        json.dumps(
            {
                "total_norms": manifest["total_norms"],
                "total_corpora": manifest["total_corpora"],
                "total_tasks": manifest["total_tasks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
