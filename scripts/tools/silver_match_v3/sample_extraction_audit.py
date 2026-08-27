#!/usr/bin/env python3
"""Build deterministic accepted/rejected packs for extraction-judge audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, write_jsonl


ID_FIELDS = ("unit_id", "thread_id", "pair_id", "source_id", "review_id", "id")
TEXT_FIELDS = ("text", "review_text", "article_text", "comment", "body", "content")


def first_value(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and normalize_space(value):
            return normalize_space(value)
    return ""


def context_around(text: str, passage: str, radius: int = 700) -> str:
    if not text:
        return ""
    start = text.find(passage) if passage else -1
    if start < 0:
        return text[: radius * 2]
    return text[max(0, start - radius) : start + len(passage) + radius]


def audit_rows(
    deploy_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    source_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    signal_deploy = [row for row in deploy_rows if row.get("signals")]
    if len(signal_deploy) != len(score_rows):
        raise ValueError(
            f"incomplete positional pair: deploy={len(signal_deploy)} score={len(score_rows)}"
        )
    output = []
    for deploy, score in zip(signal_deploy, score_rows):
        deploy_id, score_id = first_value(deploy, ID_FIELDS), first_value(score, ID_FIELDS)
        if not deploy_id or deploy_id != score_id:
            raise ValueError(f"source mismatch: {deploy_id!r} != {score_id!r}")
        signals, judged = deploy.get("signals") or [], score.get("scored") or []
        if len(signals) != len(judged):
            raise ValueError(f"signal mismatch for {deploy_id}")
        for signal_index, (signal, verdict) in enumerate(zip(signals, judged)):
            signal_text = normalize_space(signal.get("signal_text"))
            if signal_text != normalize_space(verdict.get("signal_text")):
                raise ValueError(f"signal text mismatch for {deploy_id}/{signal_index}")
            passage = normalize_space(signal.get("passage_text"))
            accepted = bool(verdict.get("faithful")) and bool(verdict.get("valid"))
            output.append(
                {
                    "audit_id": hashlib.sha256(
                        f"{deploy_id}\0{signal_index}\0{signal_text}".encode("utf-8")
                    ).hexdigest(),
                    "source_id": deploy_id,
                    "signal_index": signal_index,
                    "signal_text": signal_text,
                    "passage_text": passage,
                    "source_context": context_around(source_by_id.get(deploy_id, ""), passage),
                    "judge_accepted": accepted,
                    "judge_faithful": int(bool(verdict.get("faithful"))),
                    "judge_valid": int(bool(verdict.get("valid"))),
                    "judge_reason": normalize_space(verdict.get("reason")),
                    "human_verdict": None,
                    "human_reason": None,
                }
            )
    return output


def deterministic_sample(rows: list[dict[str, Any]], accepted_n: int, rejected_n: int) -> list[dict]:
    accepted = sorted((row for row in rows if row["judge_accepted"]), key=lambda r: r["audit_id"])
    rejected = sorted((row for row in rows if not row["judge_accepted"]), key=lambda r: r["audit_id"])
    chosen = accepted[:accepted_n] + rejected[:rejected_n]
    return sorted(chosen, key=lambda row: (not row["judge_accepted"], row["audit_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--accepted", type=int, default=100)
    parser.add_argument("--rejected", type=int, default=100)
    args = parser.parse_args()

    source_by_id = {}
    for row in read_jsonl(Path(args.source)):
        source_id = first_value(row, ID_FIELDS)
        if source_id:
            source_by_id[source_id] = first_value(row, TEXT_FIELDS)
    rows = audit_rows(
        list(read_jsonl(Path(args.deploy))),
        list(read_jsonl(Path(args.score))),
        source_by_id,
    )
    selected = deterministic_sample(rows, args.accepted, args.rejected)
    write_jsonl(Path(args.output), selected)
    print(
        json.dumps(
            {
                "signals": len(rows),
                "judge_accepted": sum(row["judge_accepted"] for row in rows),
                "audit_rows": len(selected),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
