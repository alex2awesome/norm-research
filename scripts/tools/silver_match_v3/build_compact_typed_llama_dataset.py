#!/usr/bin/env python3
"""Build and token-audit the frozen compact Humor typed-Llama dataset once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from .train_gemma4_typed_lora import tokenize_example


SCHEMA = "silver-match-v3-humor-compact-typed-llama-dataset-v1"
EXPECTED_BANK_SHA = "1b4a29d34b4ef4d999e0cb0b2d1125286372349ff6dfa21a6adc5bc8e76f0de9"
CARD_RE = re.compile(
    r"^\[([^\]]+)\] ([^\n]+)\n  Definition: (.*?)(?=\n  Bank examples:)",
    re.MULTILINE | re.DOTALL,
)
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


def _clip(value: str, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        raise ValueError(f"prompt lacks compacting marker: {start!r}/{end!r}")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _cards(text: str, candidate_ids: Sequence[str]) -> list[dict[str, str]]:
    section = _between(text, "CANDIDATE METRIC CARDS:\n", "\n\nReturn the JSON decision now.")
    parsed: dict[str, dict[str, str]] = {}
    for metric_id, name, definition in CARD_RE.findall(section):
        if metric_id in parsed:
            raise ValueError(f"duplicate candidate card: {metric_id}")
        parsed[metric_id] = {
            "metric_id": metric_id,
            "name": " ".join(name.split()),
            "definition": _clip(definition, 140),
        }
    expected = [str(value) for value in candidate_ids]
    if set(parsed) != set(expected) or len(parsed) != len(expected):
        raise ValueError(
            f"candidate card parse differs: parsed={sorted(parsed)}, expected={sorted(expected)}"
        )
    return [parsed[metric_id] for metric_id in expected]


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = list(row.get("messages") or [])
    if (
        len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("expected exactly one user and one assistant message")
    original = str(messages[0].get("content") or "")
    norm = _between(
        original,
        "HUMAN STATEMENT:",
        "\nEVIDENCE PASSAGE FROM THE HUMAN FEEDBACK:",
    )
    context = _between(
        original,
        "EVIDENCE PASSAGE FROM THE HUMAN FEEDBACK:",
        "\nEXTRACTED POLARITY (does not determine metric):",
    )
    candidate_ids = [str(value) for value in row.get("candidate_metric_ids") or []]
    cards = _cards(original, candidate_ids)
    if not norm:
        raise ValueError("human statement is empty")
    if row.get("decision") == "MATCH" and str(row.get("metric_id") or "") not in candidate_ids:
        raise ValueError("MATCH target is absent from frozen candidate slate")
    card_text = "\n".join(
        f"[{card['metric_id']}] {card['name']} — {card['definition']}" for card in cards
    )
    compact_user = (
        f"{INSTRUCTION}\nTASK BANK: humor\n"
        f"HUMAN STATEMENT (verbatim):\n{norm}\n"
        f"CONTEXT (capped at 600 characters):\n{_clip(context, 600)}\n\n"
        f"CANDIDATE METRIC CARDS (no examples):\n{card_text}\n\n"
        "Return the JSON decision now."
    )
    if "Bank examples:" in compact_user:
        raise AssertionError("bank example leaked into compact prompt")
    result = dict(row)
    result["messages"] = [
        {"role": "user", "content": compact_user},
        dict(messages[1]),
    ]
    result["compact_prompt_contract"] = {
        "human_statement_full": True,
        "context_character_cap": 600,
        "candidate_definition_character_cap": 140,
        "candidate_bank_examples_included": False,
        "assistant_target_unchanged": True,
    }
    return result


def _stream_split(
    *,
    source: Path,
    output: Path,
    expected_sha: str,
    expected_rows: int,
    expected_uids: int,
    expected_split: str,
    tokenizer: Any,
) -> tuple[dict[str, Any], set[str], set[str]]:
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    assistant_source_digest = hashlib.sha256()
    assistant_output_digest = hashlib.sha256()
    rows = 0
    uids: set[str] = set()
    groups: set[str] = set()
    lengths: Counter[int] = Counter()
    max_row: dict[str, Any] | None = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            for raw in input_handle:
                source_digest.update(raw)
                row = json.loads(raw)
                uid = str(row.get("norm_uid") or "")
                group = str(row.get("source_group") or "")
                if (
                    not uid
                    or not group
                    or row.get("split") != expected_split
                    or bool(row.get("gradient_eligible")) != (expected_split == "train")
                    or row.get("current_bank_source_sha256") != EXPECTED_BANK_SHA
                ):
                    raise ValueError(f"role/identity contract differs: {uid}")
                compact = _compact_row(row)
                source_target = str(row["messages"][-1]["content"])
                compact_target = str(compact["messages"][-1]["content"])
                assistant_source_digest.update(source_target.encode("utf-8"))
                assistant_output_digest.update(compact_target.encode("utf-8"))
                if compact_target != source_target:
                    raise AssertionError(f"assistant target changed: {uid}")
                encoded = tokenize_example(tokenizer, compact, 2048)
                length = int(encoded["length"])
                lengths[length] += 1
                if max_row is None or length > max_row["tokens"]:
                    max_row = {"norm_uid": uid, "view": compact.get("view"), "tokens": length}
                rendered = (
                    json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                output_handle.write(rendered)
                output_digest.update(rendered)
                rows += 1
                uids.add(uid)
                groups.add(group)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if source_digest.hexdigest() != expected_sha:
            raise AssertionError("streamed source SHA changed during audit")
        if rows != expected_rows or len(uids) != expected_uids:
            raise ValueError(
                f"{expected_split} cardinality differs: rows={rows}, uids={len(uids)}"
            )
        if assistant_source_digest.hexdigest() != assistant_output_digest.hexdigest():
            raise AssertionError("aggregate assistant target bytes changed")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return (
        {
            "source": {"path": str(source), "sha256": expected_sha},
            "output": {
                "path": str(output),
                "sha256": output_digest.hexdigest(),
                "rows": rows,
                "unique_norm_uids": len(uids),
                "source_groups": len(groups),
            },
            "token_audit": {
                "tokenizer": str(tokenizer.name_or_path),
                "max_allowed_tokens": 2048,
                "maximum": max_row,
                "all_rows_within_limit": True,
                "length_histogram": dict(sorted(lengths.items())),
            },
            "assistant_target_aggregate_sha256": assistant_output_digest.hexdigest(),
        },
        uids,
        groups,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    report_path = output_root / "COMPACT_DATASET_REPORT.json"
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train, train_uids, train_groups = _stream_split(
        source=Path(args.train).resolve(),
        output=output_root / "train.jsonl",
        expected_sha=args.train_sha256,
        expected_rows=34_944,
        expected_uids=17_472,
        expected_split="train",
        tokenizer=tokenizer,
    )
    dev, dev_uids, dev_groups = _stream_split(
        source=Path(args.dev).resolve(),
        output=output_root / "dev.jsonl",
        expected_sha=args.dev_sha256,
        expected_rows=2_119,
        expected_uids=2_119,
        expected_split="dev",
        tokenizer=tokenizer,
    )
    if train_uids & dev_uids or train_groups & dev_groups:
        raise ValueError("compact train/dev source-disjoint contract failed")
    report = {
        "schema_version": SCHEMA,
        "status": "COMPLETE_SINGLE_STREAMING_TOKEN_AUDIT_MAX2048",
        "task": "humor",
        "train": train,
        "dev": dev,
        "split_audit": {
            "norm_uid_overlap": 0,
            "source_group_overlap": 0,
            "test_or_blind_rows_read": 0,
        },
        "prompt_contract": {
            "human_statement_full": True,
            "context_character_cap": 600,
            "candidate_metric_ids_names_definitions": True,
            "candidate_definition_character_cap": 140,
            "bank_examples_removed": True,
            "target_assistant_json_unchanged": True,
        },
    }
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
