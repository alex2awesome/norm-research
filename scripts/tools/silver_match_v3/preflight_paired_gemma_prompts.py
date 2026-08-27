#!/usr/bin/env python3
"""Token-preflight the exact truth-blind paired Gemma inference prompts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .adjudicate_gemma import iter_work_items, load_inputs, prompt_sha256, scan_candidate_input
from .common import read_jsonl, sha256_file


ORDERS = ("original", "hashed")


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _tokens(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("unexpected batched chat-template tokenization")
        value = value[0]
    return [int(token) for token in value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--context-chars", type=int, default=1400)
    parser.add_argument("--description-chars", type=int, default=520)
    parser.add_argument("--example-chars", type=int, default=180)
    parser.add_argument("--max-examples", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    paths = {
        "manifest": Path(args.manifest).resolve(),
        "candidates": Path(args.candidates).resolve(),
        "prompt": Path(args.prompt).resolve(),
    }
    addons = [Path(value).resolve() for value in args.prompt_addon]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prompt preflight: {output}")
    candidate_rows = list(read_jsonl(paths["candidates"]))
    if not candidate_rows:
        raise ValueError("empty candidate input")
    if any(len(row.get("candidates") or []) < args.max_candidates for row in candidate_rows):
        raise ValueError("candidate input is shallower than max_candidates")

    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip()
        for path in (paths["prompt"], *addons)
    ) + "\n"
    corpora, _ = scan_candidate_input(
        paths["candidates"], done=set(), shard_id=0, num_shards=1
    )
    _, norms_by_corpus, banks = load_inputs(paths["manifest"], corpora)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(args.model).resolve())
    per_order: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []
    for order in ORDERS:
        work = iter_work_items(
            paths["candidates"],
            done=set(),
            shard_id=0,
            num_shards=1,
            max_candidates=args.max_candidates,
            order_mode=order,
            system_prompt=system_prompt,
            norms_by_corpus=norms_by_corpus,
            banks=banks,
            context_chars=args.context_chars,
            description_chars=args.description_chars,
            example_chars=args.example_chars,
            max_examples=args.max_examples,
        )
        lengths: list[int] = []
        for _candidate, norm, _cards, rendered in work:
            ids = _tokens(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": rendered}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            length = len(ids)
            lengths.append(length)
            if length + args.max_tokens > args.max_model_len:
                violations.append(
                    {
                        "norm_uid": str(norm["norm_uid"]),
                        "order": order,
                        "prompt_tokens": length,
                        "prompt_plus_reserved_generation": length + args.max_tokens,
                    }
                )
        if len(lengths) != len(candidate_rows):
            raise ValueError(f"preflight work coverage mismatch for {order}")
        per_order[order] = {
            "count": len(lengths),
            "prompt_tokens_min": min(lengths),
            "prompt_tokens_max": max(lengths),
            "prompt_tokens_mean": sum(lengths) / len(lengths),
            "prompt_plus_reserved_generation_max": max(lengths) + args.max_tokens,
        }

    report = {
        "schema_version": "silver-match-v3-paired-gemma4-prompt-token-preflight-v1",
        "status": "PASS_NO_CONTEXT_OVERFLOW" if not violations else "FAIL_CONTEXT_OVERFLOW",
        "truth_read": False,
        "orders": list(ORDERS),
        "rendering": {
            "max_candidates": args.max_candidates,
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
        },
        "generation": {
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
        },
        "prompt_sha256": prompt_sha256(system_prompt),
        "per_order": per_order,
        "violation_count": len(violations),
        "violations": violations,
        "inputs": {
            "manifest": _ref(paths["manifest"]),
            "candidates": _ref(paths["candidates"]),
            "prompt_components": [_ref(path) for path in (paths["prompt"], *addons)],
            "script": _ref(Path(__file__).resolve()),
            "model": str(Path(args.model).resolve()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": _ref(output)}, sort_keys=True), flush=True)
    if violations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
