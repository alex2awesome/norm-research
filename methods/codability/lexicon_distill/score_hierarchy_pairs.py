#!/usr/bin/env python3
"""Score frozen hierarchy pair inputs with one Gemma similarity adapter on sk2."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import LABELS, render_prompt
from .hierarchy_contracts import (
    PAIR_OUTPUT_SCHEMA,
    pair_input_sha256,
    sha256_file,
    validate_pair_files,
    validate_pair_input,
    validate_pair_output,
)
from .train_gemma4_similarity_lora import (
    _token_list,
    collate,
    label_token_ids,
)
from .evaluate_similarity_lora import _load_model, assert_sk2_host


def _read_inputs(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                rows.append(validate_pair_input(json.loads(line), context=f"{path}:{line_number}"))
    validate_pair_files(path)
    return rows


def _protocol_bundle(path: Path, protocol_id: str) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get(protocol_id)
    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
        raise ValueError(f"protocol bundle lacks {protocol_id}")
    text = row["text"]
    expected = row.get("sha256")
    import hashlib
    observed = hashlib.sha256(text.encode()).hexdigest()
    if expected != observed:
        raise ValueError(f"protocol text/hash mismatch for {protocol_id}")
    return text, observed


def _encode(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    protocol_text: str,
    *,
    max_length: int,
) -> tuple[list[dict[str, Any]], tuple[int, int, int]]:
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Return exactly 0, 1, or 2.\nLABEL:"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    label_ids = label_token_ids(tokenizer, probe)
    encoded = []
    for row in rows:
        for view, text_a, text_b in (
            ("ab", row["text_a"], row["text_b"]),
            ("ba", row["text_b"], row["text_a"]),
        ):
            user_text = render_prompt(protocol_text, row["task"], text_a, text_b)
            user_text += "\n\nLABEL MAP: 0=DIFFERENT, 1=RELATED, 2=SAME.\nLABEL:"
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = _token_list(tokenizer(prompt, add_special_tokens=False))
            if not ids or len(ids) > max_length:
                raise ValueError(f"prompt length outside 1..{max_length}: {row['pair_id']}/{view}")
            if label_token_ids(tokenizer, prompt) != label_ids:
                raise ValueError(f"label token drift: {row['pair_id']}/{view}")
            encoded.append({
                "input_ids": ids,
                "target_probs": [1.0, 0.0, 0.0],
                "weight": 0.5,
                "example_id": row["pair_id"],
                "task": row["task"],
                "view": view,
            })
    return encoded, label_ids


def _label(probabilities: Mapping[str, float]) -> str:
    return max(LABELS, key=lambda label: probabilities[label])


def assemble_outputs(
    inputs: Sequence[Mapping[str, Any]],
    view_probabilities: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    adapter_sha256: str,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    """Pure formatter shared by GPU inference and CPU contract tests."""
    outputs = []
    for row in inputs:
        views = view_probabilities.get(str(row["pair_id"])) or {}
        if set(views) != {"ab", "ba"}:
            raise ValueError(f"missing order views for {row['pair_id']}")
        rendered_views = {}
        for view in ("ab", "ba"):
            values = [float(value) for value in views[view]]
            if len(values) != 3:
                raise ValueError(f"invalid view probabilities for {row['pair_id']}/{view}")
            probabilities = dict(zip(LABELS, values))
            rendered_views[view] = {"prediction": _label(probabilities), "probabilities": probabilities}
        mean = {
            label: (rendered_views["ab"]["probabilities"][label]
                    + rendered_views["ba"]["probabilities"][label]) / 2
            for label in LABELS
        }
        output = {
            "schema_version": PAIR_OUTPUT_SCHEMA,
            "pair_id": row["pair_id"],
            "task": row["task"],
            "level": row["level"],
            "protocol_id": row["protocol_id"],
            "input_sha256": pair_input_sha256(row),
            "prediction": _label(mean),
            "probabilities": mean,
            "order_views": rendered_views,
            "order_consistent": rendered_views["ab"]["prediction"] == rendered_views["ba"]["prediction"],
            "adapter_sha256": adapter_sha256,
            "protocol_sha256": protocol_sha256,
        }
        # Validate before the caller writes an immutable output file.  A bad
        # probability vector must not strand an invalid artifact that then
        # prevents a corrected retry at the same explicit path.
        outputs.append(validate_pair_output(output, context=f"pair output {row['pair_id']}"))
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    assert_sk2_host()
    inputs_path = Path(args.inputs).expanduser().resolve()
    outputs_path = Path(args.outputs).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if outputs_path.exists() or report_path.exists():
        raise FileExistsError(outputs_path if outputs_path.exists() else report_path)
    rows = _read_inputs(inputs_path)
    cell = {(row["task"], row["level"], row["protocol_id"]) for row in rows}
    if len(cell) != 1:
        raise ValueError("hierarchy scoring requires exactly one task/level/protocol cell")
    protocol_id = next(iter(cell))[2]
    protocol_text, protocol_sha = _protocol_bundle(Path(args.protocols).resolve(), protocol_id)
    adapter_file = Path(args.adapter).expanduser().resolve() / "adapter_model.safetensors"
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    adapter_sha = sha256_file(adapter_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded, label_ids = _encode(tokenizer, rows, protocol_text, max_length=args.max_length)
    model = _load_model(args.model, args.adapter)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    model.to(device).eval()
    label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
    views: dict[str, dict[str, list[float]]] = defaultdict(dict)
    with torch.inference_mode():
        for start in range(0, len(encoded), args.batch_size):
            selected = encoded[start : start + args.batch_size]
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in collate(selected, int(tokenizer.pad_token_id)).items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    position_ids=batch["position_ids"],
                    logits_to_keep=1,
                ).logits
            probabilities = torch.softmax(
                logits[:, -1, :][:, label_tensor].float(), dim=-1
            ).cpu().tolist()
            for encoded_row, probability in zip(selected, probabilities):
                views[str(encoded_row["example_id"])][str(encoded_row["view"])] = probability
    outputs = assemble_outputs(
        rows, views, adapter_sha256=adapter_sha, protocol_sha256=protocol_sha
    )
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    outputs_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in outputs),
        encoding="utf-8",
    )
    validation = validate_pair_files(inputs_path, outputs_path)
    report = {
        "schema_version": "gemma-hierarchy-pair-scoring-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "adapter": {"path": str(adapter_file), "sha256": adapter_sha},
        "protocols": {"path": str(Path(args.protocols).resolve()), "sha256": sha256_file(Path(args.protocols).resolve())},
        "inputs": {"path": str(inputs_path), "sha256": sha256_file(inputs_path)},
        "outputs": {"path": str(outputs_path), "sha256": sha256_file(outputs_path)},
        "validation": validation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--protocols", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
