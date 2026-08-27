#!/usr/bin/env python3
"""Run the audited Nemotron trainer with exact effective-batch preservation.

The original trainer uses mean loss per microbatch and divides every loss by
``gradient_accumulation_steps``.  A larger physical batch is therefore exactly
equivalent for complete effective batches, but a final short batch needs an
additional ``actual/nominal`` factor.  This append-only wrapper supplies that
factor, writes atomic live progress, and leaves every scientific input and the
underlying trainer unchanged.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import train_nemotron_lora as base
from .common import sha256_file


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = base.parse_args()
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    if effective_batch != 32:
        raise ValueError(
            "fast wrapper is frozen for effective batch 32; "
            f"observed {args.batch_size}x{args.gradient_accumulation_steps}"
        )
    if args.batch_size != 32 or args.gradient_accumulation_steps != 1:
        raise ValueError("fast wrapper requires physical batch 32 and accumulation 1")

    from sentence_transformers import losses

    output = Path(args.output_root).resolve() / args.task
    progress_path = output / "FAST_EFFECTIVE_BATCH_PROGRESS.json"
    original_forward = losses.TripletLoss.forward
    process_started = time.monotonic()
    state: dict[str, Any] = {
        "batches": 0,
        "examples": 0,
        "triplets": None,
        "batches_per_epoch": None,
        "partial_batch_observed": False,
        "training_started": None,
    }

    def wrapped_forward(self: Any, sentence_features: Any, labels: Any) -> Any:
        loss = original_forward(self, sentence_features, labels)
        actual = int(sentence_features[0]["input_ids"].shape[0])
        if actual <= 0 or actual > args.batch_size:
            raise RuntimeError(f"invalid physical batch observed: {actual}")
        partial_scale = actual / args.batch_size
        if partial_scale < 1.0:
            state["partial_batch_observed"] = True
            loss = loss * partial_scale
        state["batches"] += 1
        state["examples"] += actual
        if state["training_started"] is None:
            state["training_started"] = time.monotonic()
        if state["triplets"] is None:
            triplet_path = output / "training_triplets.jsonl"
            with triplet_path.open("rb") as handle:
                state["triplets"] = sum(1 for line in handle if line.strip())
            state["batches_per_epoch"] = math.ceil(
                int(state["triplets"]) / args.batch_size
            )
        batches_per_epoch = int(state["batches_per_epoch"])
        if state["batches"] == 1 or state["batches"] % 50 == 0:
            elapsed = time.monotonic() - float(state["training_started"])
            startup = float(state["training_started"]) - process_started
            completed = int(state["batches"])
            total = batches_per_epoch * args.epochs
            rate = completed / elapsed if elapsed > 0 else 0.0
            _atomic_json(
                progress_path,
                {
                    "schema_version": "silver-match-v3-nemotron-fast-progress-v1",
                    "status": "TRAINING",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "task": args.task,
                    "physical_batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "effective_batch_size": effective_batch,
                    "partial_batch_loss_scale": "actual_physical_batch/nominal_physical_batch",
                    "triplets": state["triplets"],
                    "epochs": args.epochs,
                    "batches_per_epoch": batches_per_epoch,
                    "completed_batches": completed,
                    "completed_examples_including_repeated_epochs": state["examples"],
                    "current_epoch": min(args.epochs, (completed - 1) // batches_per_epoch + 1),
                    "batch_in_epoch": (completed - 1) % batches_per_epoch + 1,
                    "total_batches": total,
                    "elapsed_seconds": elapsed,
                    "startup_and_preprocessing_seconds": startup,
                    "batches_per_second": rate,
                    "eta_seconds": (total - completed) / rate if rate > 0 else None,
                    "partial_batch_observed": state["partial_batch_observed"],
                    "wrapper_sha256": sha256_file(Path(__file__).resolve()),
                    "underlying_trainer_sha256": sha256_file(Path(base.__file__).resolve()),
                },
            )
        return loss

    losses.TripletLoss.forward = wrapped_forward
    try:
        report = base.train(args)
    finally:
        losses.TripletLoss.forward = original_forward

    report_path = output / "training_report.json"
    _atomic_json(
        output / "FAST_EFFECTIVE_BATCH_PROVENANCE.json",
        {
            "schema_version": "silver-match-v3-nemotron-fast-effective-batch-v1",
            "status": "COMPLETE",
            "task": args.task,
            "wrapper": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "underlying_trainer": {
                "path": str(Path(base.__file__).resolve()),
                "sha256": sha256_file(Path(base.__file__).resolve()),
            },
            "training_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
                "status": report["status"],
            },
            "scientific_invariants": {
                "effective_batch_size": effective_batch,
                "physical_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "final_short_batch_scaled_to_original_accumulation_semantics": True,
                "seed": args.seed,
                "split_seed": args.split_seed,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "margin": args.margin,
                "negative_pool": args.hard_negative_pool,
                "negatives_per_positive": args.negatives_per_positive,
                "attention": args.attention,
                "bf16_model_dtype_from_underlying_trainer": True,
            },
        },
    )


if __name__ == "__main__":
    main()
