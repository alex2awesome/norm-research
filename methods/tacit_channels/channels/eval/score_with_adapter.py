"""Adapter-aware scoring of (cells x arms x forms x items) — npz-compatible with the tally.

Scores an executor (base model, optionally + LoRA adapter) on arm-bank cells through the
teacher-forced readout, emitting grid npz files with the same keys the frozen scorer writes
(scores matrix + json meta rows) so channels/eval/tally_exchange_rate.py and the existing
interim tallies consume them unchanged. Adapter provenance joins the npz scalars.

ACCEPTANCE TEST (--acceptance-test, MANDATORY before any reportable adapter run):
  score a slice with NO adapter through this path and compare per-row Spearman against the
  frozen scorer's npz for the same rows — every row must exceed --acceptance-rho (default
  .999). This is the gate that catches vLLM-LoRA API drift, chat-template drift, and any
  silent fork divergence, all at once.

GPU script; 1 GPU; offline batch (feedback_metric_scoring_offline_batch_vllm).

Usage (score one rung with an adapter):
  python -m methods.tacit_channels.channels.eval.score_with_adapter \
      --model Qwen/Qwen2.5-7B-Instruct --lora-adapter outputs/.../humor_x_n32 \
      --bank <bank.json> --packet-root <partitions dir> --domain humor \
      --readout-template <template> --max-text-chars <N> \
      --yes-id 14004 --no-id 8996 \
      --out-dir outputs/tacit_channels/family_scores/qwen25_7b_lora_n32 \
      --intervention-tag lora_n32

Acceptance test:
  ... --acceptance-test --reference-npz <frozen grid npz> [--limit-cells 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import parse_bank_cells, spearman
from methods.tacit_channels.channels.eval.teacher_forced_lora import (
    check_upstream_drift, score_declared_binary_lora, upstream_source_sha256,
)


def build_backend(model: str, lora_adapter: str | None, tp_size: int, max_model_len: int,
                  gpu_mem_util: float, fake: bool):
    from methods.metric_implementer import config as cfgmod
    from methods.metric_implementer.vllm_backend import make_judge_backend
    cfg = cfgmod.ImplementerConfig()
    cfg.vllm_fake = fake
    cfg.vllm_tp_size = tp_size
    cfg.vllm_max_model_len = max_model_len
    cfg.vllm_gpu_mem_util = gpu_mem_util
    if lora_adapter:
        cfg.vllm_lora_path = str(lora_adapter)  # picked up by OfflineVLLM._engine/_maybe_lora
    return make_judge_backend(model, cfg, temperature=None)


def score_grid(backend, cells: dict, items: dict, template: str, max_text_chars: int,
               label_token_ids: dict, domain: str, repetition: int = 0,
               limit_cells: int | None = None, arm_filter=None):
    texts = items["texts"]
    scores, meta = [], []
    cell_ids = sorted(cells)
    if limit_cells:
        cell_ids = cell_ids[:limit_cells]
    for cell_id in cell_ids:
        cell = cells[cell_id]
        for arm in cell["arms"]:
            if arm_filter and arm["id"] not in arm_filter:
                continue
            for form in arm["forms"]:
                prompts = [template.format(rubric=form["prompt"], text=t[:max_text_chars])
                           for t in texts]
                # identical row-seed law to the frozen scorer (score_fresh_name_arms L545-548)
                row_seed = repetition * 1_000_003 + len(meta) * 1009 + 20260713
                row = score_declared_binary_lora(
                    backend, prompts, pos="YES", neg="NO",
                    expected_token_ids=label_token_ids, seed=row_seed)
                scores.append(np.asarray(row, float))
                meta.append({"cell_id": cell_id, "domain": domain, "task": cell.get("task"),
                             "level": cell.get("level"), "bucket": cell.get("bucket"),
                             "metric_id": cell.get("metric_id"),
                             "node_id": cell.get("node_id"), "gi": cell.get("gi"),
                             "construct": cell.get("construct"), "arm_id": arm["id"],
                             "channel": arm.get("channel"),
                             "provenance": arm.get("provenance"),
                             "control_for": arm.get("control_for"),
                             "semantic_content_word_count":
                                 arm.get("semantic_content_word_count"),
                             "added_content_word_count": arm.get("added_content_word_count"),
                             "n_address_units": arm.get("n_address_units"),
                             "form": form["id"], "prompt_sha256": form.get("prompt_sha256")})
    return np.vstack(scores), meta


def run_acceptance_test(matrix, meta, reference_npz: str, floor: float) -> None:
    ref = np.load(reference_npz, allow_pickle=True)
    ref_scores = np.asarray(ref["scores"])
    ref_index = {}
    for i, s in enumerate(ref["meta"]):
        m = json.loads(s)
        ref_index[(m["cell_id"], m["arm_id"], m["form"])] = i
    rhos, missing = [], 0
    for i, m in enumerate(meta):
        j = ref_index.get((m["cell_id"], m["arm_id"], m["form"]))
        if j is None:
            missing += 1
            continue
        rhos.append(spearman(matrix[i], ref_scores[j]))
    rhos = np.asarray(rhos, float)
    print(f"acceptance: {len(rhos)} rows compared ({missing} not in reference), "
          f"min rho {np.nanmin(rhos):.6f}, median {np.nanmedian(rhos):.6f}")
    if len(rhos) == 0 or np.nanmin(rhos) < floor:
        raise SystemExit(f"ACCEPTANCE FAILED: min per-row rho {np.nanmin(rhos):.6f} < {floor} "
                         "— do NOT run adapter scoring until this passes "
                         "(vLLM-LoRA drift / template drift / fork divergence).")
    print("ACCEPTANCE PASSED — adapter path reproduces the frozen readout.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--packet-root", required=True)
    ap.add_argument("--partitions", default="tacit_breadth_search")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--readout-template", required=True)
    ap.add_argument("--max-text-chars", type=int, required=True)
    ap.add_argument("--yes-id", type=int, required=True)
    ap.add_argument("--no-id", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--intervention-tag", default="base")
    ap.add_argument("--repetition", type=int, default=0)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--arms", default=None, help="comma list; default all arms")
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--upstream-sha", default=None,
                    help="pinned sha256 of the frozen score_declared_binary source")
    ap.add_argument("--acceptance-test", action="store_true")
    ap.add_argument("--reference-npz", default=None)
    ap.add_argument("--acceptance-rho", type=float, default=0.999)
    args = ap.parse_args()

    if args.acceptance_test and args.lora_adapter:
        raise SystemExit("acceptance test runs WITHOUT an adapter (zero-adapter equivalence)")
    if args.acceptance_test and not args.reference_npz:
        raise SystemExit("--acceptance-test requires --reference-npz")

    if args.upstream_sha:
        check_upstream_drift(args.upstream_sha)
    else:
        print(f"upstream score_declared_binary sha256: {upstream_source_sha256()} "
              "(pin with --upstream-sha for reportable runs)")

    cells = parse_bank_cells(args.bank)
    cells = {cid: c for cid, c in cells.items() if c.get("domain") == args.domain}
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = {cid: c for cid, c in cells.items() if cid in wanted}
    items = _apparatus.load_domain_items(
        args.packet_root, args.domain, partitions=args.partitions.split(","))
    template = Path(args.readout_template).read_text()
    label_token_ids = {"YES": args.yes_id, "NO": args.no_id}

    backend = build_backend(args.model, args.lora_adapter, args.tp_size,
                            args.max_model_len, args.gpu_mem_util, args.fake)
    matrix, meta = score_grid(
        backend, cells, items, template, args.max_text_chars, label_token_ids,
        args.domain, repetition=args.repetition, limit_cells=args.limit_cells,
        arm_filter=set(args.arms.split(",")) if args.arms else None)

    if args.acceptance_test:
        run_acceptance_test(matrix, meta, args.reference_npz, args.acceptance_rho)
        return

    # fail-closed: refuse to persist non-finite scores (v1b post-mortem — NaN adapters
    # produced NaN grids that a "successful" chain silently wrote to disk)
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).sum())
        raise SystemExit(f"REFUSING to write grid: {bad} non-finite scores — adapter or "
                         "scoring path is broken (check training loss trajectory)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(
        f"{args.model}::{args.lora_adapter}::{args.intervention_tag}".encode()).hexdigest()[:16]
    out = out_dir / f"grid_{args.domain}_channels_{tag}_rep{args.repetition}.npz"
    np.savez_compressed(
        out, scores=matrix, meta=np.array([json.dumps(m) for m in meta], dtype=object),
        model=args.model, lora_adapter=str(args.lora_adapter),
        intervention_tag=args.intervention_tag,
        label_token_ids=json.dumps(label_token_ids),
        upstream_declared_binary_sha256=upstream_source_sha256(),
        readout="teacher_forced_declared_labels(lora-fork)")
    print(f"wrote {matrix.shape} -> {out}")


if __name__ == "__main__":
    main()
