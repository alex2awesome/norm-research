"""W1 variant-pass runner — score pass-planner rows through the frozen LoRA-fork readout.

One invocation = ONE engine load (model, optional adapter): the pass planner (passes.py)
builds every single-stage variant row (tf / exclusion / negated / composed [+ holistic]) and
each row is scored over all domain items via the identical teacher-forced path the frozen
scorer and score_with_adapter use. Output npz is battery-compatible (ArtifactContext.grid /
load_grid consume it unchanged): variant is encoded into arm_id ("name" for tf, else
"name_<variant>", "holistic"), so keys never collide with frozen name rows.

ACCEPTANCE (mandatory before any reportable run, prereg discipline "acceptance-test
freshness for any new scoring path"): --acceptance-test scores tf rows zero-adapter and
compares per-row Spearman >= --acceptance-rho against the frozen executor grid — tf content
IS the bank's name prompt, so equality is exact-path equivalence.

GPU script; 1 GPU; offline batch. Smoke discipline: --limit-cells 5 before any full slice.

Usage (executor pass, sk2):
  python -m methods.tacit_channels.battery.run_variant_pass \
      --model <hf-path> [--lora-adapter <dir>] \
      --bank <bank.json> --packet-root <partitions> --domain humor \
      --readout-template <tpl> --max-text-chars 4000 --yes-id 14004 --no-id 8996 \
      --variants tf,exclusion,negated,composed \
      --composed-pairs outputs/tacit_channels/battery/w1_composed_pairs.json \
      --out-dir <dir> --intervention-tag <tag> --upstream-sha <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import parse_bank_cells, spearman
from methods.tacit_channels.battery.passes import (
    build_holistic_row, build_single_stage_rows, plan_summary,
)

ARM_ID = {"tf": "name", "exclusion": "name_exclusion", "negated": "name_negated",
          "composed": "name_composed", "holistic": "holistic"}


def load_composed_pairs(path: str | None) -> tuple:
    if not path:
        return ()
    d = json.load(open(path))
    return tuple(tuple(p) for p in d.get("pairs_a_x_a", []) + d.get("pairs_non_a", []))


def score_rows(backend, rows: list, texts: list, template: str, max_text_chars: int,
               label_token_ids: dict, domain: str, repetition: int = 0,
               score_fn=None) -> tuple:
    """Score every planner row over all items. score_fn injectable for tests."""
    if score_fn is None:
        from methods.tacit_channels.channels.eval.teacher_forced_lora import (
            score_declared_binary_lora)
        score_fn = score_declared_binary_lora
    scores, meta = [], []
    for row in rows:
        prompts = [template.format(rubric=row["content"], text=t[:max_text_chars])
                   for t in texts]
        # identical row-seed law to the frozen scorer (score_fresh_name_arms L545-548)
        row_seed = repetition * 1_000_003 + len(meta) * 1009 + 20260713
        vec = score_fn(backend, prompts, pos="YES", neg="NO",
                       expected_token_ids=label_token_ids, seed=row_seed)
        scores.append(np.asarray(vec, float))
        meta.append({"cell_id": row["cell_id"], "domain": domain,
                     "arm_id": ARM_ID[row["variant"]], "variant": row["variant"],
                     "form": row["form"], "pair": row.get("pair"),
                     "content_sha256": hashlib.sha256(row["content"].encode()).hexdigest()})
    return np.vstack(scores), meta


def run_acceptance(matrix, meta, reference_npz: str, floor: float) -> None:
    ref = np.load(reference_npz, allow_pickle=True)
    ref_scores = np.asarray(ref["scores"])
    ref_index = {}
    for i, s in enumerate(ref["meta"]):
        m = json.loads(s)
        ref_index[(m["cell_id"], m["arm_id"], m["form"])] = i
    rhos, missing = [], 0
    for i, m in enumerate(meta):
        if m["variant"] != "tf":
            continue
        j = ref_index.get((m["cell_id"], "name", m["form"]))
        if j is None:
            missing += 1
            continue
        rhos.append(spearman(matrix[i], ref_scores[j]))
    rhos = np.asarray(rhos, float)
    if len(rhos) == 0:
        raise SystemExit("ACCEPTANCE FAILED: no tf rows matched the reference grid")
    n_nan = int(np.isnan(rhos).sum())
    if n_nan:  # a NaN rho (zero-variance row) cannot demonstrate rho >= floor
        raise SystemExit(f"ACCEPTANCE FAILED: {n_nan} zero-variance (NaN-rho) tf rows "
                         "— degenerate scoring path")
    print(f"acceptance: {len(rhos)} tf rows ({missing} unmatched), "
          f"min rho {np.nanmin(rhos):.6f}, median {np.nanmedian(rhos):.6f}")
    if np.nanmin(rhos) < floor:
        raise SystemExit(f"ACCEPTANCE FAILED: min per-row rho {np.nanmin(rhos):.6f} < {floor}")
    print("ACCEPTANCE PASSED — variant-pass runner reproduces the frozen readout.")


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
    ap.add_argument("--variants", default="tf,exclusion,negated,composed")
    ap.add_argument("--forms", default="canonical,question,boilerplate")
    ap.add_argument("--composed-pairs", default=None)
    ap.add_argument("--holistic", action="store_true")
    ap.add_argument("--cells", default=None, help="comma list; default all domain cells")
    ap.add_argument("--limit-cells", type=int, default=None)
    ap.add_argument("--out-dir", default=None, help="required unless --acceptance-test")
    ap.add_argument("--intervention-tag", default="base")
    ap.add_argument("--repetition", type=int, default=0)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--upstream-sha", default=None)
    ap.add_argument("--acceptance-test", action="store_true")
    ap.add_argument("--reference-npz", default=None)
    ap.add_argument("--acceptance-rho", type=float, default=0.999)
    args = ap.parse_args()

    if args.acceptance_test and args.lora_adapter:
        raise SystemExit("acceptance test runs WITHOUT an adapter")
    if args.acceptance_test and not args.reference_npz:
        raise SystemExit("--acceptance-test requires --reference-npz")
    if not args.acceptance_test and not args.out_dir:
        raise SystemExit("--out-dir is required for scoring runs")
    from methods.tacit_channels.channels.eval.teacher_forced_lora import (
        check_upstream_drift, upstream_source_sha256)
    if args.upstream_sha:
        check_upstream_drift(args.upstream_sha)

    cells = parse_bank_cells(args.bank)
    cells = {cid: c for cid, c in cells.items() if c.get("domain") == args.domain}
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = {cid: c for cid, c in cells.items() if cid in wanted}
    if args.limit_cells:
        cells = {cid: cells[cid] for cid in sorted(cells)[:args.limit_cells]}

    variants = tuple(v for v in args.variants.split(",") if v)
    pairs = load_composed_pairs(args.composed_pairs) if "composed" in variants else ()
    if "composed" in variants:
        pairs = tuple(p for p in pairs if p[0] in cells and p[1] in cells)
    rows = build_single_stage_rows(cells, variants, composed_pairs=pairs,
                                   forms=tuple(args.forms.split(",")))
    if args.holistic:
        rows.append(build_holistic_row(args.domain))
    print(f"pass plan: {plan_summary(rows)} over {len(cells)} cells")

    items = _apparatus.load_domain_items(
        args.packet_root, args.domain, partitions=args.partitions.split(","))
    template = Path(args.readout_template).read_text()
    label_token_ids = {"YES": args.yes_id, "NO": args.no_id}

    from methods.tacit_channels.channels.eval.score_with_adapter import build_backend
    backend = build_backend(args.model, args.lora_adapter, args.tp_size,
                            args.max_model_len, args.gpu_mem_util, args.fake)
    matrix, meta = score_rows(backend, rows, items["texts"], template,
                              args.max_text_chars, label_token_ids, args.domain,
                              repetition=args.repetition)

    if args.acceptance_test:
        run_acceptance(matrix, meta, args.reference_npz, args.acceptance_rho)
        return

    if not np.isfinite(matrix).all():  # fail-closed, v1b post-mortem discipline
        bad = int((~np.isfinite(matrix)).sum())
        raise SystemExit(f"REFUSING to write grid: {bad} non-finite scores")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(
        f"{args.model}::{args.lora_adapter}::{args.intervention_tag}::w1"
        .encode()).hexdigest()[:16]
    out = out_dir / f"grid_{args.domain}_w1variants_{tag}_rep{args.repetition}.npz"
    np.savez_compressed(
        out, scores=matrix, meta=np.array([json.dumps(m) for m in meta], dtype=object),
        model=args.model, lora_adapter=str(args.lora_adapter),
        intervention_tag=args.intervention_tag,
        variants=",".join(variants) + (",holistic" if args.holistic else ""),
        composed_pairs_file=str(args.composed_pairs),
        label_token_ids=json.dumps(label_token_ids),
        upstream_declared_binary_sha256=upstream_source_sha256(),
        readout="teacher_forced_declared_labels(lora-fork,w1-variants)")
    print(f"wrote {matrix.shape} -> {out}")


if __name__ == "__main__":
    main()
