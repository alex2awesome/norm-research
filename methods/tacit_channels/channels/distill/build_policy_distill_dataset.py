"""Channel B, step 1 — build the soft-label distillation dataset from the TARGET's own scores.

For each selected cell, the training example is:
  prompt  = readout_template.format(rubric=<name-arm form prompt>, text=<item text truncated>)
            (BYTE-IDENTICAL to the frozen scorer's rendering — verified against
             score_fresh_name_arms.py L540-544)
  p_yes   = the target model's teacher-forced P(YES) for that (form, item) row

Reconstruction-only discipline: p_yes is the TARGET MODEL's name-invoked judgment, never a
human label. Items come only from OPEN partitions (default tacit_breadth_search); the frozen
calibration/eval partitions are refused unless --allow-frozen (which you should not use).

Splits are stable-hash on item text_sha256 (feedback_stable_hash_splits).

Usage:
  python -m methods.tacit_channels.channels.distill.build_policy_distill_dataset \
      --scores-root notebooks/data/two_faces_20260702/family_scores_qwen25 \
      --target-job qwen25_72b_name_target --domain humor \
      --bank notebooks/data/two_faces_20260702/tacit_breadth_arm_bank_v3.json \
      --packet-root notebooks/data/two_faces_20260702/tacit_breadth_item_partitions_v2 \
      --readout-template methods/codability/experiments/<template file> \
      --cells <cell_id>[,<cell_id>...] \
      --out outputs/tacit_channels/distill/humor_<cell>.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from methods.tacit_channels import _apparatus
from methods.tacit_channels.channels.common import (
    load_grid, parse_bank_cells, stable_split, write_jsonl,
)

OPEN_PARTITIONS = ("tacit_breadth_search",)


def build(scores_root: str, target_job: str, domain: str, bank_path: str,
          packet_root: str, readout_template: str, cells: list[str] | None,
          max_text_chars: int, partitions=OPEN_PARTITIONS) -> list[dict]:
    bank = parse_bank_cells(bank_path)
    items = _apparatus.load_domain_items(packet_root, domain, partitions=list(partitions))
    texts = items["texts"] if isinstance(items, dict) and "texts" in items else None
    hashes = items.get("hashes") if isinstance(items, dict) else None
    if texts is None:
        # load_domain_items returns a dict; derive parallel lists from its rows if needed
        rows = items["rows"] if "rows" in items else None
        if rows is None:
            raise ValueError(f"unrecognized load_domain_items payload keys: {list(items)}")
        texts = [r["text"] for r in rows]
        hashes = [r["text_sha256"] for r in rows]
    if hashes is None:
        hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    tgt, meta = load_grid(scores_root, target_job, domain)
    if not tgt:
        raise ValueError(f"no target grids under {scores_root}/{target_job} for {domain}")
    template = Path(readout_template).read_text()

    out = []
    wanted = set(cells) if cells else None
    for (cell_id, arm_id, form), vec in sorted(tgt.items()):
        if arm_id != "name":
            continue  # the name-invoked policy IS the estimand
        if wanted is not None and cell_id not in wanted:
            continue
        if len(vec) != len(texts):
            raise ValueError(
                f"{cell_id}/{form}: score vector has {len(vec)} items but partition(s) "
                f"{partitions} supply {len(texts)} — wrong partition set for these grids")
        cell = bank.get(cell_id)
        form_prompt = None
        if cell:
            for arm in cell["arms"]:
                if arm["id"] == "name":
                    for fm in arm["forms"]:
                        if fm["id"] == form:
                            form_prompt = fm["prompt"]
        if form_prompt is None:
            raise ValueError(f"{cell_id}: name arm form {form!r} not found in bank")
        for text, item_hash, p_yes in zip(texts, hashes, vec):
            out.append({
                "domain": domain, "cell_id": cell_id, "arm_id": "name", "form": form,
                "item_sha256": item_hash,
                "prompt": template.format(rubric=form_prompt, text=text[:max_text_chars]),
                "p_yes": float(p_yes),
                "split": stable_split(item_hash),
                "target_job": target_job,
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-root", required=True)
    ap.add_argument("--target-job", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--packet-root", required=True)
    ap.add_argument("--readout-template", required=True)
    ap.add_argument("--cells", default=None, help="comma-separated cell ids (default: all)")
    ap.add_argument("--max-text-chars", type=int, required=True,
                    help="MUST equal the task's max_text_chars used by the frozen scorer")
    ap.add_argument("--partitions", default=",".join(OPEN_PARTITIONS))
    ap.add_argument("--allow-frozen", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    partitions = tuple(args.partitions.split(","))
    if not args.allow_frozen:
        bad = [p for p in partitions if p not in OPEN_PARTITIONS]
        if bad:
            raise SystemExit(f"refusing frozen/eval partitions {bad} without --allow-frozen "
                             "(training must stay on the open search partition)")
    rows = build(args.scores_root, args.target_job, args.domain, args.bank,
                 args.packet_root, args.readout_template,
                 args.cells.split(",") if args.cells else None,
                 args.max_text_chars, partitions)
    n = write_jsonl(args.out, rows)
    n_train = sum(1 for r in rows if r["split"] == "train")
    manifest = {
        "schema": "tacit_channels_distill_dataset/v1",
        "n_rows": n, "n_train": n_train, "n_eval": n - n_train,
        "readout_template_sha256": _apparatus.sha256_file(args.readout_template),
        "bank": args.bank, "target_job": args.target_job, "domain": args.domain,
        "partitions": list(partitions), "max_text_chars": args.max_text_chars,
    }
    Path(args.out + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {n} rows ({n_train} train) -> {args.out}")


if __name__ == "__main__":
    main()
