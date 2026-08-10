#!/usr/bin/env python3
"""Build the RAW-MATCHED displacement-control arm for a V3-grid cell.

WHY THIS ARM EXISTS
-------------------
On the four long-document cells (nc_agree, nc_outcome, nc_responded,
press_verdict) the k=20 criteria block is NOT free: at max_len 1024 it costs
219-260 tokens, so the fraction of rows whose document is cut rises from ~2%
(raw text) to ~.29-.31 (with block) -- and to .45 on press_verdict.  The block
is PREPENDed, so it always survives; what gets deleted is the document tail
that the raw/T arm kept.  A "V3 <= T" or "V3 <= bank" result on those cells is
therefore confounded: the two arms did not read the same amount of document,
and the confound runs in the direction that makes V3 look worse.

This script builds the resolving control: the SAME rows, the SAME splits, the
SAME recipe, no criterion block, and the document truncated to the budget the
V3 arm actually left for it,

    max_length = 1024 - manifest.truncation.block_tokens

so that (V3 - raw_matched) isolates the criterion block's contribution at a
matched document budget, and (raw_matched - T_original) prices the lost text on
its own.

WHAT IS ASSERTED (build aborts on any failure)
----------------------------------------------
  * the source cell is PREPEND (an APPEND cell has no displacement story)
  * eval/test row sets are IDENTICAL to the original dense split
  * per split, elementwise: judgement, group and did equal the V3 arm's
  * per row, the V3 text ENDS WITH "\\n" + the original dense text, so the raw
    text is recovered byte-exactly (checked two independent ways: against the
    original dense split CSV, and by stripping the block's k+1 leading lines)
  * the raw-matched arm's truncation RATE at its reduced budget reproduces the
    V3 arm's `with_block` rate at 1024 (that equality is the definition of
    "matched budget"; see `budget_equivalence` in the manifest)

Usage:
    python3 build_v3_rawmatched.py --slug nc_agree [--slug nc_outcome ...]
    python3 build_v3_rawmatched.py --all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
REPO = TD.parent.parent
DATA = FUS / "dense_data"

MAX_LEN = 1024
SPLITS = ("train", "eval", "test")
CELLS = ("nc_agree", "nc_outcome", "nc_responded", "press_verdict")

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    return _TOK


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trunc_stats(tok, texts, max_len):
    """Same estimator as build_v3_cell.truncation_stats, at an arbitrary budget."""
    lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
    return {"n": int(len(lens)),
            "rate_over_max_len": float((lens > max_len).mean()),
            "tok_median": int(np.median(lens)),
            "tok_p95": int(np.percentile(lens, 95)),
            "tok_max": int(lens.max())}


def orig_split_dir(man: dict) -> Path:
    """The cell's original dense-standard split dir, re-rooted onto THIS repo.

    The manifest records the path as it existed on the machine that built the
    cell (a mac checkout); everything after `norm-research/` is stable.
    """
    raw = man["dense_standard_split_dir"]
    tail = raw.split("norm-research/", 1)[1]
    p = REPO / tail
    if not (p / "split").is_dir():
        raise SystemExit(f"original dense split dir not found: {p}/split "
                         f"(manifest recorded {raw})")
    return p


def build(slug: str, force: bool = False) -> dict:
    src = DATA / f"v3grid_{slug}"
    dst = DATA / f"v3grid_{slug}_rawmatched"
    man = json.loads((src / "manifest.json").read_text())

    # ---- preconditions --------------------------------------------------
    if man.get("block_placement") != "PREPEND":
        raise SystemExit(f"{slug}: block_placement is {man.get('block_placement')}, "
                         f"not PREPEND -- no displacement story to control for")
    if man.get("eval_test_row_sets", {}).get("status") != "IDENTICAL":
        raise SystemExit(f"{slug}: eval/test row sets are "
                         f"{man.get('eval_test_row_sets', {}).get('status')}, "
                         f"not IDENTICAL -- refusing to build a matched control")
    block_tokens = int(man["truncation"]["block_tokens"])
    if man["truncation"]["max_len"] != MAX_LEN:
        raise SystemExit(f"{slug}: source arm max_len is "
                         f"{man['truncation']['max_len']}, expected {MAX_LEN}")
    budget = MAX_LEN - block_tokens
    k = int(man["k"])
    odir = orig_split_dir(man)

    if dst.exists() and not force:
        print(f"[{slug}] {dst} exists -- pass --force to rebuild")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "split").mkdir(exist_ok=True)

    tok = tokenizer()
    frames, raw_all, checks = {}, [], {}

    for s in SPLITS:
        v3 = pd.read_csv(src / "split" / f"{s}.csv")
        og = pd.read_csv(odir / "split" / f"{s}.csv")
        assert len(v3) == len(og) == man["n"][s], (
            f"{slug}/{s}: n mismatch v3={len(v3)} orig={len(og)} "
            f"manifest={man['n'][s]}")

        v3_text = [str(t) for t in v3["text"]]
        og_text = [str(t) for t in og["text"]]

        # --- elementwise label/group identity with the V3 arm ------------
        assert (v3["judgement"].astype(int).to_numpy()
                == og["judgement"].astype(int).to_numpy()).all(), \
            f"{slug}/{s}: judgement differs between the V3 split and the original dense split"
        assert [str(g) for g in v3["group"]] == [str(g) for g in og["group"]], \
            f"{slug}/{s}: group differs between the V3 split and the original dense split"

        # --- raw text recovered TWO independent ways ---------------------
        n_suffix, n_lines = 0, 0
        for a, t in zip(v3_text, og_text):
            assert a.endswith("\n" + t), \
                f"{slug}/{s}: V3 text does not end with '\\n' + the original text"
            n_suffix += 1
            # independent derivation: drop the block's k+1 leading lines
            assert "\n".join(a.split("\n")[k + 1:]) == t, \
                f"{slug}/{s}: line-strip derivation disagrees with the suffix derivation"
            n_lines += 1

        out = pd.DataFrame({
            "text": og_text,                                   # RAW text, no block
            "judgement": v3["judgement"].astype(int).to_numpy(),
            "group": [str(g) for g in v3["group"]],
            "did": [str(d) for d in v3["did"]],                # ids carried from V3
        })
        # order is positional and untouched, so did/group/judgement match by row
        assert list(out["did"]) == [str(d) for d in v3["did"]]
        assert list(out["judgement"]) == list(v3["judgement"].astype(int))
        out.to_csv(dst / "split" / f"{s}.csv", index=False)
        frames[s] = out
        raw_all.extend(og_text)
        checks[s] = {"n": int(len(out)),
                     "n_rows_suffix_verified": n_suffix,
                     "n_rows_linestrip_verified": n_lines,
                     "v3_split_sha256": sha256(src / "split" / f"{s}.csv"),
                     "orig_split_sha256": sha256(odir / "split" / f"{s}.csv"),
                     "rawmatched_split_sha256": sha256(dst / "split" / f"{s}.csv")}

    pd.concat([frames[s] for s in SPLITS], ignore_index=True).to_csv(
        dst / "data.csv", index=False)

    # ---- budget equivalence: the definition of "matched" ----------------
    # V3 overruns 1024 iff  1 + n_block + n_doc > 1024  iff  n_doc > 1023 - n_block.
    # raw-matched overruns `budget` iff 1 + n_doc > 1024 - n_block, the SAME
    # condition.  So the two truncation rates must agree up to the per-row
    # wobble in the block's own length.
    rm_trunc = trunc_stats(tok, raw_all, budget)
    v3_rate = float(man["truncation"]["with_block"]["rate_over_max_len"])
    delta = abs(rm_trunc["rate_over_max_len"] - v3_rate)
    assert delta < 0.01, (
        f"{slug}: raw-matched truncation rate {rm_trunc['rate_over_max_len']:.4f} "
        f"at budget {budget} does not reproduce the V3 arm's with-block rate "
        f"{v3_rate:.4f} (delta {delta:.4f}) -- the budgets are NOT matched")

    n_tot = sum(len(frames[s]) for s in SPLITS)
    out_man = {
        "cell": slug,
        "arm": f"v3grid_{slug}_rawmatched",
        "role": ("RAW-TEXT-ONLY DISPLACEMENT CONTROL for the V3 arm: identical "
                 "rows, splits and recipe, no criterion block, document "
                 "truncated to the budget the V3 block left it"),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_arm": f"v3grid_{slug}",
        "source_manifest_sha256": sha256(src / "manifest.json"),

        # ---- the budget and how it was derived --------------------------
        "max_length": budget,
        "budget_derivation": {
            "formula": "max_length = 1024 - manifest.truncation.block_tokens",
            "source_max_len": MAX_LEN,
            "block_tokens": block_tokens,
            "block_tokens_source": (
                f"v3grid_{slug}/manifest.json truncation.block_tokens -- "
                "build_v3_cell.py measures it as len(tok(block, "
                "add_special_tokens=False)) on the FIRST TRAIN ROW's rendered "
                "block, so it is a per-cell scalar, not a per-row length"),
            "result": budget,
        },
        "budget_equivalence": {
            "claim": ("a V3 row overruns 1024 iff 1+n_block+n_doc > 1024, i.e. "
                      "n_doc > 1023-n_block; a raw-matched row overruns its "
                      "budget iff 1+n_doc > 1024-n_block, the SAME condition. "
                      "Equal truncation rates therefore certify equal document "
                      "budgets."),
            "v3_rate_over_max_len_with_block_at_1024": v3_rate,
            "rawmatched_rate_over_max_len_at_budget": rm_trunc["rate_over_max_len"],
            "abs_delta": round(delta, 6),
            "tolerance": 0.01,
            "passed": True,
        },
        "truncation": {
            "max_len": budget,
            "raw_text_at_budget": rm_trunc,
            "reference_v3_raw_text_only_at_1024": man["truncation"]["raw_text_only"],
            "reference_v3_with_block_at_1024": man["truncation"]["with_block"],
        },

        # ---- everything held fixed --------------------------------------
        "selection_split": man["selection_split"],
        "trainer_entry": man["trainer_entry"],
        "trainer_entry_reason": man["trainer_entry_reason"],
        "group_column": man["group_column"],
        "dense_standard_split_dir": man["dense_standard_split_dir"],
        "orig_split_sha256": man["orig_split_sha256"],
        "n": {s: int(len(frames[s])) for s in SPLITS},
        "n_orig": man["n_orig"],
        "split_fractions": {s: len(frames[s]) / n_tot for s in SPLITS},
        "eval_test_row_sets": {
            "identical_to_original": True,
            "status": "IDENTICAL",
            "identical_to_v3_arm": True,
        },
        "recipe": (
            "IDENTICAL to the V3 arm (frozen 8B LoRA r16/a32, lr 5e-5, bs16, "
            "eval_bs 32, grad-accum 1, 2 epochs, gradient checkpointing, seed 42, "
            "the cell's own selection_split, NO --class_weight_auto) EXCEPT: "
            f"no criterion block, and max_length {budget} instead of {MAX_LEN}"),
        "only_difference_from_v3": [
            "the k=20 'name: score' block is absent (text is the raw document)",
            f"--max_length {budget} instead of {MAX_LEN}",
        ],
        "class_weight_auto": False,
        "class_weight_note": (
            "deliberately NOT used: the V3 and T chains for these cells used no "
            "class weighting, and adding it would confound the block's "
            "contribution with a loss change"),

        # ---- byte-level assertions --------------------------------------
        "byte_assertions": [
            "per split, n equals the V3 arm's and the original dense split's",
            "judgement equal elementwise to the V3 arm's split rows",
            "group equal elementwise to the V3 arm's split rows",
            "did carried verbatim from the V3 arm's split rows, same order",
            "every V3 text ENDS WITH '\\n' + this arm's text (suffix derivation)",
            "dropping the block's k+1 leading lines from the V3 text reproduces "
            "this arm's text (independent line-strip derivation)",
            "raw-matched truncation rate at the reduced budget reproduces the V3 "
            "arm's with-block rate at 1024 to <0.01",
        ],
        "per_split_checks": checks,
        "leakage_rules": [
            "no criterion scores are rendered at all in this arm",
            "y NEVER appears in a prompt",
            "splits are the V3 arm's split CSVs with the block removed, row for row",
        ],
    }
    (dst / "manifest.json").write_text(json.dumps(out_man, indent=2))

    print(f"[{slug}] built {dst}")
    print(f"[{slug}]   block_tokens {block_tokens} -> max_length {budget} "
          f"(selection_split={man['selection_split']})")
    print(f"[{slug}]   n train/eval/test = "
          f"{out_man['n']['train']}/{out_man['n']['eval']}/{out_man['n']['test']}")
    print(f"[{slug}]   truncation: V3 with-block @1024 {v3_rate:.4f} vs "
          f"raw-matched @{budget} {rm_trunc['rate_over_max_len']:.4f} "
          f"(delta {delta:.5f}) OK")
    return out_man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    slugs = list(CELLS) if a.all else a.slug
    if not slugs:
        raise SystemExit("give --slug or --all")
    for s in slugs:
        build(s, force=a.force)


if __name__ == "__main__":
    main()
