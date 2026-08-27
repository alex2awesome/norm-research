#!/usr/bin/env python3
"""Build data.csv + stable-hash grouped 80/10/10 split for the STANDARDIZED dense
arm (Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) on the patents
"claim-fell" cell (V4 remaining-cells task, 2026-08-06).

BACKGROUND / WHY THIS SCRIPT EXISTS DESPITE THE REGISTRY NOTE: notes/2026-07-27
__vat-run-registry.md line 54 and methods/taste_decomposition/patents_verdict_
layer1.py both record "NO honest dense model exists" for this cell -- but that
note documents that a dense run was never BUILT for this exact population, not
that the raw text is unlocatable. This script is the text-location check the V4
task brief asked for: methods/taste_decomposition/patents_verdict_layer1.py's own
provenance comment says the V/A feature matrix (notebooks/data/patents_va_
features.csv, 59,937 rows) was row-aligned (0/59,937 mismatches, verified) against
datasets/patents/processed/option3_claims_gemma_scale.jsonl. That JSONL (sk3 only,
224MB) turns out to carry the RAW TEXT the V features were computed from --
"element" (the claim-element text) and, per candidate reference, "spans" (the
verbatim prior-art passage text) -- so a from-text dense model is buildable after
all. This build re-verifies the alignment itself (belt & suspenders) before
trusting the population.

LEAKAGE GUARD: each reference in option3_claims_gemma_scale.jsonl carries not just
doc_id/spans but also "discloses" (bool) and "vreason" (free text) -- these are
Gemma's OWN disclosure judgments, i.e. exactly the intermediate labels the A_ONLY_
COLS (a_n_disclose etc.) are aggregated from. Feeding "discloses"/"vreason" into
the dense reader's text would hand it the A-bank's output directly (and gate very
close to the "fell" label). The text field below uses ONLY element + doc_id +
spans (the primary source text a verifier would read) -- never discloses/vreason/
is_gold/gold_docs.

Population/y/group verified against methods/taste_decomposition/results/
patents_verdict_layer1.json: n=59937, pos_rate=0.6014315030782321, n_groups=21447,
group_column="app_id". y ("fell") = 1 iff jsonl "label"=="pos" else 0 (verified
identical to the CSV's "fell" column via the alignment check below, and identical
to the check patents_verdict_layer1.py already ran).

Run ON SK3 (CPU only; the 224MB source jsonl is sk3-only):
  $HOME/envs/ai_usage/bin/python3 build_dense_standard_claimfell.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JL = REPO / "datasets" / "patents" / "processed" / "option3_claims_gemma_scale.jsonl"
CSV_PATH = REPO / "notebooks" / "data" / "patents_va_features.csv"
OUT = Path(__file__).resolve().parent / "dense_standard"

EXPECTED_N = 59937
EXPECTED_POS_RATE = 0.6014315030782321
EXPECTED_N_GROUPS = 21447


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Deterministic greedy + hill-climb repair bin-packing of groups into
    train/eval/test buckets targeting 80/10/10 BY ROW COUNT *AND* matched
    per-bucket pos-rate. No seeded shuffle. Verbatim pattern from
    datasets/humor/hashtagwars/build_dense_standard.py.

    THE POS-RATE TERM IS LOAD-BEARING HERE: on this cell, corr(app_id group
    size, mean fell) = +.30 (larger apps skew toward more rejections), so a
    row-count-only version of this bucketer (first attempt, 2026-08-06) dumped
    nearly all large high-reject apps into train via the largest-first greedy
    order, landing train pos-rate .66 vs eval/test .365 -- a real train/eval
    domain shift. With the pos-rate term added, this converges to .6014/
    .6014/.6014 in 2 hill-climb iterations (verified)."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(g)))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2 for b in targets)
        return o

    for g in order:
        best_b, best_o = None, None
        for b in targets:
            filled[b] += sizes[g]; filled_pos[b] += pos[g]
            o = obj()
            if best_o is None or o < best_o:
                best_o, best_b = o, b
            filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
        bmap[g] = best_b
        filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]

    improved = True
    n_iter = 0
    while improved and n_iter < 20:
        improved = False
        n_iter += 1
        for g in order:
            cur = bmap[g]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[b] += sizes[g]; filled_pos[b] += pos[g]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
                filled[cur] += sizes[g]; filled_pos[cur] += pos[g]
            if best_b != cur:
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]
                bmap[g] = best_b
                improved = True
    return bmap


def build_text(r: dict) -> str:
    parts = [f"CLAIM ELEMENT:\n{r['element']}"]
    for i, ref in enumerate(r.get("refs") or []):
        spans = " ".join(ref.get("spans") or [])
        parts.append(f"REFERENCE {i + 1} (patent {ref.get('doc_id', '?')}):\n{spans}")
    return "\n\n".join(parts)


def main():
    print(f"loading {JL} ...", flush=True)
    jrows = [json.loads(l) for l in open(JL) if l.strip()]
    print(f"loaded {len(jrows)} jsonl rows", flush=True)

    print(f"loading {CSV_PATH} for alignment re-check ...", flush=True)
    crows = list(csv.DictReader(open(CSV_PATH)))
    assert len(crows) == len(jrows), f"row-count mismatch: csv={len(crows)} jsonl={len(jrows)}"

    mism = 0
    for c, j in zip(crows, jrows):
        ok = (int(float(c["fell"])) == (1 if j["label"] == "pos" else 0)
              and int(float(c["v_n_refs"])) == int(j["n_refs"])
              and int(float(c["a_n_disclose"])) == int(j["n_disclose"])
              and int(float(c["gold_disclose"])) == int(bool(j["gold_disclose"])))
        mism += not ok
    print(f"alignment re-check: {mism}/{len(crows)} mismatches", flush=True)
    assert mism == 0, "CSV<->JSONL alignment broken -- population invalid, ABORT"

    out_rows = []
    for r in jrows:
        y = 1 if r["label"] == "pos" else 0
        text = build_text(r)
        out_rows.append({
            "text": text, "judgement": y, "group": str(r["app_id"]),
            "claim_num": r["claim_num"], "rejection_type": r.get("rejection_type"),
        })

    n = len(out_rows)
    pos_rate = sum(r["judgement"] for r in out_rows) / n
    n_groups = len(set(r["group"] for r in out_rows))
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}", flush=True)

    assert n == EXPECTED_N, f"n mismatch: {n} != {EXPECTED_N} (Layer-1 population)"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-9, \
        f"pos_rate mismatch: {pos_rate!r} != {EXPECTED_POS_RATE!r} (Layer-1 population)"
    assert n_groups == EXPECTED_N_GROUPS, f"n_groups mismatch: {n_groups} != {EXPECTED_N_GROUPS}"
    print("ASSERTION PASS: rows are exactly the patents_verdict_layer1.json population "
          "(n, pos_rate, n_groups all match to float precision)", flush=True)

    # empty-text sanity (a row with 0 refs would produce element-only text)
    empty_refs = sum(1 for r in jrows if not r.get("refs"))
    zero_len_text = sum(1 for r in out_rows if len(r["text"]) < 20)
    print(f"rows with 0 refs: {empty_refs}; suspiciously short text (<20 chars): {zero_len_text}",
          flush=True)

    y_by_group = {}
    for r in out_rows:
        y_by_group.setdefault(r["group"], []).append(r["judgement"])
    print(f"bucketing {len(y_by_group)} groups ...", flush=True)
    bmap = stable_hash_bucket_map(y_by_group)
    by_split = {"train": [], "eval": [], "test": []}
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "claim_num", "rejection_type"]
    print("writing data.csv ...", flush=True)
    with open(OUT / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    for split in ("train", "eval", "test"):
        with open(OUT / "split" / f"{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(by_split[split])
    print("wrote split files", flush=True)

    manifest = {
        "cell": "patents_verdict (claim-fell)",
        "source": str(JL),
        "population_recipe": "verbatim methods/taste_decomposition/patents_verdict_layer1.py load_data() "
                              "population (same CSV<->JSONL row alignment, re-verified here)",
        "layer1_reference": "methods/taste_decomposition/results/patents_verdict_layer1.json",
        "prior_registry_note": "notes/2026-07-27__vat-run-registry.md line 54 and "
                                "patents_verdict_layer1.py special_rule both say 'NO honest dense model "
                                "exists' for this cell as of 2026-07-27/08-05 -- that reflects no dense "
                                "run ever having been attempted on this population, not that raw text was "
                                "confirmed unlocatable. This build locates and uses the raw text "
                                "(option3_claims_gemma_scale.jsonl element+refs[].spans). FLAG FOR REVIEW "
                                "before letting this T supersede the special_rule in downstream ledgers.",
        "text_construction": "CLAIM ELEMENT text + per-candidate-reference doc_id + verbatim spans text "
                              "ONLY -- excludes refs[].discloses/vreason/is_gold and gold_docs (those are "
                              "Gemma's own disclosure judgments / the A-bank's source material; including "
                              "them would leak the A-bank machinery and the label into the dense reader).",
        "n": n,
        "pos_rate": pos_rate,
        "n_groups": n_groups,
        "group_column": "app_id",
        "y_definition": '1 iff option3 jsonl "label"=="pos" (claim element fell / was rejected under this '
                         "rejection_type given these references), else 0 -- identical to CSV 'fell' column",
        "alignment_recheck_mismatches": mism,
        "rows_with_zero_refs": empty_refs,
        "rows_with_short_text_lt20chars": zero_len_text,
        "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: (sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1))
                             for s in by_split},
        "split_fractions": {s: len(by_split[s]) / n for s in by_split},
        "assertion_rows_subseteq_layer1_population": (
            f"n={n} == {EXPECTED_N}, pos_rate matches to float precision, "
            f"n_groups={n_groups} == {EXPECTED_N_GROUPS} -- PASS"
        ),
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, select-on-eval (dense-standard, no deviation)",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("prior_registry_note", "text_construction")},
                      indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
