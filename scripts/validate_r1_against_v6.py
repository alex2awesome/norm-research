"""Validate an R1 family build against v6 judge pair labels.

Inputs:
  outputs/analyses/structural_metrics/validation/<task>_v6_verdicts.jsonl
    (key_a, key_b, score 0/1/2)
  outputs/analyses/structural_metrics/clusters_<task>.json
    (member key -> cluster_id mapping)
  outputs/analyses/structural_metrics/<r1_dir>/r1_families_<task>.json
    (R1 families with cluster_ids per family)

Metric:
  For every v6-labeled pair where BOTH keys map to a cluster id present in R1:
    predicted = "same R1 family" (within-family)
    truth = (v6 score == 2)
  Report: precision, recall, F1, and a 3x2 confusion matrix
          (rows = v6 score 0/1/2; cols = within-family / different-family).

Run:
  python scripts/validate_r1_against_v6.py --task peer-review \\
      --r1-dirs r1_v4a r1_v4a_subagent r1_v4a_subagent_bs200
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_key2cluster(task: str) -> dict[str, int]:
    """Schema is already key -> cluster_id."""
    p = Path("outputs/analyses/structural_metrics") / f"clusters_{task}.json"
    return {k: int(v) for k, v in json.loads(p.read_text()).items()}


def load_cluster2family(r1_path: Path) -> dict[int, int]:
    d = json.loads(r1_path.read_text())
    fams = d.get("families", [])
    mp = {}
    for fi, f in enumerate(fams):
        for c in f.get("cluster_ids") or f.get("members") or []:
            s = str(c).strip().lstrip("C")
            try: mp[int(s)] = fi
            except ValueError: continue
    return mp, len(fams)


def validate(r1_path: Path, key2cluster: dict[str, int], pairs):
    cl2fam, n_fams = load_cluster2family(r1_path)
    valid_clusters = set(cl2fam)

    tp = fp = fn = tn = 0
    confusion = defaultdict(int)  # (v6_score, within_family) -> count
    n_skipped_unknown_key = 0
    n_skipped_unknown_cluster = 0
    n_used = 0

    for p in pairs:
        ka, kb = p["key_a"], p["key_b"]
        if ka not in key2cluster or kb not in key2cluster:
            n_skipped_unknown_key += 1
            continue
        ca, cb = key2cluster[ka], key2cluster[kb]
        if ca == cb:
            # same L0 cluster — trivially same family; skip (uninformative)
            continue
        if ca not in cl2fam or cb not in cl2fam:
            n_skipped_unknown_cluster += 1
            continue
        n_used += 1
        same_fam = (cl2fam[ca] == cl2fam[cb])
        score = p["score"]
        is_same_truth = (score == 2)
        confusion[(score, same_fam)] += 1
        if same_fam and is_same_truth: tp += 1
        elif same_fam and not is_same_truth: fp += 1
        elif not same_fam and is_same_truth: fn += 1
        else: tn += 1

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "r1": str(r1_path),
        "n_families": n_fams,
        "n_used_pairs": n_used,
        "n_skipped_unknown_key": n_skipped_unknown_key,
        "n_skipped_unknown_cluster": n_skipped_unknown_cluster,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "confusion": {f"v6={s}|same_fam={f}": n
                      for (s, f), n in confusion.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--r1-dirs", nargs="+", required=True)
    ap.add_argument("--verdicts",
                    default="outputs/analyses/structural_metrics/validation/"
                            "{task}_v6_verdicts.jsonl")
    args = ap.parse_args()

    verdicts_path = Path(args.verdicts.format(task=args.task))
    pairs = [json.loads(l) for l in verdicts_path.open()]
    print(f"loaded {len(pairs)} v6 pairs for {args.task}")

    key2cluster = load_key2cluster(args.task)
    print(f"loaded key->cluster mapping: {len(key2cluster)} keys -> "
          f"{len(set(key2cluster.values()))} clusters")

    base = Path("outputs/analyses/structural_metrics")
    results = []
    for d in args.r1_dirs:
        rp = base / d / f"r1_families_{args.task}.json"
        if not rp.exists():
            print(f"  SKIP {d}: no file at {rp}")
            continue
        r = validate(rp, key2cluster, pairs)
        results.append(r)

    print()
    print(f"{'r1_dir':<28} {'#fams':>7} {'#used':>7} {'TP':>5} {'FP':>5} "
          f"{'FN':>6} {'TN':>6} {'P':>6} {'R':>6} {'F1':>6}")
    for r in results:
        d = Path(r["r1"]).parts[-2]
        print(f"{d:<28} {r['n_families']:>7} {r['n_used_pairs']:>7} "
              f"{r['tp']:>5} {r['fp']:>5} {r['fn']:>6} {r['tn']:>6} "
              f"{r['precision']:>.3f} {r['recall']:>.3f} {r['f1']:>.3f}"
              .replace(" 0.", "  ."))
    print()
    print("Confusion (rows v6_score 0/1/2; cols same_fam False/True):")
    for r in results:
        d = Path(r["r1"]).parts[-2]
        print(f"\n  {d}:")
        for s in (0, 1, 2):
            t = r['confusion'].get(f"v6={s}|same_fam=True", 0)
            f = r['confusion'].get(f"v6={s}|same_fam=False", 0)
            print(f"    score={s}: same_fam={t:>5}  diff_fam={f:>6}")


if __name__ == "__main__":
    main()
