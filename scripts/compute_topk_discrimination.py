#!/usr/bin/env python
"""Post-hoc: compute mean|mean_p - 0.5| over top-K clusters from a sweep dir.

Rescues the metric for already-run sweeps whose driver predated it. Reads each
top_{K}_named_*.json in a sweep directory and augments sweep_summary.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()

    named_files = sorted(args.sweep_dir.glob("top_*_named_*.json"))
    if not named_files:
        logger.error("No top_*_named_*.json files in %s", args.sweep_dir)
        sys.exit(1)

    rows = []
    for path in named_files:
        with open(path) as f:
            named = json.load(f)
        # Parse config from filename, e.g. top_30_named_tw0.25.json or top_30_named_eps0.40.json
        name = path.stem.split("_named_", 1)[-1]
        top_k = int(path.stem.split("_")[1])
        devs = [abs(n["mean_p_positive"] - 0.5) for n in named
                if n.get("mean_p_positive") is not None]
        mean_abs_p_dev = sum(devs) / len(devs) if devs else 0.0
        # Also count top-K pure clusters (|p-0.5| >= 0.4)
        n_pure = sum(1 for d in devs if d >= 0.4)
        n_mixed = sum(1 for d in devs if d < 0.1)
        rows.append({
            "config": name,
            "top_k": top_k,
            "n_named": len(named),
            "mean_abs_p_dev_topk": mean_abs_p_dev,
            "n_pure_topk": n_pure,
            "n_mixed_topk": n_mixed,
        })
        logger.info("%s: K=%d, mean|p-0.5|_topk=%.3f (%d pure, %d near-0.5)",
                    name, top_k, mean_abs_p_dev, n_pure, n_mixed)

    out = args.sweep_dir / "topk_discrimination.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Saved → %s", out)

    # Also augment sweep_summary.json if present
    summary_path = args.sweep_dir / "sweep_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        lookup = {r["config"]: r for r in rows}
        for s in summary:
            # Build the config key the same way name_files produced
            for key_candidate in (
                f"tw{s.get('target_weight', 0):.2f}",
                f"eps{s.get('epsilon', 0):.2f}",
            ):
                if key_candidate in lookup:
                    s["mean_abs_p_dev_topk"] = lookup[key_candidate]["mean_abs_p_dev_topk"]
                    s["n_pure_topk"] = lookup[key_candidate]["n_pure_topk"]
                    s["n_mixed_topk"] = lookup[key_candidate]["n_mixed_topk"]
                    break
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Augmented %s", summary_path)


if __name__ == "__main__":
    main()
