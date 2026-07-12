"""Split-half stability analysis for Face-1 metric-count certificates.

Test-retest reliability: compute certificates on even/odd probe splits + random half-subsamples
to measure stability of OPT_Omega (bits) and verdict (CODIFIABLE/DEEP/UNDERSAMPLED/FORM-DOMINATED).

Usage (CPU-only, no GPU):
    python -m methods.metric_implementer.experiments.split_half_stability \\
        --dir /path/to/checkpoint/dir --out stability_results.json

Outputs per domain: JSON with per-metric split-half results + aggregate stability stats
(Spearman correlation, flip rate, undersampled fraction).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

from . import alpha_probe as ap
from .value_certificate import certificate


def split_half_analysis(sigs: np.ndarray, M_i: np.ndarray, tags: List,
                        prompts: Optional[List[str]] = None,
                        form_invariance: Optional[dict] = None,
                        n_subsamples: int = 20, seed: int = 0) -> dict:
    """Run split-half stability analysis on one metric checkpoint.

    Args:
        sigs: (n_criteria, n_probes) criterion signatures
        M_i: (n_probes,) metric's own verdict (target)
        tags: source tags per criterion
        prompts: criterion strings (optional)
        form_invariance: form-invariance json if available
        n_subsamples: number of random half-subsamples (default 20)
        seed: random seed

    Returns:
        Dict with:
            - full_cert: certificate on all probes
            - even_cert: certificate on even indices
            - odd_cert: certificate on odd indices
            - subsample_certs: list of n_subsamples certificate dicts
            - verdicts: {full, even, odd, subsamples (list)}
            - opt_omegas: {full, even, odd, subsamples (list)}
    """
    n_probes = sigs.shape[1]
    rng = np.random.default_rng(seed)

    # Even/odd splits
    even_idx = np.arange(0, n_probes, 2)
    odd_idx = np.arange(1, n_probes, 2)

    # Full certificate
    full_cert = certificate(sigs, M_i, tags, prompts=prompts, quotient="behavioral",
                           delta=0.05, top_k=30, combiner="linear", n_orders=8,
                           seed=seed)
    full_verdict = ap.decide({"f1_over_N": full_cert["f1_over_N"]},
                            form_invariance=form_invariance, certificate=full_cert)

    # Even half (note: form_invariance applies to full metric, not subset)
    even_cert = certificate(sigs[:, even_idx], M_i[even_idx], tags, prompts=prompts,
                           quotient="behavioral", delta=0.05, top_k=30, combiner="linear",
                           n_orders=8, seed=seed)
    even_verdict = ap.decide({"f1_over_N": even_cert["f1_over_N"]},
                            certificate=even_cert)  # No form gate on subsets

    # Odd half
    odd_cert = certificate(sigs[:, odd_idx], M_i[odd_idx], tags, prompts=prompts,
                          quotient="behavioral", delta=0.05, top_k=30, combiner="linear",
                          n_orders=8, seed=seed)
    odd_verdict = ap.decide({"f1_over_N": odd_cert["f1_over_N"]},
                           certificate=odd_cert)

    # Random half-subsamples
    subsample_certs = []
    subsample_verdicts = []
    subsample_opts = []
    half_size = n_probes // 2

    for i in range(n_subsamples):
        sub_idx = rng.choice(n_probes, size=half_size, replace=False)
        sub_idx = np.sort(sub_idx)

        sub_cert = certificate(sigs[:, sub_idx], M_i[sub_idx], tags, prompts=prompts,
                              quotient="behavioral", delta=0.05, top_k=30, combiner="linear",
                              n_orders=8, seed=seed + i + 1)
        sub_verdict = ap.decide({"f1_over_N": sub_cert["f1_over_N"]},
                               certificate=sub_cert)

        subsample_certs.append({
            "opt_omega_bits": sub_cert["opt_omega_bits"],
            "eps_bits_adv": sub_cert.get("eps_bits_adv", sub_cert["eps_bits"]),
            "verdict": sub_verdict,
            "f1_over_N": sub_cert["f1_over_N"]
        })
        subsample_verdicts.append(sub_verdict)
        subsample_opts.append(sub_cert["opt_omega_bits"])

    return {
        "full_cert": {
            "opt_omega_bits": full_cert["opt_omega_bits"],
            "eps_bits_adv": full_cert.get("eps_bits_adv", full_cert["eps_bits"]),
            "verdict": full_verdict,
            "f1_over_N": full_cert["f1_over_N"]
        },
        "even_cert": {
            "opt_omega_bits": even_cert["opt_omega_bits"],
            "eps_bits_adv": even_cert.get("eps_bits_adv", even_cert["eps_bits"]),
            "verdict": even_verdict,
            "f1_over_N": even_cert["f1_over_N"]
        },
        "odd_cert": {
            "opt_omega_bits": odd_cert["opt_omega_bits"],
            "eps_bits_adv": odd_cert.get("eps_bits_adv", odd_cert["eps_bits"]),
            "verdict": odd_verdict,
            "f1_over_N": odd_cert["f1_over_N"]
        },
        "subsample_certs": subsample_certs,
        "verdicts": {
            "full": full_verdict,
            "even": even_verdict,
            "odd": odd_verdict,
            "subsamples": subsample_verdicts
        },
        "opt_omegas": {
            "full": full_cert["opt_omega_bits"],
            "even": even_cert["opt_omega_bits"],
            "odd": odd_cert["opt_omega_bits"],
            "subsamples": subsample_opts
        }
    }


def compute_aggregate_stats(results: List[dict]) -> dict:
    """Compute aggregate stability statistics across metrics.

    Args:
        results: list of per-metric split-half results

    Returns:
        Dict with:
            - spearman_even_odd: Spearman correlation of OPT_even vs OPT_odd
            - flip_rate: fraction of metrics where even/odd verdicts disagree
                        (among determinate verdicts only)
            - undersampled_fraction: fraction of half-samples that are UNDERSAMPLED
    """
    from scipy.stats import spearmanr

    even_opts = [r["opt_omegas"]["even"] for r in results]
    odd_opts = [r["opt_omegas"]["odd"] for r in results]

    # Spearman correlation
    if len(even_opts) >= 3:
        rho, pval = spearmanr(even_opts, odd_opts)
    else:
        rho, pval = float("nan"), float("nan")

    # Flip rate among determinate (non-UNDERSAMPLED) verdicts
    flips = 0
    determinate = 0
    for r in results:
        even_v = r["verdicts"]["even"]
        odd_v = r["verdicts"]["odd"]
        # Count only if BOTH halves are determinate
        if even_v not in ("UNDERSAMPLED", "INDETERMINATE", "NEEDS-CERTIFICATE") and \
           odd_v not in ("UNDERSAMPLED", "INDETERMINATE", "NEEDS-CERTIFICATE"):
            determinate += 1
            if even_v != odd_v:
                flips += 1

    flip_rate = flips / determinate if determinate > 0 else float("nan")

    # Undersampled fraction across all half-samples
    total_halves = 0
    undersampled_halves = 0
    for r in results:
        # Even and odd
        total_halves += 2
        if r["verdicts"]["even"] == "UNDERSAMPLED":
            undersampled_halves += 1
        if r["verdicts"]["odd"] == "UNDERSAMPLED":
            undersampled_halves += 1
        # Subsamples
        for v in r["verdicts"]["subsamples"]:
            total_halves += 1
            if v == "UNDERSAMPLED":
                undersampled_halves += 1

    undersampled_frac = undersampled_halves / total_halves if total_halves > 0 else float("nan")

    return {
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "flip_rate_determinate": float(flip_rate),
        "n_determinate_pairs": int(determinate),
        "undersampled_fraction": float(undersampled_frac),
        "n_total_halves": int(total_halves),
        "n_metrics": len(results)
    }


def run_directory(checkpoint_dir: str, n_subsamples: int = 20, seed: int = 0) -> dict:
    """Run split-half analysis on all metric checkpoints in a directory.

    Args:
        checkpoint_dir: path to directory with *_metric*_sigs.npz files
        n_subsamples: number of random half-subsamples per metric
        seed: random seed

    Returns:
        Dict with:
            - checkpoint_dir: path
            - per_metric: list of {name, file, split_half_results}
            - aggregate: aggregate stability statistics
    """
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*_metric*_sigs.npz")))

    if not ckpt_files:
        raise ValueError(f"No *_metric*_sigs.npz files found in {checkpoint_dir}")

    print(f"Found {len(ckpt_files)} checkpoint files in {checkpoint_dir}")

    per_metric = []

    for f in ckpt_files:
        try:
            z = np.load(f, allow_pickle=True)
            sigs = np.asarray(z["sigs"], float)
            tags = list(z["tags"])
            prompts = list(z["prompts"]) if "prompts" in z.files else None
            name = str(z["name"]) if "name" in z.files else os.path.basename(f)

            if "M_i" not in z.files:
                print(f"  SKIP {os.path.basename(f)}: no M_i in checkpoint")
                continue

            M_i = np.asarray(z["M_i"], float)

        except Exception as e:
            print(f"  SKIP {os.path.basename(f)}: {e}")
            continue

        # Check for form-invariance json (adjacent to npz)
        fi_path = f.replace("_sigs.npz", "_forminv.json")
        fi = json.load(open(fi_path)) if os.path.exists(fi_path) else None

        print(f"  Processing {name}...")

        result = split_half_analysis(sigs, M_i, tags, prompts=prompts,
                                    form_invariance=fi, n_subsamples=n_subsamples,
                                    seed=seed)

        per_metric.append({
            "name": name,
            "file": os.path.basename(f),
            **result
        })

        print(f"    Full: {result['verdicts']['full']} (OPT={result['opt_omegas']['full']:.3f})")
        print(f"    Even: {result['verdicts']['even']} (OPT={result['opt_omegas']['even']:.3f})")
        print(f"    Odd:  {result['verdicts']['odd']} (OPT={result['opt_omegas']['odd']:.3f})")

    # Aggregate statistics
    agg = compute_aggregate_stats(per_metric)

    print(f"\n=== Aggregate Stability Statistics ===")
    print(f"Spearman(OPT_even, OPT_odd): rho={agg['spearman_rho']:.3f}, p={agg['spearman_pval']:.4f}")
    print(f"Flip rate (determinate only): {agg['flip_rate_determinate']:.3f} ({agg['n_determinate_pairs']} pairs)")
    print(f"Undersampled fraction: {agg['undersampled_fraction']:.3f} ({agg['n_total_halves']} halves)")

    return {
        "checkpoint_dir": checkpoint_dir,
        "n_subsamples": n_subsamples,
        "seed": seed,
        "per_metric": per_metric,
        "aggregate": agg
    }


def main(argv=None):
    import sys
    p = argparse.ArgumentParser(prog="split_half_stability", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="checkpoint directory with *_metric*_sigs.npz files")
    p.add_argument("--out", required=True, help="output JSON file")
    p.add_argument("--n-subsamples", type=int, default=20,
                  help="number of random half-subsamples (default 20)")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    a = p.parse_args(argv if argv is not None else sys.argv[1:])

    results = run_directory(a.dir, n_subsamples=a.n_subsamples, seed=a.seed)

    # Save to JSON
    with open(a.out, "w") as f:
        json.dump(_clean_for_json(results), f, indent=2)

    print(f"\nResults saved to {a.out}")


def _clean_for_json(obj):
    """Recursively clean numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.complexfloating)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean_for_json(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


if __name__ == "__main__":
    main()
