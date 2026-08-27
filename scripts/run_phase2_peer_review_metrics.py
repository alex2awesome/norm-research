#!/usr/bin/env python3
"""Phase 2 prompt-optimization validation for peer-review metric clusters.

This runner is intentionally small and local:
  * samples 10 representative v6 clusters from clusters_peer-review.json;
  * resolves each representative metric id back to its parsed rubric text;
  * grows a 30-item Omega with deterministic GEPA-style paraphrase mutations;
  * runs the same OmegaCertificate path used by run_real_test.py;
  * computes held-out recovery, channel-cleanliness, and missing-impact metrics;
  * writes outputs/analyses/phase2_results.json.

The real vLLM backend can be used with --real-backend. By default this uses the
repository's FakeVLLM dry-run backend so the pipeline validates on a laptop.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.metric_implementer import config as cfgmod
from methods.metric_implementer import vinfo
from methods.metric_implementer.experiments.omega_certificate import OmegaCertificate
from methods.metric_implementer.experiments.orthogonalize import (
    adversarial_saturation,
    submodular_tail_bound,
)
from methods.metric_implementer.experiments.real_gamma import _median_split
from methods.metric_implementer.experiments.real_gamma import _signal
from methods.metric_implementer.experiments.small_omega_brute_force import _compile
from methods.metric_implementer.experiments.small_omega_brute_force import _criteria_from_rubric
from methods.metric_implementer.vllm_backend import make_judge_backend


CLUSTERS = ROOT / "outputs/analyses/structural_metrics/clusters_peer-review.json"
OUT_JSON = ROOT / "outputs/analyses/phase2_results.json"
WORK_DIR = ROOT / "outputs/analyses/phase2_peer_review_metrics"
PARSED_DIR = ROOT / "datasets/peer-review/online-rubrics/gpt-parsed/gpt-5-mini"
POOL = ROOT / "datasets/peer-review/splits/train.csv.gz"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _safe_slug(s: str, limit: int = 120) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return slug[:limit].strip("_") or "metric"


def load_clusters(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(f"missing clustering file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"clustering file has no metric assignments: {path}")
    return {str(k): int(v) for k, v in data.items()}


def cluster_records(clusters: dict[str, int]) -> list[dict]:
    members: dict[int, list[str]] = defaultdict(list)
    for key, cid in clusters.items():
        members[cid].append(key)
    out = []
    for cid, keys in members.items():
        out.append({"cluster_id": cid, "size": len(keys), "keys": sorted(keys)})
    return sorted(out, key=lambda r: (-r["size"], r["cluster_id"]))


def sample_representatives(records: list[dict], n: int) -> list[dict]:
    """Pick clusters spanning large, medium, small, and singleton sizes."""
    targets = [31, 29, 27, 20, 12, 8, 5, 3, 2, 1]
    picked: list[dict] = []
    used: set[int] = set()
    for target in targets:
        choices = [r for r in records if r["cluster_id"] not in used]
        if not choices:
            break
        best = min(choices, key=lambda r: (abs(r["size"] - target), -r["size"], r["cluster_id"]))
        used.add(best["cluster_id"])
        picked.append(best)
        if len(picked) == n:
            break
    if len(picked) < n:
        for r in records:
            if r["cluster_id"] not in used:
                picked.append(r)
                used.add(r["cluster_id"])
                if len(picked) == n:
                    break
    return picked


def _metric_file_candidates(source_dir: str, filename: str) -> list[Path]:
    exact = PARSED_DIR / f"{source_dir}__{filename}.json"
    if exact.exists():
        return [exact]
    return [Path(p) for p in sorted(glob.glob(str(PARSED_DIR / f"{source_dir}__{filename}*.json")))]


def _metric_text_from_obj(obj: dict) -> str:
    parts = []
    for key in ("name", "description", "guidance"):
        val = _clean(obj.get(key, ""))
        if val:
            parts.append(val)
    return ". ".join(parts)


def resolve_metric_text(metric_id: str) -> str:
    parts = metric_id.split("::")
    if len(parts) != 4:
        raise ValueError(f"unexpected metric id shape: {metric_id}")
    _task, source_dir, filename, idx_s = parts
    idx = int(idx_s)
    files = _metric_file_candidates(source_dir, filename)
    if not files:
        raise FileNotFoundError(f"no parsed rubric JSON found for {metric_id}")

    metrics = []
    for fp in files:
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError:
            continue
        extracted = data.get("extracted") or {}
        chunk_metrics = extracted.get("rubrics_metrics") or []
        if isinstance(chunk_metrics, list):
            metrics.extend(chunk_metrics)
        if len(files) == 1:
            break

    if idx >= len(metrics):
        raise IndexError(f"metric index {idx} out of range for {metric_id} ({len(metrics)} parsed)")
    text = _metric_text_from_obj(metrics[idx])
    if not text:
        raise ValueError(f"empty metric text for {metric_id}")
    return text


def generate_gepa_variations(metric_text: str, count: int = 30) -> list[str]:
    """Minimal GEPA-style local mutation pool when no per-metric GEPA API exists."""
    base = _clean(metric_text).rstrip(".")
    templates = [
        "{base}.",
        "Assess whether the peer review satisfies this criterion: {base}.",
        "Judge whether the review demonstrates the following standard: {base}.",
        "Evaluate the review for this requirement: {base}.",
        "Check that the review meets this expectation: {base}.",
        "Determine whether the review clearly addresses: {base}.",
        "Rate how well the review fulfills this standard: {base}.",
        "The review should be scored higher when it reflects: {base}.",
        "A strong review should show evidence of this property: {base}.",
        "Look for whether the reviewer has covered this point: {base}.",
        "Verify that the review attends to the following issue: {base}.",
        "Score the review by asking whether it includes: {base}.",
        "Use this as the evaluation rule for the review: {base}.",
        "Treat this as a positive review-quality signal: {base}.",
        "The criterion is met when the review substantively handles: {base}.",
    ]
    replacements = [
        ("clear", "explicit"),
        ("adequate", "sufficient"),
        ("provide", "supply"),
        ("include", "contain"),
        ("describe", "explain"),
        ("summary", "concise account"),
        ("relevant", "pertinent"),
        ("important", "material"),
        ("methods", "methodological approach"),
        ("results", "findings"),
        ("limitations", "constraints"),
        ("impact", "contribution"),
        ("quality", "merit"),
        ("evidence", "supporting evidence"),
        ("validity", "soundness"),
    ]

    candidates = [tmpl.format(base=base) for tmpl in templates]
    lower_base = base
    for old, new in replacements:
        pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
        if pattern.search(lower_base):
            candidates.append(pattern.sub(new, lower_base) + ".")
            candidates.append(f"Assess whether the review satisfies this restated criterion: {pattern.sub(new, lower_base)}.")

    prefixes = [
        "Does the review",
        "Is the review successful in showing that it",
        "Would a careful evaluator find that the review",
        "Does the reviewer substantively",
        "Can the review be said to",
        "Does the review give enough attention to whether it",
        "Does the review make a meaningful attempt to",
        "Does the review offer a reviewer-facing treatment of",
        "Is there clear evidence in the review that it",
        "Does the review's content support the conclusion that it",
    ]
    for prefix in prefixes:
        candidates.append(f"{prefix} satisfies this standard: {base}?")

    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = _clean(c)
        key = _norm(c)
        if len(c) > 12 and key not in seen:
            out.append(c)
            seen.add(key)
        if len(out) == count:
            return out

    i = 1
    while len(out) < count:
        c = f"GEPA local paraphrase {i}: evaluate whether the peer review meets this same standard: {base}."
        key = _norm(c)
        if key not in seen:
            out.append(c)
            seen.add(key)
        i += 1
    return out


def write_rubric(path: Path, variations: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"- {_clean(v)}" for v in variations) + "\n")


def load_certificate_texts(n_items: int) -> list[str]:
    df = pd.read_csv(POOL, usecols=["text"])
    df = df[df["text"].astype(str).str.len() > 50]
    return df["text"].astype(str).tolist()[:n_items]


def _subset_tuple(s: str) -> tuple[int, ...]:
    if not s:
        return ()
    return tuple(int(x) for x in str(s).split(",") if x != "")


def subset_candidates(npz_path: Path, K: int, *, max_subsets: int = 220) -> list[tuple[int, ...]]:
    d = np.load(npz_path, allow_pickle=True)
    out: list[tuple[int, ...]] = []
    for x in d["subset_order"]:
        subset = _subset_tuple(str(x))
        if subset and subset not in out:
            out.append(subset)
    must = [(i,) for i in range(K)] + [tuple(range(K))]
    for subset in must:
        if subset not in out:
            out.append(subset)
    if len(out) <= max_subsets:
        return out
    keep: list[tuple[int, ...]] = []
    for subset in must:
        if subset not in keep:
            keep.append(subset)
    for subset in out:
        if subset not in keep:
            keep.append(subset)
        if len(keep) >= max_subsets:
            break
    return keep


def score_measurement_views(
    rubric_file: Path,
    texts: list[str],
    subsets: list[tuple[int, ...]],
    *,
    model: str,
    fake: bool,
    compiler: str,
) -> tuple[np.ndarray, dict[tuple[int, ...], np.ndarray], int]:
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), "peer-review")
    cfg.vllm_fake = fake
    backend = make_judge_backend(model, cfg)
    crits = _criteria_from_rubric(str(rubric_file))
    K = len(crits)
    full = tuple(range(K))
    if full not in subsets:
        subsets = list(subsets) + [full]
    full_rubric = _compile(crits, full, compiler=compiler)
    M = _median_split(_signal(backend, full_rubric, texts, cfg.max_text_chars)).astype(float)
    scores: dict[tuple[int, ...], np.ndarray] = {}
    for j, subset in enumerate(subsets, start=1):
        rub = _compile(crits, subset, compiler=compiler)
        scores[subset] = _signal(backend, rub, texts, cfg.max_text_chars)
        if j % 50 == 0 or j == len(subsets):
            print(f"  measurement rescore: {j}/{len(subsets)} subsets", flush=True)
    return M, scores, K


def heldout_measure(
    M: np.ndarray,
    subset_scores: dict[tuple[int, ...], np.ndarray],
    K: int,
    *,
    frac: float,
    seed: int,
) -> tuple[float, dict, tuple[int, ...]]:
    n = len(M)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = max(4, int(round(n * (1.0 - frac))))
    tr, te = perm[:n_tr], perm[n_tr:]
    if len(te) < 4:
        raise ValueError(f"held-out split too small: n={n}, test={len(te)}")

    r_train = []
    r_test = []
    valid_subsets: list[tuple[int, ...]] = []
    for subset, vals in subset_scores.items():
        vals = np.asarray(vals, float)
        if vals.shape[0] != n or not np.isfinite(vals).all():
            continue
        rt = vinfo.tvd_recovery(vals[tr], M[tr], n_boot=100, n_perm=16, seed=seed)["tvd_recovery"]
        rv = vinfo.tvd_recovery(vals[te], M[te], n_boot=100, n_perm=16, seed=seed)["tvd_recovery"]
        if np.isfinite(rt) and np.isfinite(rv):
            valid_subsets.append(subset)
            r_train.append(float(rt))
            r_test.append(float(rv))
    if not valid_subsets:
        raise ValueError("no finite subset scores available for held-out recovery")

    best_pos = int(np.argmax(np.asarray(r_train)))
    s_star = valid_subsets[best_pos]
    guard = vinfo.tvd_guardrail(subset_scores[s_star][te], M[te], n_boot=100, n_perm=16, seed=seed)

    r_by_subset = {valid_subsets[k]: r_test[k] for k in range(len(valid_subsets))}
    full = tuple(range(K))
    full_r = r_by_subset.get(full)
    best_r = max(r_by_subset.values()) if r_by_subset else float("nan")
    no_prune_help = bool(full_r is None or not np.isfinite(best_r) or full_r + 0.01 >= best_r)
    detail = {
        "S_star_train": list(s_star),
        "R_train_star": r_train[best_pos],
        "R_test_star": float(guard["R_tvd"]),
        "T_test": float(guard["T_tvd"]),
        "A_test": float(guard["A_tvd"]),
        "dpi_ok": bool(guard["dpi_tvd_ok"]),
        "no_prune_help_scored": no_prune_help,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }
    return float(guard["R_tvd"]), detail, s_star


def missing_impact_measure(
    M: np.ndarray,
    subset_scores: dict[tuple[int, ...], np.ndarray],
    K: int,
    texts: list[str],
) -> tuple[float, dict]:
    singleton_cols = []
    for e in range(K):
        vals = subset_scores.get((e,))
        if vals is None:
            continue
        singleton_cols.append(_median_split(vals))
    if len(singleton_cols) < 3:
        return 1.0, {"reason": "fewer than 3 singleton criterion signals", "K_omega": len(singleton_cols)}

    X_omega = np.column_stack(singleton_cols).astype(int)
    tb = submodular_tail_bound(M, X_omega)
    lengths = np.asarray([len(t) for t in texts[: len(M)]])
    probe = (lengths >= np.median(lengths)).astype(int)[:, None]
    sat = adversarial_saturation(M, X_omega, probe)
    bound = float(tb.get("certified_bound", float("nan")))
    if not np.isfinite(bound):
        bound = 1.0
    detail = {
        "K_omega": int(X_omega.shape[1]),
        "tail_bound_certified": bound,
        "tail_bound_loo": float(tb.get("tail_bound", float("nan"))),
        "saturated": bool(sat.get("saturated")),
        "max_probe_cmi": float(sat.get("max_cmi", float("nan"))),
    }
    return bound, detail


def cleanliness_score(heldout: dict, missing: dict) -> float:
    checks = [
        bool(heldout.get("dpi_ok")),
        bool(heldout.get("no_prune_help_scored")),
        bool(missing.get("saturated")),
    ]
    return float(sum(checks) / len(checks))


def finite_float(x: float, *, name: str, metric_id: str) -> float:
    if x is None or not np.isfinite(float(x)):
        raise ValueError(f"non-finite {name} for {metric_id}: {x}")
    return float(x)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=10)
    ap.add_argument("--variations", type=int, default=30)
    ap.add_argument("--n-items", type=int, default=30)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--large-k", type=int, default=15)
    ap.add_argument("--holdout-frac", type=float, default=0.5)
    ap.add_argument("--holdout-seed", type=int, default=0)
    ap.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--compiler", default="conjunction", choices=["conjunction", "weighted_sum", "prose_join"])
    ap.add_argument("--real-backend", action="store_true", help="Use real vLLM backend instead of FakeVLLM")
    ap.add_argument("--verbose-cert", action="store_true", help="Show OmegaCertificate's per-subset logs")
    args = ap.parse_args()

    clusters = load_clusters(CLUSTERS)
    records = cluster_records(clusters)
    reps = sample_representatives(records, args.sample_size)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Verified clustering: {CLUSTERS}")
    print(f"  metric assignments: {len(clusters)}")
    print(f"  clusters: {len(records)}")
    print(f"  sampled representatives: {len(reps)}")
    print(f"Backend: {'real vLLM' if args.real_backend else 'FakeVLLM dry-run'}")

    texts = load_certificate_texts(args.n_items)
    print(f"Loaded {len(texts)} peer-review texts from {POOL}")

    rows = []
    for i, rec in enumerate(reps, start=1):
        metric_id = rec["keys"][0]
        print(f"\n[{i}/{len(reps)}] metric={metric_id} cluster={rec['cluster_id']} size={rec['size']}", flush=True)
        original_text = resolve_metric_text(metric_id)
        variations = generate_gepa_variations(original_text, args.variations)
        if len(variations) < 25:
            raise RuntimeError(f"variation generator under-produced for {metric_id}: {len(variations)}")
        print(f"  GEPA-style Omega growth: {len(variations)} variations", flush=True)

        slug = _safe_slug(metric_id)
        rubric_file = WORK_DIR / f"{i:02d}_{slug}.rubric.txt"
        npz_file = WORK_DIR / f"{i:02d}_{slug}.npz"
        write_rubric(rubric_file, variations)

        cert = OmegaCertificate(
            rubric_file=str(rubric_file),
            pool=str(POOL),
            text_col="text",
            task="peer-review",
            model=args.target_model,
            n_items=args.n_items,
            budget=args.budget,
            large_k=args.large_k,
            fake=not args.real_backend,
            out=str(npz_file),
            compiler=args.compiler,
            prose_prompt=original_text,
            holdout_frac=args.holdout_frac,
            holdout_seed=args.holdout_seed,
        )
        if args.verbose_cert:
            result = cert.run()
        else:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = cert.run()
            print(f"  certificate log suppressed ({len(buf.getvalue().splitlines())} lines); use --verbose-cert to show", flush=True)
        print(f"  certificate executed: mode={result.get('mode')} K={result.get('K')} subsets={result.get('subsets_scored')}", flush=True)

        K = int(result.get("K") or len(variations))
        candidates = subset_candidates(npz_file, K)
        M_meas, subset_scores, K_meas = score_measurement_views(
            rubric_file,
            texts,
            candidates,
            model=args.target_model,
            fake=not args.real_backend,
            compiler=args.compiler,
        )
        recovery, heldout_detail, _ = heldout_measure(
            M_meas,
            subset_scores,
            K_meas,
            frac=args.holdout_frac,
            seed=args.holdout_seed,
        )
        missing_impact, missing_detail = missing_impact_measure(M_meas, subset_scores, K_meas, texts)
        cleanliness = cleanliness_score(heldout_detail, missing_detail)
        print(
            "  measurements: "
            f"heldout_recovery={recovery:.4f} "
            f"cleanliness={cleanliness:.3f} "
            f"missing_impact={missing_impact:.4f}",
            flush=True,
        )

        rows.append(
            {
                "metric_id": metric_id,
                "original_text": original_text,
                "variations_count": int(len(variations)),
                "recovery_score": finite_float(recovery, name="recovery", metric_id=metric_id),
                "channel_cleanliness": finite_float(cleanliness, name="cleanliness", metric_id=metric_id),
                "missing_impact": finite_float(missing_impact, name="missing_impact", metric_id=metric_id),
            }
        )

    payload = {
        "metrics": rows,
        "summary": {
            "total_processed": len(rows),
            "mean_recovery": mean([r["recovery_score"] for r in rows]),
            "mean_cleanliness": mean([r["channel_cleanliness"] for r in rows]),
            "mean_missing_impact": mean([r["missing_impact"] for r in rows]),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"\nWrote valid Phase 2 JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
