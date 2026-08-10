"""Extensive validation of the pure-Python MOB engine against R ``partykit::glmtree``.

Two layers, both over ~100 synthetic datasets with *known* instability structure:

A. **Python self-validation** (always runs, no R): on scenarios with a planted coefficient
   break the engine should detect the break on the correct ``z`` (power); on null scenarios it
   should not (size / false-positive rate). Also checks split-variable and cutpoint recovery.

B. **R parity** (runs iff ``Rscript`` + partykit are installed): for each dataset we fit
   ``glmtree(y ~ X | z, binomial)`` in R and compare
     - selected split variable (target: high agreement),
     - split cutpoint (within tolerance),
     - per-``z`` test statistic vs ``strucchange``'s ``sctest`` (rank correlation -- scaling
       invariant, the "closeness to R" check independent of permutation-vs-asymptotic p).

Run:
    PYTHONPATH=methods python -m metrics_tree_infilling.tests.validate_against_partykit --n 100
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from methods.metrics_tree_infilling.config import InfillConfig
from methods.metrics_tree_infilling.mob.glmtree import GapTree
from methods.metrics_tree_infilling.mob.mfluctuation import (
    cov_inverse, fit_node_glm, score_contributions, test_node,
)


# --------------------------------------------------------------------------------------
# Scenario generation
# --------------------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    X: np.ndarray                  # (n, p) within-node features
    y: np.ndarray                  # (n,)
    z: Dict[str, tuple]            # var -> (values, kind)
    has_break: bool
    break_var: Optional[str]
    break_kind: Optional[str]
    true_cut: Optional[float]


def generate_scenarios(n_datasets: int, seed: int = 0) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    for d in range(n_datasets):
        n = int(rng.choice([300, 500, 800, 1200]))
        p = int(rng.choice([1, 2, 3]))
        n_znum = int(rng.choice([2, 3]))
        n_zcat = int(rng.choice([1, 2]))
        beta = float(rng.uniform(1.8, 3.5))
        kind = "null" if rng.uniform() < 0.30 else rng.choice(["numeric", "categorical"])

        X = rng.normal(size=(n, p))
        z = {}
        for k in range(n_znum):
            z[f"znum{k}"] = (rng.uniform(size=n), "numeric")
        for k in range(n_zcat):
            z[f"zcat{k}"] = (rng.integers(0, int(rng.choice([2, 3, 4])), size=n), "categorical")

        break_var = break_kind = None
        true_cut = None
        if kind == "numeric":
            break_var = f"znum{int(rng.integers(n_znum))}"
            cut = float(rng.uniform(0.35, 0.65))
            true_cut, break_kind = cut, "numeric"
            sign = np.where(z[break_var][0] >= cut, -1.0, 1.0)
            logit = sign * beta * X[:, 0]
        elif kind == "categorical":
            break_var = f"zcat{int(rng.integers(n_zcat))}"
            break_kind = "categorical"
            vals = z[break_var][0]
            lvls = np.unique(vals)
            grp = np.isin(vals, lvls[: max(1, len(lvls) // 2)])
            logit = np.where(grp, beta, -beta) * X[:, 0]
        else:
            logit = beta * X[:, 0]    # stable: y depends on X, no z modulation

        y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
        scenarios.append(Scenario(
            name=f"d{d:03d}_{kind}", X=X, y=y, z=z,
            has_break=(kind != "null"), break_var=break_var,
            break_kind=break_kind, true_cut=true_cut,
        ))
    return scenarios


# --------------------------------------------------------------------------------------
# Python side
# --------------------------------------------------------------------------------------

@dataclass
class PyResult:
    scenario: str
    per_z_stat: Dict[str, float]
    per_z_p: Dict[str, float]
    selected_var: Optional[str]
    selected_cut: Optional[float]


def run_python(sc: Scenario, cfg: InfillConfig) -> PyResult:
    rng = np.random.default_rng(cfg.random_seed)
    _, p, X_design = fit_node_glm(sc.X, sc.y)
    psi = score_contributions(X_design, sc.y, p)
    Jinv = cov_inverse(psi)
    res = test_node(psi, sc.z, trim=cfg.mfluct_trim, n_perm=cfg.n_permutations,
                    bonferroni=cfg.bonferroni, rng=rng, Jinv=Jinv)
    per_stat = {r.variable: r.statistic for r in res}
    per_p = {r.variable: r.adj_pvalue for r in res}

    # selected var + cutpoint via a depth-1 GapTree fit
    tree = GapTree(InfillConfig(**{**cfg.__dict__, "max_depth": 1})).fit(
        sc.X, sc.y, sc.z, feature_names=[f"x{i}" for i in range(sc.X.shape[1])])
    sel_var = tree.root.split.variable if tree.root.split else None
    sel_cut = tree.root.split.threshold if (tree.root.split and tree.root.split.kind == "numeric") else None
    return PyResult(sc.name, per_stat, per_p, sel_var, sel_cut)


# --------------------------------------------------------------------------------------
# R side
# --------------------------------------------------------------------------------------

_R_SCRIPT = r"""
suppressMessages({library(partykit); library(strucchange)})
args <- commandArgs(trailingOnly=TRUE)
manifest <- read.csv(args[1], stringsAsFactors=FALSE)
out <- list()
for (i in seq_len(nrow(manifest))) {
  csv <- manifest$csv[i]; nm <- manifest$name[i]
  df <- read.csv(csv)
  xs <- grep('^x', names(df), value=TRUE)
  zs <- grep('^z', names(df), value=TRUE)
  for (zc in grep('^zcat', zs, value=TRUE)) df[[zc]] <- as.factor(df[[zc]])
  fml <- as.formula(paste0('y ~ ', paste(xs, collapse='+'), ' | ', paste(zs, collapse='+')))
  res <- list(name=nm, sel_var=NA, sel_cut=NA, stat=list(), p=list())
  tryCatch({
    m <- glmtree(fml, data=df, family=binomial, maxdepth=2, alpha=0.05)
    st <- tryCatch(sctest(m, node=1L), error=function(e) NULL)
    if (!is.null(st)) {
      for (zc in colnames(st)) {
        res$stat[[zc]] <- unname(st['statistic', zc])
        res$p[[zc]]    <- unname(st['p.value', zc])
      }
    }
    rn <- node_party(m)
    if (!is.terminal(rn)) {
      sp <- split_node(rn); vid <- varid_split(sp)
      res$sel_var <- names(df)[vid]
      br <- breaks_split(sp); if (!is.null(br)) res$sel_cut <- br[1]
    }
  }, error=function(e) {})
  out[[i]] <- res
}
writeLines(jsonlite::toJSON(out, auto_unbox=TRUE, na='null'), args[2])
"""


def run_r(scenarios: List[Scenario], workdir: Path) -> Optional[Dict[str, dict]]:
    if shutil.which("Rscript") is None:
        return None
    import pandas as pd
    rows = []
    for sc in scenarios:
        df = pd.DataFrame({f"x{i}": sc.X[:, i] for i in range(sc.X.shape[1])})
        for name, (vals, _) in sc.z.items():
            df[name] = vals
        df["y"] = sc.y.astype(int)
        csv = workdir / f"{sc.name}.csv"
        df.to_csv(csv, index=False)
        rows.append({"name": sc.name, "csv": str(csv)})
    manifest = workdir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    script = workdir / "validate.R"
    script.write_text(_R_SCRIPT)
    out_json = workdir / "r_results.json"
    try:
        subprocess.run(["Rscript", str(script), str(manifest), str(out_json)],
                       check=True, capture_output=True, text=True, timeout=3600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print("R run failed:", getattr(e, "stderr", e))
        return None
    data = json.loads(out_json.read_text())
    return {r["name"]: r for r in data}


# --------------------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------------------

def python_self_report(scenarios: List[Scenario], pyres: List[PyResult], alpha: float) -> dict:
    detect = fp = n_break = n_null = 0
    var_correct = cut_err = cut_n = 0
    for sc, pr in zip(scenarios, pyres):
        flagged = pr.selected_var is not None
        if sc.has_break:
            n_break += 1
            if flagged:
                detect += 1
                if pr.selected_var == sc.break_var:
                    var_correct += 1
                if sc.break_kind == "numeric" and pr.selected_cut is not None and sc.true_cut is not None:
                    cut_err += abs(pr.selected_cut - sc.true_cut)
                    cut_n += 1
        else:
            n_null += 1
            fp += int(flagged)
    return {
        "n_break": n_break, "n_null": n_null,
        "detection_rate": detect / max(n_break, 1),
        "false_positive_rate": fp / max(n_null, 1),
        "correct_var_rate": var_correct / max(detect, 1),
        "mean_cut_abs_err": (cut_err / cut_n) if cut_n else None,
    }


def r_parity_report(scenarios: List[Scenario], pyres: List[PyResult],
                    rres: Dict[str, dict]) -> dict:
    from scipy.stats import spearmanr
    var_match = both = 0
    cut_diffs, py_stats, r_stats = [], [], []
    for sc, pr in zip(scenarios, pyres):
        rr = rres.get(sc.name)
        if rr is None:
            continue
        both += 1
        if pr.selected_var == (rr.get("sel_var") or None):
            var_match += 1
        if pr.selected_cut is not None and rr.get("sel_cut") is not None:
            cut_diffs.append(abs(float(pr.selected_cut) - float(rr["sel_cut"])))
        for zc, st in (rr.get("stat") or {}).items():
            if zc in pr.per_z_stat and st is not None:
                py_stats.append(pr.per_z_stat[zc]); r_stats.append(float(st))
    rho = float(spearmanr(py_stats, r_stats).correlation) if len(py_stats) > 2 else None
    return {
        "n_compared": both,
        "split_var_agreement": var_match / max(both, 1),
        "median_cut_abs_diff": float(np.median(cut_diffs)) if cut_diffs else None,
        "statistic_spearman": rho,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-perm", type=int, default=499)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = InfillConfig(n_permutations=args.n_perm, min_node_size=30, random_seed=args.seed)
    scenarios = generate_scenarios(args.n, seed=args.seed)
    print(f"Generated {len(scenarios)} scenarios; running Python MOB ...")
    pyres = [run_python(sc, cfg) for sc in scenarios]

    report = {"python_self": python_self_report(scenarios, pyres, cfg.alpha)}
    print("\n== Python self-validation ==")
    for k, v in report["python_self"].items():
        print(f"  {k}: {v}")

    with tempfile.TemporaryDirectory() as td:
        rres = run_r(scenarios, Path(td))
        if rres is None:
            print("\n[R parity skipped] Rscript/partykit not available. "
                  "Install R + partykit + strucchange + jsonlite to enable.")
        else:
            report["r_parity"] = r_parity_report(scenarios, pyres, rres)
            print("\n== R parity (vs partykit::glmtree) ==")
            for k, v in report["r_parity"].items():
                print(f"  {k}: {v}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out}")
    return 0


# fast pytest: Python self-validation on a modest batch (no R required)
def test_python_self_validation():
    cfg = InfillConfig(n_permutations=99, min_node_size=30, random_seed=1)
    scenarios = generate_scenarios(24, seed=1)
    pyres = [run_python(sc, cfg) for sc in scenarios]
    rep = python_self_report(scenarios, pyres, cfg.alpha)
    assert rep["detection_rate"] >= 0.75, rep
    assert rep["false_positive_rate"] <= 0.25, rep
    assert rep["correct_var_rate"] >= 0.70, rep


if __name__ == "__main__":
    raise SystemExit(main())
