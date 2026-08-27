"""Tag the 141K-PR labeled corpus with V-axis verifiability status.

Per PR, decides whether its touched files yield clean V signal under our
test-execution pipeline. Routing rules:

  - repo not in repos.yaml             -> unknown_repo
  - repo status signal_strong          -> verifiable_clean
  - repo status signal_unverifiable /
    failed_build / not_built           -> unverifiable_repo
  - repo status signal_partial:
        any touched file under         -> unverifiable_infra
        infra_required_paths
        all touched under              -> verifiable_clean
        signal_paths
        otherwise                      -> verifiable_adjacency

Input  : 3 PR-level CSVs at
         datasets/code-review/code_review_dense_4096tok/{train,eval,test}.csv
         (paper_id = "owner/repo#pr_num")
         + per-PR diff files at
         datasets/code-review/diffs/{owner}__{repo}__{pr}.diff
Output : sidecar parquet
         outputs/code_review_v_verifiable_tags.parquet
         columns: paper_id, owner, repo, pr_num, v_verifiable_status,
                  n_touched, n_match_signal, n_match_infra, split
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT_DEFAULT = "/lfs/skampere3/0/alexspan/norm-research"
SPLITS = ("train", "eval", "test")

# statuses we treat as "give up at the repo level"
UNVERIFIABLE_STATUSES = {"signal_unverifiable", "failed_build", "not_built"}

DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/.+$")


def load_repos_yaml(path: Path) -> dict[str, dict]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    # normalize keys to lowercase for case-insensitive matching
    return {k.lower(): v for k, v in raw.items()}


def parse_diff_paths(diff_text: str) -> list[str]:
    """Pull the touched-file paths out of a unified diff."""
    out = []
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        m = DIFF_HEADER_RE.match(line)
        if m:
            out.append(m.group(1))
    return out


def _norm_glob(p: str) -> str:
    """Strip Go-style './...' wildcards into fnmatch-friendly globs."""
    p = p.strip()
    if p.endswith("/..."):
        p = p[:-3] + "**"
    elif p == "./...":
        p = "**"
    if p.startswith("./"):
        p = p[2:]
    return p


def _matches_any(file_path: str, globs: list[str]) -> bool:
    for g in globs:
        ng = _norm_glob(g)
        # treat a trailing slash as "anything under this dir"
        if ng.endswith("/"):
            if file_path.startswith(ng):
                return True
        elif fnmatch.fnmatch(file_path, ng) or file_path.startswith(ng):
            return True
    return False


def classify_pr(
    touched: list[str],
    cfg: dict,
) -> tuple[str, int, int]:
    """Return (status, n_match_signal, n_match_infra) for one PR."""
    status = (cfg.get("status") or "").lower()
    if status in UNVERIFIABLE_STATUSES:
        return ("unverifiable_repo", 0, 0)

    sig = cfg.get("signal_paths") or []
    infra = cfg.get("infra_required_paths") or []

    # filter out the obviously-non-glob entries (e.g. the mx-chain-go note
    # "touched dirs only — auto-resolved by runner from PR diff")
    sig = [s for s in sig if "/" in s or s.startswith("./") or "*" in s]
    infra = [s for s in infra if "/" in s or s.startswith("./") or "*" in s]

    if not touched:
        return ("verifiable_adjacency", 0, 0)

    n_sig = sum(1 for f in touched if _matches_any(f, sig)) if sig else 0
    n_infra = sum(1 for f in touched if _matches_any(f, infra)) if infra else 0

    if status == "signal_strong":
        # Strong-signal repos: every PR is "clean enough" (the runner handles
        # the touched-dirs resolution). Still surface infra hits if any.
        if n_infra > 0:
            return ("unverifiable_infra", n_sig, n_infra)
        return ("verifiable_clean", n_sig, n_infra)

    if status == "signal_partial":
        if n_infra > 0:
            return ("unverifiable_infra", n_sig, n_infra)
        if n_sig == len(touched) and n_sig > 0:
            return ("verifiable_clean", n_sig, n_infra)
        return ("verifiable_adjacency", n_sig, n_infra)

    # unknown / empty status defaults to repo-unverifiable
    return ("unverifiable_repo", n_sig, n_infra)


def load_diff_paths(diffs_dir: Path, owner: str, repo: str, pr_num: str) -> list[str]:
    p = diffs_dir / f"{owner}__{repo}__{pr_num}.diff"
    if not p.exists():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    return parse_diff_paths(text)


PAPER_ID_RE = re.compile(r"^(.+?)/(.+?)#(\d+)$")


def split_paper_id(paper_id: str) -> tuple[str, str, str] | None:
    m = PAPER_ID_RE.match(paper_id.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-yaml", required=True, type=Path)
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path(REPO_ROOT_DEFAULT) /
                    "datasets/code-review/code_review_dense_4096tok")
    ap.add_argument("--diffs-dir", type=Path,
                    default=Path(REPO_ROOT_DEFAULT) / "datasets/code-review/diffs")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--report-only", action="store_true",
                    help="Skip writing parquet; just print counts.")
    args = ap.parse_args()

    repos = load_repos_yaml(args.repos_yaml)
    print(f"[info] loaded {len(repos)} repos from {args.repos_yaml}",
          file=sys.stderr)

    rows = []
    for split in SPLITS:
        path = args.corpus_dir / f"{split}.csv"
        if not path.exists():
            print(f"[warn] missing split: {path}", file=sys.stderr)
            continue
        df = pd.read_csv(path, usecols=["paper_id", "judgement", "language"])
        df["split"] = split
        rows.append(df)
    corpus = pd.concat(rows, ignore_index=True)
    print(f"[info] corpus rows: {len(corpus)}", file=sys.stderr)

    out_records = []
    n_missing_diff = 0
    n_bad_id = 0
    for paper_id, judgement, lang, split in zip(
        corpus["paper_id"], corpus["judgement"], corpus["language"], corpus["split"]
    ):
        parts = split_paper_id(str(paper_id))
        if parts is None:
            n_bad_id += 1
            continue
        owner, repo, pr_num = parts
        repo_key = f"{owner}/{repo}".lower()
        cfg = repos.get(repo_key)

        if cfg is None:
            status = "unknown_repo"
            n_touched = n_sig = n_infra = 0
        else:
            touched = load_diff_paths(args.diffs_dir, owner, repo, pr_num)
            if not touched:
                n_missing_diff += 1
            status, n_sig, n_infra = classify_pr(touched, cfg)
            n_touched = len(touched)

        out_records.append({
            "paper_id": paper_id,
            "owner": owner,
            "repo": repo,
            "pr_num": int(pr_num),
            "split": split,
            "judgement": int(judgement) if pd.notna(judgement) else None,
            "language": lang,
            "v_verifiable_status": status,
            "n_touched": n_touched,
            "n_match_signal": n_sig,
            "n_match_infra": n_infra,
        })

    out_df = pd.DataFrame(out_records)
    print(f"[info] tagged rows: {len(out_df)} "
          f"(bad paper_id: {n_bad_id}, missing diff: {n_missing_diff})",
          file=sys.stderr)

    # Overall status counts
    print("\n=== PRs per v_verifiable_status ===")
    print(out_df["v_verifiable_status"].value_counts().to_string())

    # Top-20 repos by PR count, broken down by status
    top_repos = (
        out_df.assign(repo_key=out_df["owner"] + "/" + out_df["repo"])
        .groupby("repo_key").size().sort_values(ascending=False).head(20).index
    )
    sub = (out_df
           .assign(repo_key=out_df["owner"] + "/" + out_df["repo"])
           .query("repo_key in @top_repos"))
    pivot = (sub
             .groupby(["repo_key", "v_verifiable_status"]).size()
             .unstack(fill_value=0)
             .reindex(top_repos)
             .assign(total=lambda x: x.sum(axis=1))
             .sort_values("total", ascending=False))
    print("\n=== Top-20 repos x status ===")
    print(pivot.to_string())

    # Trial-batch repo summary
    trial = [
        "alteryx/evalml", "tektoncd/pipeline", "spiffe/spire",
        "multiversx/mx-chain-go", "apache/iceberg", "autotest/tp-qemu",
        "red-hat-storage/ocs-ci", "containerbuildsystem/atomic-reactor",
        "autotest/tp-libvirt", "easybuilders/easybuild-framework",
        "mozilla/addons-frontend", "metoppv/improver", "salto-io/salto",
    ]
    trial_sub = (out_df
                 .assign(repo_key=out_df["owner"] + "/" + out_df["repo"])
                 .query("repo_key in @trial"))
    trial_pivot = (trial_sub
                   .groupby(["repo_key", "v_verifiable_status"]).size()
                   .unstack(fill_value=0)
                   .assign(total=lambda x: x.sum(axis=1))
                   .sort_values("total", ascending=False))
    print("\n=== Trial-batch repos x status ===")
    print(trial_pivot.to_string())

    if not args.report_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(args.out, index=False)
        print(f"\n[info] wrote {args.out} ({len(out_df)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
