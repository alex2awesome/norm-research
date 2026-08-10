#!/usr/bin/env python3
"""First-push reconstruction + cheap V-metric pilot (mathlib friction PRs).

Context: datasets/math/mathlib/README.md (§4 data plan, §5 "Where V
discrimination actually lives") and METRICS.md. V metrics must be computed on
the FIRST-PUSH state, not the merged state (the merged state is
review-polished and saturates them by construction).

Pipeline:
 1. Sample 200 PRs (100/class, split=train, seed 42) from
    friction_dataset_v2.csv.gz.
 2. Batch-fetch refs/pull/N/head into refs/remotes/pr/N (~50 refspecs per
    fetch, --no-tags; per-ref retry on batch failure; failures recorded).
 3. State resolution: first_commit_oid if the object is present after the
    fetch ('first_commit'), else the fetched pr/N head ('head_fallback').
 4. Diff `git merge-base <state> origin/master` .. <state> over '*.lean',
    extract ADDED lines, compute text-rule metrics v02,v04-v07,v10,v13,
    v15,v16,v17.
 5. Per-metric point-biserial r + single-feature AUC vs y, combined logistic
    regression (5-fold stratified CV), printed as markdown.

Approximations (documented; no Lean toolchain):
- "first-push state": first_commit_oid comes from GraphQL commits(first:1),
  which reflects the FINAL branch state. If the branch was force-pushed, the
  true original push is unrecoverable here; first_commit_oid is the first
  commit of the rewritten branch. 'head_fallback' = full final PR head.
- v06 non-terminal simp: a simp tactic call is flagged non-terminal if
  (a) followed on the same line by ';' or '<;>' plus more tactics, or
  (b) the next non-blank/non-comment ADDED line in the same consecutive
  added-line run starts at column >= the simp token's column and looks
  tactic-like (not a declaration/section keyword, closing bracket, or a
  focus dot at shallower indent). Multi-line `simp only [...]` lemma lists
  are skipped via bracket balancing. Because only added lines are visible,
  simps at hunk edges can be miscounted in either direction.
- v15 proof blocks: a block opens at `:= by` (or a line ending in `by`) and
  extends over subsequent lines of the same added-line run with column >
  the opening line's indent. Runs truncate at hunk boundaries, so block
  lengths are lower bounds. Term-vs-tactic mode is judged from the first
  `:=` within 20 lines of each theorem/lemma/def header.
- v10: docstring = an added line ending in `-/` directly above the decl
  (skipping attribute `@[...]` lines); a docstring present as diff CONTEXT
  (unmodified) above an added decl is invisible -> overcounts "missing".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
REPO = os.path.join(BASE, "mathlib4_repo")
GIT_ENV = dict(os.environ, HOME="/lfs/skampere3/0/alexspan")

# ---------------------------------------------------------------- git helpers

def git(*args: str, check: bool = False, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", REPO, *args], env=GIT_ENV,
                          capture_output=True, text=True, check=check,
                          timeout=timeout)


def batch_fetch(numbers: list[int], batch: int = 50) -> dict[int, bool]:
    """Fetch refs/pull/N/head -> refs/remotes/pr/N in batches; per-ref retry."""
    ok: dict[int, bool] = {}
    for i in range(0, len(numbers), batch):
        chunk = numbers[i:i + batch]
        refspecs = [f"+pull/{n}/head:refs/remotes/pr/{n}" for n in chunk]
        r = git("fetch", "origin", "--no-tags", "--quiet", *refspecs,
                timeout=1200)
        if r.returncode == 0:
            for n in chunk:
                ok[n] = True
        else:
            print(f"  batch {i//batch} failed (rc={r.returncode}); "
                  f"retrying per-ref", file=sys.stderr)
            for n in chunk:
                r1 = git("fetch", "origin", "--no-tags", "--quiet",
                         f"+pull/{n}/head:refs/remotes/pr/{n}")
                ok[n] = (r1.returncode == 0)
                if not ok[n]:
                    print(f"    pr {n}: fetch failed: "
                          f"{r1.stderr.strip()[:200]}", file=sys.stderr)
        print(f"  fetched batch {i//batch + 1}/"
              f"{(len(numbers)+batch-1)//batch}", file=sys.stderr)
    return ok


def object_exists(oid: str) -> bool:
    return git("cat-file", "-e", f"{oid}^{{commit}}").returncode == 0


def resolve_state(number: int, first_commit_oid: str,
                  head_fetched: bool) -> tuple[str | None, str | None]:
    """Return (rev, state_used)."""
    if isinstance(first_commit_oid, str) and object_exists(first_commit_oid):
        return first_commit_oid, "first_commit"
    if head_fetched:
        ref = f"refs/remotes/pr/{number}"
        if git("rev-parse", "--verify", ref).returncode == 0:
            return ref, "head_fallback"
    return None, None


def added_line_runs(rev: str) -> list[list[str]] | None:
    """Diff merge-base(rev, origin/master)..rev over *.lean; return runs of
    consecutive added lines (per hunk segment)."""
    mb = git("merge-base", rev, "origin/master")
    if mb.returncode != 0:
        return None
    base = mb.stdout.strip()
    d = git("diff", "--no-color", base, rev, "--", "*.lean", timeout=600)
    if d.returncode != 0:
        return None
    runs: list[list[str]] = []
    cur: list[str] = []
    for line in d.stdout.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            cur.append(line[1:])
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs

# ---------------------------------------------------------------- metric rules

DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|scoped|local)\s+)*"
    r"(theorem|lemma|def|abbrev|instance|structure|inductive|class)\s")
TLD = {"theorem", "lemma", "def"}
SIMP_RE = re.compile(r"(?<![\w_'])simp\b")
SIMP_UNSQUEEZED_RE = re.compile(r"(?<![\w_'])simp\b(?!\s+only\b)")
ERW_RE = re.compile(r"(?<![\w_'])erw\b")
FUN_ARROW_RE = re.compile(r"\bfun\b[^\n↦=]*=>")
FOCUS_DOT_RE = re.compile(r"^\s*·")
BY_OPEN_RE = re.compile(r":=\s*by\b|\bby\s*$")
NONTACTIC_START = re.compile(
    r"^\s*(?:end\b|namespace\b|section\b|variable|open\b|import\b|attribute\b"
    r"|universe\b|set_option\b|#|/--|--|@\[|deriving\b|where\b|[\]\)\}]+\s*$)")


ATTR_SPAN_RE = re.compile(r"@\[[^\]]*\]")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def strip_attrs(code: str) -> str:
    """Blank out @[...] attribute spans so `@[simp]` etc. don't count as
    tactic uses (preserves column positions)."""
    return ATTR_SPAN_RE.sub(lambda m: " " * len(m.group(0)), code)


def comment_mask(run: list[str]) -> list[bool]:
    """True for lines that are entirely comment (line `--` or inside /- -/)."""
    mask, depth = [], 0
    for line in run:
        s = line.strip()
        starts_in_comment = depth > 0
        depth += line.count("/-") - line.count("-/")
        depth = max(depth, 0)
        mask.append(starts_in_comment or s.startswith("--")
                    or s.startswith("/-"))
    return mask


def simp_metrics(run: list[str], mask: list[bool]) -> tuple[int, int, int]:
    """Return (n_simp, n_unsqueezed, n_nonterminal) for one run."""
    n_simp = n_unsq = n_nonterm = 0
    for i, raw in enumerate(run):
        if mask[i]:
            continue
        line = strip_attrs(raw)
        for m in SIMP_RE.finditer(line):
            n_simp += 1
            col = m.start()
            rest = line[m.end():]
            # multi-line bracket list: find the line where brackets balance
            bal = rest.count("[") - rest.count("]")
            j = i
            while bal > 0 and j + 1 < len(run):
                j += 1
                bal += run[j].count("[") - run[j].count("]")
            tail = rest if j == i else run[j]
            if re.search(r"(<;>|;)\s*\S", tail.split("--")[0]):
                n_nonterm += 1
            else:
                # next effective added line in this run
                k = j + 1
                while k < len(run) and (not run[k].strip() or mask[k]):
                    k += 1
                if k < len(run):
                    nxt = run[k]
                    if (indent_of(nxt) >= col
                            and not NONTACTIC_START.match(nxt)
                            and not DECL_RE.match(nxt)
                            and not (FOCUS_DOT_RE.match(nxt)
                                     and indent_of(nxt) < col)):
                        n_nonterm += 1
        for _ in SIMP_UNSQUEEZED_RE.finditer(line.split("--")[0]):
            n_unsq += 1
    return n_simp, n_unsq, n_nonterm


def decl_metrics(run: list[str], mask: list[bool]) -> tuple[int, int, int, int, int]:
    """(n_decls, n_tld, n_tld_missing_doc, n_tactic_mode, n_term_mode)."""
    n_decls = n_tld = n_missing = n_tac = n_term = 0
    for i, line in enumerate(run):
        if mask[i]:
            continue
        m = DECL_RE.match(line)
        if not m:
            continue
        n_decls += 1
        if m.group(1) not in TLD:
            continue
        n_tld += 1
        # docstring: walk back over attribute-only / blank lines; the
        # comment must OPEN with `/--` (not a plain `/-` header block)
        j = i - 1
        while j >= 0 and (not run[j].strip()
                          or run[j].strip().startswith("@[")):
            j -= 1
        has_doc = False
        if j >= 0 and run[j].strip().endswith("-/"):
            for k in range(j, max(j - 30, -1), -1):
                s = run[k].strip()
                if s.startswith("/--"):
                    has_doc = True
                    break
                if s.startswith("/-"):
                    break
        if not has_doc:
            n_missing += 1
        # term vs tactic: first ':=' within 20 lines
        for k in range(i, min(i + 20, len(run))):
            seg = run[k]
            if ":=" in seg:
                after = seg.split(":=", 1)[1]
                if re.match(r"\s*by\b", after):
                    n_tac += 1
                elif after.strip() == "" and k + 1 < len(run) \
                        and run[k + 1].lstrip().startswith("by"):
                    n_tac += 1
                else:
                    n_term += 1
                break
            if DECL_RE.match(seg) and k > i:
                break
    return n_decls, n_tld, n_missing, n_tac, n_term


def proof_blocks(run: list[str], mask: list[bool]) -> list[int]:
    """Lengths (non-blank lines) of `:= by` blocks within this run."""
    lengths = []
    for i, line in enumerate(run):
        if mask[i] or not BY_OPEN_RE.search(line.split("--")[0]):
            continue
        ind = indent_of(line)
        # `:= by` often sits on an indented signature-continuation line;
        # the body is indented relative to the DECL header, so use the
        # nearest preceding decl line's indent if it is shallower.
        for k in range(i, max(i - 10, -1), -1):
            if DECL_RE.match(run[k]):
                ind = min(ind, indent_of(run[k]))
                break
        n = 0
        for k in range(i + 1, len(run)):
            if not run[k].strip():
                continue
            if indent_of(run[k]) <= ind:
                break
            n += 1
        lengths.append(n)
    return lengths


def compute_metrics(runs: list[list[str]]) -> dict:
    out = dict(n_added_lines=0, v02_n_long=0, v04_lambda=0, v05_erw=0,
               v06_nonterminal_simp=0, v07_unsqueezed_simp=0, n_simp=0,
               v10_missing_doc=0, n_tld=0, v13_focus_dots=0,
               v16_n_decls=0, v17_deprecated_nodate=0,
               n_tactic_mode=0, n_term_mode=0)
    block_lengths: list[int] = []
    for run in runs:
        mask = comment_mask(run)
        out["n_added_lines"] += len(run)
        for i, line in enumerate(run):
            if len(line) > 100:
                out["v02_n_long"] += 1
            if mask[i]:
                continue
            code = line.split("--")[0]
            out["v04_lambda"] += code.count("λ") + len(FUN_ARROW_RE.findall(code))
            out["v05_erw"] += len(ERW_RE.findall(code))
            if FOCUS_DOT_RE.match(line):
                out["v13_focus_dots"] += 1
            if "@[deprecated" in code and "since" not in code:
                out["v17_deprecated_nodate"] += 1
        ns, nu, nn = simp_metrics(run, mask)
        out["n_simp"] += ns
        out["v07_unsqueezed_simp"] += nu
        out["v06_nonterminal_simp"] += nn
        nd, ntld, nmiss, ntac, nterm = decl_metrics(run, mask)
        out["v16_n_decls"] += nd
        out["n_tld"] += ntld
        out["v10_missing_doc"] += nmiss
        out["n_tactic_mode"] += ntac
        out["n_term_mode"] += nterm
        block_lengths += proof_blocks(run, mask)
    out["v15_n_blocks"] = len(block_lengths)
    out["v15_mean_block_len"] = (float(np.mean(block_lengths))
                                 if block_lengths else np.nan)
    out["v15_max_block_len"] = (int(np.max(block_lengths))
                                if block_lengths else 0)
    nm = out["n_tactic_mode"] + out["n_term_mode"]
    out["v15_tactic_ratio"] = out["n_tactic_mode"] / nm if nm else np.nan
    return out

# ---------------------------------------------------------------- analysis

RATE_FEATURES = [  # (column, how)
    ("v02_frac_long", None),
    ("v04_lambda", "per100"), ("v05_erw", "per100"),
    ("v06_nonterminal_simp", "per100"), ("v07_unsqueezed_simp", "per100"),
    ("v10_frac_missing_doc", None), ("v13_focus_dots", "per100"),
    ("v15_mean_block_len", None), ("v15_max_block_len", None),
    ("v15_tactic_ratio", None), ("v16_n_decls", "per100"),
    ("v17_deprecated_nodate", "per100"), ("log_added_lines", None),
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    denom = df.n_added_lines.clip(lower=1)
    f["v02_frac_long"] = df.v02_n_long / denom
    f["v10_frac_missing_doc"] = (df.v10_missing_doc
                                 / df.n_tld.clip(lower=1)).where(df.n_tld > 0, 0.0)
    f["v15_mean_block_len"] = df.v15_mean_block_len.fillna(0)
    f["v15_max_block_len"] = df.v15_max_block_len
    f["v15_tactic_ratio"] = df.v15_tactic_ratio.fillna(0.5)
    f["log_added_lines"] = np.log1p(df.n_added_lines)
    for col, how in RATE_FEATURES:
        if how == "per100":
            f[col] = 100.0 * df[col] / denom
    return f


def analyze(df: pd.DataFrame) -> str:
    from scipy.stats import pointbiserialr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = build_features(df)
    y = df.y.values
    lines = ["| feature | point-biserial r | p | AUC |",
             "|---|---|---|---|"]
    for col, _ in RATE_FEATURES:
        x = feats[col].values
        if np.std(x) == 0:
            lines.append(f"| {col} | (constant) | — | 0.500 |")
            continue
        r, p = pointbiserialr(y, x)
        auc = roc_auc_score(y, x)
        lines.append(f"| {col} | {r:+.3f} | {p:.3f} | {auc:.3f} |")
    X = feats[[c for c, _ in RATE_FEATURES]].values
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0))
    n_splits = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    lines.append("")
    lines.append(f"Combined logistic regression 5-fold CV AUC: "
                 f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
                 f"(folds: {', '.join(f'{a:.3f}' for a in aucs)})")
    return "\n".join(lines)

# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(
        BASE, "friction_dataset_v2.csv.gz"))
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(
        BASE, "firstpush_pilot_metrics.csv"))
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="process the entire dataset (all splits), no sampling")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)
    if args.all:
        sample = df.reset_index(drop=True)
    else:
        train = df[df.split == "train"]
        sample = pd.concat([
            train[train.y == 1].sample(
                min(args.n_per_class, int(train.y.sum())),
                random_state=args.seed),
            train[train.y == 0].sample(
                min(args.n_per_class, int((1 - train.y).sum())),
                random_state=args.seed),
        ]).reset_index(drop=True)
    print(f"sampled {len(sample)} PRs "
          f"(y=1: {int(sample.y.sum())}, y=0: {int((1-sample.y).sum())})",
          file=sys.stderr)

    numbers = sample.number.tolist()
    if args.skip_fetch:
        fetched = {n: True for n in numbers}
    else:
        fetched = batch_fetch(numbers)

    rows = []
    for idx, row in sample.iterrows():
        n = int(row.number)
        rec = dict(number=n, y=int(row.y),
                   fetch_ok=bool(fetched.get(n, False)),
                   state_used=None, n_added_lines=np.nan)
        rev, state = resolve_state(n, row.first_commit_oid, fetched.get(n, False))
        if rev is None:
            rec["state_used"] = "unresolved"
            rows.append(rec)
            print(f"  [{idx+1}/{len(sample)}] PR {n}: UNRESOLVED",
                  file=sys.stderr)
            continue
        rec["state_used"] = state
        runs = added_line_runs(rev)
        if runs is None:
            rec["state_used"] = state + "_diff_failed"
            rows.append(rec)
            continue
        rec.update(compute_metrics(runs))
        rows.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(sample)}] done", file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)", file=sys.stderr)

    n_fetch_ok = int(out.fetch_ok.sum())
    states = out.state_used.value_counts().to_dict()
    print("\n## Fetch / state summary")
    print(json.dumps({"n_sampled": len(out), "fetch_ok": n_fetch_ok,
                      "states": states}, indent=2))

    usable = out[out.state_used.isin(["first_commit", "head_fallback"])]
    usable = usable[usable.n_added_lines.notna()]
    print(f"\nusable rows for analysis: {len(usable)} "
          f"(y=1: {int(usable.y.sum())})")
    print(f"rows with 0 added .lean lines: "
          f"{int((usable.n_added_lines == 0).sum())}")
    print("\n## Per-metric analysis (rates per 100 added lines "
          "unless noted)\n")
    print(analyze(usable))


if __name__ == "__main__":
    main()
