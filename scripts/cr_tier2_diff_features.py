"""Tier 2: features derived from parsing the diff text in dense_4096tok.

No external tools. Just regex/structural parsing of diff hunks.

Features per PR (in addition to Tier 1):
  - num_new_files
  - num_deleted_files
  - num_modified_files
  - test_file_count, test_line_changes
  - doc_file_count, doc_line_changes
  - config_file_count
  - generated_file_count (lockfiles, autogen)
  - max_file_size_change_loc
  - todo_added, todo_removed
  - import_added, import_removed (rough)
  - comment_loc_added, code_loc_added (rough)
  - hunk_count
  - test_to_source_ratio
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42

TEST_PATTERNS = re.compile(
    r"(^|/)(test_|test/|tests/|spec/|specs/|.*_test\.py|.*\.test\.|.*_spec\.|__tests__/)",
    re.IGNORECASE,
)
DOC_EXTS = {".md", ".rst", ".txt", ".adoc", ".tex", ".rtf"}
CONFIG_EXTS = {
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".dockerfile", ".dockerignore", ".gitignore", ".editorconfig", ".gitattributes",
}
GENERATED_HINTS = re.compile(
    r"(package-lock\.json|yarn\.lock|poetry\.lock|Pipfile\.lock|Cargo\.lock|"
    r"go\.sum|composer\.lock|gradle\.lockfile|generated|autogen|"
    r"\.pb\.go$|\.pb\.py$|\.g\.dart$|protoc|generated_)",
    re.IGNORECASE,
)
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
IMPORT_RE = re.compile(r"^(import\s|from\s.+\simport|#include|require\(|use\s|using\s)")
COMMENT_HINT_RE = re.compile(r"^\s*(#|//|/\*|\*|--)")


def parse_diff_features(text: str) -> dict:
    """Walk through the unified diff portion of the text and aggregate features."""
    f = {
        "num_new_files": 0,
        "num_deleted_files": 0,
        "num_modified_files": 0,
        "test_file_count": 0,
        "test_added": 0,
        "test_removed": 0,
        "doc_file_count": 0,
        "doc_added": 0,
        "config_file_count": 0,
        "generated_file_count": 0,
        "max_file_loc_added": 0,
        "max_file_loc_removed": 0,
        "todo_added": 0,
        "todo_removed": 0,
        "import_added": 0,
        "import_removed": 0,
        "comment_loc_added": 0,
        "code_loc_added": 0,
        "hunk_count": 0,
        "file_count_with_changes": 0,
        "added_lines_total": 0,
        "removed_lines_total": 0,
    }

    current_file = None
    is_new = False
    is_deleted = False
    file_added = 0
    file_removed = 0
    file_type_test = file_type_doc = file_type_config = file_type_generated = False

    def flush():
        nonlocal current_file, is_new, is_deleted, file_added, file_removed
        nonlocal file_type_test, file_type_doc, file_type_config, file_type_generated
        if current_file is None:
            return
        f["file_count_with_changes"] += 1
        if is_new:
            f["num_new_files"] += 1
        elif is_deleted:
            f["num_deleted_files"] += 1
        else:
            f["num_modified_files"] += 1
        if file_type_test:
            f["test_file_count"] += 1
            f["test_added"] += file_added
            f["test_removed"] += file_removed
        if file_type_doc:
            f["doc_file_count"] += 1
            f["doc_added"] += file_added
        if file_type_config:
            f["config_file_count"] += 1
        if file_type_generated:
            f["generated_file_count"] += 1
        if file_added > f["max_file_loc_added"]:
            f["max_file_loc_added"] = file_added
        if file_removed > f["max_file_loc_removed"]:
            f["max_file_loc_removed"] = file_removed
        # reset
        current_file = None
        is_new = is_deleted = False
        file_added = file_removed = 0
        file_type_test = file_type_doc = file_type_config = file_type_generated = False

    for line in text.split("\n"):
        if line.startswith("diff --git"):
            flush()
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            current_file = m.group(2) if m else None
            if current_file:
                file_type_test = bool(TEST_PATTERNS.search(current_file))
                low = current_file.lower()
                ext = "." + low.rsplit(".", 1)[-1] if "." in low else ""
                file_type_doc = ext in DOC_EXTS
                file_type_config = ext in CONFIG_EXTS
                file_type_generated = bool(GENERATED_HINTS.search(current_file))
        elif line.startswith("new file"):
            is_new = True
        elif line.startswith("deleted file"):
            is_deleted = True
        elif line.startswith("@@"):
            f["hunk_count"] += 1
        elif line.startswith("+") and not line.startswith("+++"):
            content = line[1:]
            file_added += 1
            f["added_lines_total"] += 1
            if TODO_RE.search(content):
                f["todo_added"] += 1
            if IMPORT_RE.match(content.lstrip()):
                f["import_added"] += 1
            if COMMENT_HINT_RE.match(content):
                f["comment_loc_added"] += 1
            else:
                f["code_loc_added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            content = line[1:]
            file_removed += 1
            f["removed_lines_total"] += 1
            if TODO_RE.search(content):
                f["todo_removed"] += 1
            if IMPORT_RE.match(content.lstrip()):
                f["import_removed"] += 1
    flush()

    # Derived
    src_added = max(f["code_loc_added"] - f["test_added"], 0)
    f["test_to_source_ratio"] = (f["test_added"] / src_added) if src_added > 0 else 0.0
    f["comment_density_added"] = (
        f["comment_loc_added"] / max(f["added_lines_total"], 1)
    )
    f["generated_file_ratio"] = (
        f["generated_file_count"] / max(f["file_count_with_changes"], 1)
    )

    return f


def main():
    print("Loading v2 datapoints...")
    dps = json.loads((REPO / "runs/validity_full/v2/code_review/datapoints.json").read_text())
    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "y": d.get("judgement"),
        "title": (re.match(r"PR TITLE: ([^\n]+)", d.get("text", "")) or [None, None])[1],
    } for d in dps if d.get("judgement") is not None]).dropna(subset=["title"])

    print("Loading dense_4096tok...")
    dense = pd.read_csv(
        REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz",
        usecols=["text", "judgement", "language", "num_files", "num_comments",
                 "pr_additions", "pr_deletions"],
    )
    dense["title"] = dense["text"].str.extract(r"## PR Title\s*(.+?)(?:\n|$)", expand=False)

    print("Joining v2 → dense by title...")
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left")
    j = j.dropna(subset=["text"])
    print(f"joined rows: {len(j)}")

    print("Parsing diffs (Tier 2 features)...")
    feats = []
    for i, row in enumerate(j.itertuples()):
        d = parse_diff_features(row.text)
        d["datapoint_id"] = row.datapoint_id
        feats.append(d)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(j)} done")
    tier2 = pd.DataFrame(feats)
    print(f"tier2 features: {len(tier2)} rows, {tier2.shape[1] - 1} feature columns")

    # Combine with Tier 1
    j_meta = j[["datapoint_id", "y", "num_files", "num_comments",
                "pr_additions", "pr_deletions", "language"]].copy()
    combo = j_meta.merge(tier2, on="datapoint_id")
    # language one-hot
    lang = combo["language"].fillna("unknown").str.lower()
    top_langs = lang.value_counts().head(6).index.tolist()
    for L in top_langs:
        combo[f"lang_{L}"] = (lang == L).astype(int)
    combo = combo.drop(columns=["language"])
    # Drop nulls
    combo = combo.dropna()

    feat_cols = [c for c in combo.columns if c not in ("datapoint_id", "y")]
    X = combo[feat_cols].values
    y = combo["y"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]
    auc_rf = roc_auc_score(yte, p_rf)
    acc_rf = accuracy_score(yte, (p_rf > 0.5).astype(int))

    sc = StandardScaler()
    lr = LogisticRegression(
        penalty="l2", C=0.5, class_weight="balanced",
        max_iter=2000, solver="lbfgs")
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)
    lr.fit(Xtr_s, ytr)
    p_lr = lr.predict_proba(Xte_s)[:, 1]
    auc_lr = roc_auc_score(yte, p_lr)

    print()
    print("=" * 60)
    print("TIER 1 + TIER 2 (diff-derived features)")
    print("=" * 60)
    print(f"  features: {len(feat_cols)}")
    print(f"  n_train={len(ytr)}  n_test={len(yte)}")
    print(f"  RF AUC={auc_rf:.3f}  acc={acc_rf:.1%}")
    print(f"  LR AUC={auc_lr:.3f}")
    print()
    print("RF top 15 features by importance:")
    imps = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1])[:15]
    for n, i in imps:
        print(f"  {n:<28} {i:.4f}")

    out_p = REPO / "outputs/v2_analysis/cr_tier12_features.parquet"
    combo.to_parquet(out_p)
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
