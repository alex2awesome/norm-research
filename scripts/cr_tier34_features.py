"""Tier 3 (linter-based complexity) + Tier 4a/b (test-signal text features).

For each v2 datapoint:
  Tier 3 (run lizard on extracted code from diff hunks):
    - avg_ccn, max_ccn          : cyclomatic complexity
    - avg_fn_len, max_fn_len    : function NLOC
    - total_nloc                : total non-comment LOC of changed code
    - n_functions               : function count in changed code
  Tier 4a (test code patterns added):
    - n_test_functions_added    : def test_*, function test*, it(...), @Test
    - n_assertions_added        : assert / expect / should / assertEquals etc.
    - n_test_fixtures_added     : @pytest.fixture, beforeEach, setUp, @Before
  Tier 4b (review-comment test discussion):
    - n_test_pass_mentions      : "tests pass", "CI green", etc.
    - n_test_fail_mentions      : "tests fail", "regression", "CI failing"
    - n_needs_test_mentions     : "needs test", "missing test", "add test"

Then trains RF combining Tier 1+2 (existing) with these.
"""
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
SEED = 42
TIER12 = REPO / "outputs/v2_analysis/cr_tier12_features.parquet"


EXT_BY_LANG = {
    ".py": "py", ".js": "js", ".ts": "ts", ".jsx": "jsx", ".tsx": "tsx",
    ".java": "java", ".go": "go", ".rs": "rs", ".rb": "rb", ".c": "c",
    ".cpp": "cpp", ".cc": "cpp", ".h": "h", ".hpp": "h", ".cs": "cs",
    ".kt": "kt", ".scala": "scala", ".php": "php", ".swift": "swift",
}

TEST_FN_RE = re.compile(
    r"^\+\s*(?:async\s+)?(?:def\s+test_\w+|function\s+test\w*|"
    r"@Test\s*$|it\s*\(['\"]|describe\s*\(['\"]|test\s*\(['\"])",
    re.MULTILINE,
)
ASSERT_RE = re.compile(
    r"^\+\s*(?:assert\b|expect\s*\(|self\.assert\w*|"
    r"chai\.assert|chai\.expect|should\.(?:equal|be)|"
    r"assertThat|assertEqual|assertTrue|assertFalse|assertRaises)",
    re.MULTILINE,
)
FIXTURE_RE = re.compile(
    r"^\+\s*(?:@pytest\.fixture|@fixture\b|beforeEach\s*\(|"
    r"afterEach\s*\(|setUp\s*\(|tearDown\s*\(|@Before\s*$|"
    r"before\s*\(|after\s*\()",
    re.MULTILINE,
)

# Tier 4b patterns (case-insensitive, applied to review-comment text)
TEST_PASS_RE = re.compile(
    r"\b(tests?\s+pass(?:ing|es|ed)?|all\s+tests?\s+pass|"
    r"ci\s+(?:is\s+)?green|build\s+(?:is\s+)?passing|"
    r"all\s+green|tests?\s+(?:are\s+)?fine)\b", re.IGNORECASE)
TEST_FAIL_RE = re.compile(
    r"\b(tests?\s+(?:are\s+)?fail(?:ing|s|ed)?|test\s+failure|"
    r"breaks?\s+(?:the\s+|existing\s+)?tests?|"
    r"ci\s+(?:is\s+)?(?:failing|red|broken)|"
    r"regression|broke\s+(?:the\s+)?build)\b", re.IGNORECASE)
NEEDS_TEST_RE = re.compile(
    r"\b(needs?\s+(?:a\s+)?test|missing\s+tests?|no\s+tests?\s+for|"
    r"add\s+(?:some\s+)?tests?|please\s+(?:add\s+)?tests?|"
    r"test\s+coverage|should\s+have\s+tests?|where(?:'s|\s+is)\s+the\s+test)\b",
    re.IGNORECASE)


def parse_diff_added_code(text: str):
    """Walk through diff hunks; return {ext: [added_line, ...]} for code we can lint."""
    out = {}
    current_ext = None
    for line in text.split("\n"):
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(\S+) b/(\S+)", line)
            if not m:
                current_ext = None
                continue
            path = m.group(2).lower()
            ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
            current_ext = EXT_BY_LANG.get(ext)
        elif current_ext and line.startswith("+") and not line.startswith("+++"):
            content = line[1:]
            out.setdefault(current_ext, []).append(content)
    return out


def run_lizard_aggregated(added_by_ext: dict) -> dict:
    """Write extracted code to temp files, run lizard, aggregate."""
    out = {
        "tier3_avg_ccn": 0.0, "tier3_max_ccn": 0.0,
        "tier3_avg_fn_len": 0.0, "tier3_max_fn_len": 0.0,
        "tier3_total_nloc": 0.0, "tier3_n_functions": 0.0,
    }
    if not added_by_ext:
        return out

    ccns, fn_lens = [], []
    total_nloc = 0
    n_fns = 0

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for ext_short, lines in added_by_ext.items():
            if not lines:
                continue
            ext_dot = next((k for k, v in EXT_BY_LANG.items() if v == ext_short), None)
            if ext_dot is None:
                continue
            (td_p / f"snippet{ext_dot}").write_text("\n".join(lines))

        files = list(td_p.iterdir())
        if not files:
            return out

        try:
            res = subprocess.run(
                ["/lfs/skampere3/0/alexspan/miniconda3/bin/lizard",
                 "--CCN", "1", "-l", "py", "-l", "java", "-l", "go",
                 "-l", "js", "-l", "cpp", "-l", "rb",
                 "--csv", *[str(f) for f in files]],
                capture_output=True, text=True, timeout=15,
            )
            for line in res.stdout.splitlines():
                # NLOC,CCN,token,param,length,location,file,func,longname,start,end
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                try:
                    nloc = int(parts[0])
                    ccn = int(parts[1])
                    length = int(parts[4])
                except (ValueError, IndexError):
                    continue
                ccns.append(ccn)
                fn_lens.append(length)
                total_nloc += nloc
                n_fns += 1
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if ccns:
        out["tier3_avg_ccn"] = float(np.mean(ccns))
        out["tier3_max_ccn"] = float(np.max(ccns))
    if fn_lens:
        out["tier3_avg_fn_len"] = float(np.mean(fn_lens))
        out["tier3_max_fn_len"] = float(np.max(fn_lens))
    out["tier3_total_nloc"] = float(total_nloc)
    out["tier3_n_functions"] = float(n_fns)
    return out


def extract_diff_added_lines(text: str) -> str:
    """All `+` lines concatenated, for Tier 4a regex matching."""
    parts = []
    for line in text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            parts.append(line)
    return "\n".join(parts)


def tier4a_test_features(text: str) -> dict:
    added = extract_diff_added_lines(text)
    return {
        "tier4a_test_fns_added": float(len(TEST_FN_RE.findall(added))),
        "tier4a_assertions_added": float(len(ASSERT_RE.findall(added))),
        "tier4a_fixtures_added": float(len(FIXTURE_RE.findall(added))),
    }


def tier4b_review_features(v2_text: str) -> dict:
    """v2_text is the thin artifact (PR TITLE + REVIEW COMMENTS).
       Reviewers' test-discussion patterns."""
    # Strip the title line, keep the comments
    body = re.sub(r"^PR TITLE:.*?\n", "", v2_text, count=1)
    return {
        "tier4b_pass_mentions": float(len(TEST_PASS_RE.findall(body))),
        "tier4b_fail_mentions": float(len(TEST_FAIL_RE.findall(body))),
        "tier4b_needs_test_mentions": float(len(NEEDS_TEST_RE.findall(body))),
    }


def main():
    print("Loading v2 datapoints + dense source...")
    dps = json.loads((REPO / "runs/validity_full/v2/code_review/datapoints.json").read_text())
    dps = [d for d in dps if d.get("judgement") is not None and d.get("text")]

    # We need dense_4096tok text (has the diff) for Tier 3 + 4a
    # And v2 task text (has the comments) for Tier 4b
    v2 = pd.DataFrame([{
        "datapoint_id": d["datapoint_id"],
        "v2_text": d["text"],
        "title": (re.match(r"PR TITLE: ([^\n]+)", d["text"]) or [None, None])[1],
        "y": int(d["judgement"]),
    } for d in dps]).dropna(subset=["title"])

    dense = pd.read_csv(
        REPO / "datasets/code-review/code_review_dense_4096tok.csv.gz",
        usecols=["text"])
    dense["title"] = dense["text"].str.extract(r"## PR Title\s*(.+?)(?:\n|$)", expand=False)
    j = v2.merge(dense.drop_duplicates("title", keep="first"),
                 on="title", how="left").dropna(subset=["text"])
    print(f"joined: {len(j)}")

    print("Computing Tier 3 + Tier 4a/b features...")
    feats = []
    for i, row in enumerate(j.itertuples()):
        added = parse_diff_added_code(row.text)
        d3 = run_lizard_aggregated(added)
        d4a = tier4a_test_features(row.text)
        d4b = tier4b_review_features(row.v2_text)
        feats.append({"datapoint_id": row.datapoint_id,
                      "y": row.y, **d3, **d4a, **d4b})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(j)} done")

    new = pd.DataFrame(feats)
    out_p = REPO / "outputs/v2_analysis/cr_tier34_features.parquet"
    new.to_parquet(out_p)
    print(f"wrote {out_p}")

    print("\nFeature distributions:")
    for c in [col for col in new.columns if col not in ("datapoint_id", "y")]:
        s = new[c]
        print(f"  {c:<30} mean={s.mean():.2f} max={s.max():.1f} nonzero={(s > 0).mean() * 100:.0f}%")

    # Combine with Tier 1+2
    print("\nCombining with Tier 1+2...")
    t12 = pd.read_parquet(TIER12)
    combo = t12.merge(new.drop(columns=["y"]), on="datapoint_id", how="inner")
    feat_cols = [c for c in combo.columns if c not in ("datapoint_id", "y")]
    X = combo[feat_cols].values
    y = combo["y"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    acc = accuracy_score(yte, (p > 0.5).astype(int))

    print()
    print("=" * 70)
    print(f"TIER 1+2+3+4ab combined : {len(feat_cols)} features")
    print(f"  RF AUC = {auc:.3f}  acc = {acc:.1%}")
    print()
    print("Top 12 features by importance:")
    for n, i in sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1])[:12]:
        print(f"  {n:<30} {i:.4f}")


if __name__ == "__main__":
    main()
