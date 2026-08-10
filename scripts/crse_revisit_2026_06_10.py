"""Revisit CR.SE as a quantitative code-quality cell (2026-06-10).

Two framings:
  (A) ANSWER-side: within-question pairs of (accepted answer, non-accepted answer
      posted within ±7 days). Predict accepted_vs_other from metrics on the
      ANSWER artifact (prose+code-block — articulability framing).
  (B) QUESTION-side: question_high_score vs question_low_score within
      top-tag groups (controls topic popularity). Predict from metrics on
      QUESTION code (submitted code — code-quality framing).

For each:
  - Sample ~3-5K units.
  - Score the full coded metric bank (existing_metrics_runner.coded.metrics).
  - Grouped 5-fold (group = question_id for A; group = top_tag for B).
  - LR and RF, both report mean AUC ± std.

Outputs:
  outputs/v2_analysis/crse_revisit_2026_06_10.json
  outputs/v2_analysis/crse_revisit_2026_06_10.md

Run on sk3 from /lfs/skampere3/0/alexspan/norm-research.
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
sys.path.insert(0, str(REPO))

POSTS = REPO / "datasets/codereview_se/posts.parquet"
OUT_DIR = REPO / "outputs/v2_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "crse_revisit_2026_06_10.json"
OUT_MD = OUT_DIR / "crse_revisit_2026_06_10.md"

RNG = np.random.RandomState(0)

# ---------------------------------------------------------------------------
# HTML / code extraction
# ---------------------------------------------------------------------------
PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")

LANG_PRIORITY = [
    "python", "py", "cpp", "c++", "c", "java", "javascript", "js",
    "typescript", "ts", "go", "ruby", "rb", "rust", "rs", "csharp", "c#",
    "cs", "php", "kotlin", "kt", "swift", "scala",
]
LANG_TO_EXT = {
    "python": "py", "py": "py",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp",
    "c": "c",
    "java": "java",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "go": "go",
    "ruby": "rb", "rb": "rb",
    "rust": "rs", "rs": "rs",
    "csharp": "cs", "c#": "cs", "cs": "cs",
    "php": "php",
    "kotlin": "kt", "kt": "kt",
    "swift": "swift",
    "scala": "scala",
}


def extract_blocks(body_html: str) -> tuple[str, str]:
    """Return (code_concat, prose). Concat all <pre><code> blocks for code."""
    if not body_html:
        return "", ""
    code_chunks = []
    # multi-line <pre> blocks
    pre_blocks = PRE_RE.findall(body_html)
    body_no_pre = PRE_RE.sub(" ", body_html)
    for pb in pre_blocks:
        # strip nested <code>...</code>
        m = CODE_RE.search(pb)
        inner = m.group(1) if m else pb
        inner = html.unescape(TAG_RE.sub("", inner))
        code_chunks.append(inner)
    prose_html = body_no_pre
    # remove inline <code>
    prose_html = CODE_RE.sub(" ", prose_html)
    prose = html.unescape(TAG_RE.sub(" ", prose_html))
    prose = re.sub(r"\s+", " ", prose).strip()
    return "\n\n".join(code_chunks).strip(), prose


def split_tags(tags_str: str) -> list[str]:
    if not tags_str:
        return []
    return [t for t in tags_str.split("|") if t]


def pick_lang(tags: list[str]) -> str:
    for t in tags:
        if t in LANG_TO_EXT:
            return t
    return "unknown"


# ---------------------------------------------------------------------------
# Build the two framings
# ---------------------------------------------------------------------------

def build_framing_A(posts: pd.DataFrame, n_target: int = 4000,
                    window_days: int = 7) -> pd.DataFrame:
    """ANSWER-side: within-Q, accepted answer (pos) vs same-day other
    answer (neg). One pair per question."""
    q = posts[posts.PostTypeId == 1][
        ["Id", "Title", "Tags", "AcceptedAnswerId", "Score"]
    ].rename(columns={"Id": "q_id", "Score": "q_score"})
    q = q[q.AcceptedAnswerId > 0].copy()
    a = posts[posts.PostTypeId == 2][
        ["Id", "ParentId", "Score", "Body", "CreationDate"]
    ].rename(columns={"Id": "a_id", "ParentId": "q_id",
                      "Score": "a_score", "Body": "a_body",
                      "CreationDate": "a_creation"})
    # Date
    a["a_day"] = pd.to_datetime(a["a_creation"], errors="coerce").dt.date

    # accepted set
    acc_set = set(q["AcceptedAnswerId"].tolist())
    a["is_accepted"] = a["a_id"].isin(acc_set).astype(int)

    # Join Q meta onto A
    merged = a.merge(q[["q_id", "Title", "Tags", "AcceptedAnswerId",
                        "q_score"]], on="q_id", how="inner")

    # For each Q: pick accepted answer + sample one non-accepted within
    # ±window_days of accepted answer.
    rows = []
    rng = np.random.RandomState(0)
    # Group by question
    by_q = merged.groupby("q_id", sort=False)
    qids = list(by_q.groups.keys())
    rng.shuffle(qids)
    for qid in qids:
        sub = by_q.get_group(qid)
        accepted = sub[sub.is_accepted == 1]
        if len(accepted) == 0:
            continue
        acc_row = accepted.iloc[0]
        acc_day = acc_row["a_day"]
        if acc_day is None:
            continue
        others = sub[sub.is_accepted == 0]
        if len(others) == 0:
            continue
        # within window
        diff = others["a_day"].apply(
            lambda d: abs((d - acc_day).days) if d is not None else 1e9)
        same_day = others[diff <= window_days]
        if len(same_day) == 0:
            continue
        # pick one at random
        neg = same_day.sample(n=1, random_state=int(qid % (2**31)))
        rows.append({
            "q_id": qid,
            "a_id": int(acc_row["a_id"]),
            "label": 1,
            "a_score": int(acc_row["a_score"]),
            "a_body": acc_row["a_body"],
            "Title": acc_row["Title"],
            "Tags": acc_row["Tags"],
            "q_score": int(acc_row["q_score"]),
        })
        nrow = neg.iloc[0]
        rows.append({
            "q_id": qid,
            "a_id": int(nrow["a_id"]),
            "label": 0,
            "a_score": int(nrow["a_score"]),
            "a_body": nrow["a_body"],
            "Title": nrow["Title"],
            "Tags": nrow["Tags"],
            "q_score": int(nrow["q_score"]),
        })
        if len(rows) >= n_target * 2:
            break

    df = pd.DataFrame(rows)
    # Extract code + prose, tags, lang
    code_prose = df["a_body"].apply(extract_blocks)
    df["code"] = code_prose.apply(lambda x: x[0])
    df["prose"] = code_prose.apply(lambda x: x[1])
    df["tag_list"] = df["Tags"].apply(split_tags)
    df["lang"] = df["tag_list"].apply(pick_lang)
    df["artifact_text"] = df.apply(
        lambda r: (r["code"] + "\n\n# PROSE:\n" + r["prose"])
        if r["code"] else r["prose"], axis=1)
    df = df[df["artifact_text"].str.len() > 10].copy()
    df = df.reset_index(drop=True)
    return df


def build_framing_B(posts: pd.DataFrame, n_target: int = 4000) -> pd.DataFrame:
    """QUESTION-side: top-tag matched, high-score vs low-score question.

    Within each top-100 single-language tag, split questions into top vs
    bottom quartile by Score. Predict from question CODE."""
    q = posts[posts.PostTypeId == 1].copy()
    q["tag_list"] = q["Tags"].apply(split_tags)
    q["lang"] = q["tag_list"].apply(pick_lang)
    # keep only langs we have an extension for
    q = q[q["lang"] != "unknown"].copy()

    code_prose = q["Body"].apply(extract_blocks)
    q["q_code"] = code_prose.apply(lambda x: x[0])
    q["q_prose"] = code_prose.apply(lambda x: x[1])
    q = q[q["q_code"].str.len() > 20].copy()

    # Per-lang top/bottom quartile
    rows = []
    rng = np.random.RandomState(0)
    for lang, sub in q.groupby("lang"):
        if len(sub) < 200:
            continue
        scores = sub["Score"].astype(int)
        q_lo, q_hi = scores.quantile(0.25), scores.quantile(0.75)
        lo = sub[sub["Score"] <= q_lo]
        hi = sub[sub["Score"] >= q_hi]
        n_each = min(len(lo), len(hi), 600)
        if n_each < 30:
            continue
        lo_s = lo.sample(n=n_each, random_state=0)
        hi_s = hi.sample(n=n_each, random_state=0)
        for _, r in lo_s.iterrows():
            rows.append({
                "q_id": int(r["Id"]),
                "label": 0,
                "lang": lang,
                "Score": int(r["Score"]),
                "code": r["q_code"],
                "prose": r["q_prose"],
                "Title": r["Title"],
                "Tags": r["Tags"],
            })
        for _, r in hi_s.iterrows():
            rows.append({
                "q_id": int(r["Id"]),
                "label": 1,
                "lang": lang,
                "Score": int(r["Score"]),
                "code": r["q_code"],
                "prose": r["q_prose"],
                "Title": r["Title"],
                "Tags": r["Tags"],
            })
    df = pd.DataFrame(rows)
    if len(df) > n_target * 2:
        df = df.sample(n=n_target * 2, random_state=0).reset_index(drop=True)
    df["artifact_text"] = df["code"]
    return df


# ---------------------------------------------------------------------------
# Metric bank scoring
# ---------------------------------------------------------------------------

def to_synthetic_diff(file_path: str, file_text: str) -> str:
    if not file_text:
        return ""
    lines = file_text.split("\n")
    body = "\n".join("+" + ln for ln in lines)
    n = len(lines)
    return (
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- /dev/null\n"
        f"+++ b/{file_path}\n"
        f"@@ -0,0 +1,{n} @@\n"
        f"{body}\n"
    )


MAX_ARTIFACT_CHARS = 20_000   # CR.SE answers can embed huge code dumps
PER_CALL_TIMEOUT_S = 5        # SIGALRM guard against regex blowup


class _CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CallTimeout()


def _score_chunk(rows):
    import signal
    sys.path.insert(0, str(REPO))
    from methods.existing_metrics_runner.coded.metrics import load_all
    metrics = load_all()
    signal.signal(signal.SIGALRM, _alarm_handler)

    def _guarded(fn, arg):
        signal.alarm(PER_CALL_TIMEOUT_S)
        try:
            return fn(arg)
        finally:
            signal.alarm(0)

    out = []
    for r in rows:
        ext = LANG_TO_EXT.get(str(r["lang"]).lower(), "txt")
        fp = f"candidate.{ext}"
        text = (r["artifact_text"] or "")[:MAX_ARTIFACT_CHARS]
        diff = to_synthetic_diff(fp, text)
        rec = {
            "row_id": r["row_id"],
            "label": r["label"],
            "group": r["group"],
        }
        for m in metrics:
            applied = 0
            score = float("nan")
            if diff:
                try:
                    a = _guarded(m.applies, diff)
                except (Exception, _CallTimeout):
                    a = False
                applied = int(bool(a))
                if applied:
                    try:
                        s = _guarded(m.score, diff)
                        if s is not None and not (
                            isinstance(s, float) and math.isnan(s)
                        ):
                            score = float(s)
                    except (Exception, _CallTimeout):
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def score_bank(df: pd.DataFrame, group_col: str, n_workers: int = 8
               ) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df["row_id"] = df.index
    df["group"] = df[group_col]
    rows = df[["row_id", "label", "group", "lang", "artifact_text"]
              ].to_dict(orient="records")
    chunk_size = max(25, len(rows) // (n_workers * 10))
    chunks = [rows[i:i + chunk_size]
              for i in range(0, len(rows), chunk_size)]
    print(f"  scoring {len(rows)} rows, {len(chunks)} chunks, "
          f"{n_workers} workers", flush=True)
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_score_chunk, c) for c in chunks]
        for k, f in enumerate(as_completed(futs)):
            results.extend(f.result())
            print(f"    chunk {k+1}/{len(chunks)} done "
                  f"(elapsed={time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# AUC eval
# ---------------------------------------------------------------------------

def eval_grouped(scored: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    score_cols = sorted(
        c for c in scored.columns
        if c.endswith("_score") and not c.startswith("group")
    )
    # filter to columns with >=5% non-nan AND non-zero variance
    feat_cols = []
    for c in score_cols:
        v = scored[c]
        if v.notna().mean() < 0.05:
            continue
        nz = v[v.notna()]
        if len(nz) < 50 or nz.nunique() < 3:
            continue
        feat_cols.append(c)
    print(f"  feat_cols n_eligible={len(feat_cols)} of {len(score_cols)}",
          flush=True)

    X = scored[feat_cols].values
    y = scored["label"].values
    groups = scored["group"].values

    # Number of unique groups limits the number of folds
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    out = {"feat_cols_used": len(feat_cols), "n_rows": int(len(y)),
           "pos_rate": float(y.mean()), "n_groups": int(n_groups),
           "n_splits": n_splits}

    for name, clf_factory in [
        ("LR", lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, C=1.0,
                                       solver="liblinear")),
        ])),
        ("RF", lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=3,
                n_jobs=-1, random_state=0)),
        ])),
    ]:
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            clf = clf_factory()
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        out[f"auc_{name}_mean"] = float(np.mean(aucs))
        out[f"auc_{name}_std"] = float(np.std(aucs))
        out[f"auc_{name}_folds"] = [float(a) for a in aucs]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading posts.parquet ...", flush=True)
    posts = pd.read_parquet(POSTS)
    print(f"  posts: {len(posts)}", flush=True)

    print("\n==== Framing A: ANSWER-side (within-Q accepted vs same-day) ====",
          flush=True)
    dfA = build_framing_A(posts, n_target=2500, window_days=7)
    print(f"A: n={len(dfA)}  pos={dfA.label.mean():.3f}  "
          f"unique_q={dfA.q_id.nunique()}", flush=True)
    print(f"A lang dist: {dfA.lang.value_counts().head(10).to_dict()}",
          flush=True)
    print("Scoring metric bank on A ...", flush=True)
    scoredA = score_bank(dfA[["q_id", "label", "lang", "artifact_text"]
                              ].assign(group=dfA["q_id"]),
                         group_col="q_id", n_workers=8)
    resA = eval_grouped(scoredA)
    print(f"A results: {json.dumps({k: v for k, v in resA.items() if 'folds' not in k}, indent=2)}",
          flush=True)

    print("\n==== Framing B: QUESTION-side (per-lang top vs bottom quartile)"
          " ====", flush=True)
    dfB = build_framing_B(posts, n_target=2500)
    print(f"B: n={len(dfB)}  pos={dfB.label.mean():.3f}  "
          f"unique_lang={dfB.lang.nunique()}", flush=True)
    print(f"B lang dist: {dfB.lang.value_counts().to_dict()}", flush=True)
    # Group by question id (since each row is a question and we want
    # ungrouped 5-fold split; here groups are q_id giving us trivially many)
    # For "within-topic-tag matching", we want to control for lang in the
    # split — but a 5-fold on q_id with stratification by lang gives the
    # cleanest comparison. Use q_id as the group key so test sets are
    # disjoint at the question level (no leakage).
    print("Scoring metric bank on B ...", flush=True)
    scoredB = score_bank(dfB[["q_id", "label", "lang", "artifact_text"]
                              ].assign(group=dfB["q_id"]),
                         group_col="q_id", n_workers=8)
    resB = eval_grouped(scoredB)
    print(f"B results: {json.dumps({k: v for k, v in resB.items() if 'folds' not in k}, indent=2)}",
          flush=True)

    # Per-lang B sub-analysis (Python only; bank is Python-flavored per prior
    # finding)
    sub = scoredB.merge(dfB[["q_id", "lang"]].rename(columns={"q_id": "group"}),
                        on="group", how="left")
    by_lang = {}
    for lang in ["python", "cpp", "java", "javascript"]:
        sl = sub[sub.lang == lang]
        if len(sl) < 200 or sl.label.nunique() < 2:
            continue
        try:
            by_lang[lang] = eval_grouped(sl)
        except Exception as e:
            by_lang[lang] = {"error": str(e)}

    summary = {
        "date": "2026-06-10",
        "data": str(POSTS),
        "metric_bank_size": int(sum(
            1 for c in scoredA.columns if c.endswith("_score"))),
        "framing_A_answer_side": {
            "n_pairs": int(len(dfA)),
            "pos_rate": float(dfA.label.mean()),
            "unique_questions": int(dfA.q_id.nunique()),
            "lang_dist": dfA.lang.value_counts().head(10).to_dict(),
            "results": resA,
        },
        "framing_B_question_side": {
            "n_units": int(len(dfB)),
            "pos_rate": float(dfB.label.mean()),
            "lang_dist": dfB.lang.value_counts().to_dict(),
            "results": resB,
            "per_lang": by_lang,
        },
        "prior_numbers_for_comparison": {
            "initial_accepted_answer_prediction": 0.648,
            "combined_corpus_MI_ladder": 0.52,
            "comment": "RF-vs-LR boost on LC was +0.05; bank a400-a522 expanded since.",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Markdown report
    lines = []
    lines.append("# CR.SE Revisit — 2026-06-10\n")
    lines.append("Goal: test whether CR.SE merits re-entry into the "
                 "quantitative portfolio after (i) RF beats LR and "
                 "(ii) metric bank growth (a400-a522).\n")
    lines.append("## Framing A — ANSWER-side")
    a = summary["framing_A_answer_side"]
    lines.append(f"- Setup: within-question, ACCEPTED answer vs a "
                 f"non-accepted answer posted within ±7 days. Score "
                 f"answer code+prose with full bank. Grouped 5-fold "
                 f"by q_id.")
    lines.append(f"- n={a['n_pairs']} (pairs), pos_rate={a['pos_rate']:.3f}, "
                 f"unique_qs={a['unique_questions']}")
    lines.append(f"- top langs: {a['lang_dist']}")
    r = a["results"]
    lines.append(f"- LR AUC = {r['auc_LR_mean']:.3f} ± {r['auc_LR_std']:.3f}")
    lines.append(f"- RF AUC = {r['auc_RF_mean']:.3f} ± {r['auc_RF_std']:.3f}")
    lines.append(f"- features used: {r['feat_cols_used']}\n")

    lines.append("## Framing B — QUESTION-side")
    b = summary["framing_B_question_side"]
    lines.append(f"- Setup: per-language top-quartile vs bottom-quartile by "
                 f"question Score (within-language matching). Score QUESTION "
                 f"code with full bank. Grouped 5-fold by q_id.")
    lines.append(f"- n={b['n_units']}  pos_rate={b['pos_rate']:.3f}")
    lines.append(f"- lang dist: {b['lang_dist']}")
    r = b["results"]
    lines.append(f"- LR AUC = {r['auc_LR_mean']:.3f} ± {r['auc_LR_std']:.3f}")
    lines.append(f"- RF AUC = {r['auc_RF_mean']:.3f} ± {r['auc_RF_std']:.3f}")
    lines.append(f"- features used: {r['feat_cols_used']}\n")
    if b["per_lang"]:
        lines.append("### Per-language (B):")
        for lang, lr in b["per_lang"].items():
            if "error" in lr:
                lines.append(f"- {lang}: error={lr['error']}")
            else:
                lines.append(f"- {lang}: n={lr['n_rows']} "
                             f"LR={lr['auc_LR_mean']:.3f}  "
                             f"RF={lr['auc_RF_mean']:.3f}")
    lines.append("\n## Prior numbers")
    lines.append("- initial accepted-answer prediction: 0.648")
    lines.append("- combined-corpus MI ladder: ~0.52 (chance-level)")
    lines.append("- RF vs LR gain on LC: +0.05 (0.62 → 0.68)")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
