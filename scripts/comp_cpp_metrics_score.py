"""Score the new C++ metrics (a600-a614) on the CF / AC / CC eval cells.

Writes:
  outputs/v2_analysis/comp_cpp_metrics/cf_cpp_scores.parquet
  outputs/v2_analysis/comp_cpp_metrics/ac_cpp_scores.parquet
  outputs/v2_analysis/comp_cpp_metrics/cc_cpp_scores.parquet

Each parquet has pair_id + 15 pairs of (aNNN_score, aNNN_applied) for ALL
rows in the eval cell (not just C++) — applied=0 for non-C++.

Does NOT modify the existing bank parquets.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path("/Users/spangher/Projects/stanford-research/norm-research")
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "outputs/v2_analysis/comp_cpp_metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_IDS = [f"a{i}" for i in range(600, 615)]

LANG_TO_EXT = {
    "python": "py", "py": "py",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cpp14": "cpp",
    "c": "c",
    "java": "java",
    "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts",
    "go": "go", "ruby": "rb", "rust": "rs", "rs": "rs",
    "csharp": "cs", "kotlin": "kt", "swift": "swift",
    "python2": "py",
    "unknown": "txt",
}


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


def score_one_cell(rows, metrics):
    out = []
    for r in rows:
        pid = r["pair_id"]
        lang = str(r.get("candidate_lang") or r.get("language") or "unknown")
        ext = LANG_TO_EXT.get(lang.lower(), "txt")
        fp = f"candidate.{ext}"
        diff = to_synthetic_diff(fp, r.get("candidate_text") or "")
        rec = {"pair_id": pid}
        for m in metrics:
            applied = 0
            score_v = float("nan")
            if diff:
                try:
                    a = m.applies(diff)
                except Exception:
                    a = False
                applied = int(bool(a))
                if applied:
                    try:
                        s = m.score(diff)
                        if s is not None and not (
                            isinstance(s, float) and math.isnan(s)
                        ):
                            score_v = float(s)
                    except Exception:
                        pass
            rec[f"{m.ASPECT_ID}_score"] = score_v
            rec[f"{m.ASPECT_ID}_applied"] = applied
        out.append(rec)
    return out


def main():
    from methods.existing_metrics_runner.coded.metrics import load_all
    all_m = load_all()
    metrics = [m for m in all_m if m.ASPECT_ID in set(NEW_IDS)]
    print(f"loaded {len(metrics)}/{len(NEW_IDS)} new metrics", flush=True)
    assert len(metrics) == len(NEW_IDS)

    # ---- CF eval cell ---- #
    print("\n[CF] loading cf_bank_input.parquet")
    cf = pd.read_parquet(REPO /
        "outputs/v2_analysis/comp_fourplatform_cells/cf_bank_input.parquet")
    print(f"  shape={cf.shape}  lang dist="
          f"{cf['candidate_lang'].value_counts().to_dict()}")
    t0 = time.time()
    out_cf = score_one_cell(cf.to_dict(orient="records"), metrics)
    df_cf = pd.DataFrame(out_cf)
    df_cf.to_parquet(OUT_DIR / "cf_cpp_scores.parquet")
    print(f"  wrote {OUT_DIR / 'cf_cpp_scores.parquet'}  shape={df_cf.shape}  "
          f"runtime={time.time()-t0:.1f}s")
    for aid in NEW_IDS:
        a = df_cf[f"{aid}_applied"]
        s = df_cf[f"{aid}_score"]
        print(f"    {aid}: applied={a.mean():.3f}  "
              f"scored={s.notna().mean():.3f}  mean={s.mean():.3f}")

    # ---- AC eval cell ---- #
    print("\n[AC] loading ac_bank_input.parquet")
    ac = pd.read_parquet(REPO /
        "outputs/v2_analysis/comp_fourplatform_cells/ac_bank_input.parquet")
    print(f"  shape={ac.shape}  lang dist="
          f"{ac['candidate_lang'].value_counts().to_dict()}")
    t0 = time.time()
    out_ac = score_one_cell(ac.to_dict(orient="records"), metrics)
    df_ac = pd.DataFrame(out_ac)
    df_ac.to_parquet(OUT_DIR / "ac_cpp_scores.parquet")
    print(f"  wrote {OUT_DIR / 'ac_cpp_scores.parquet'}  shape={df_ac.shape}  "
          f"runtime={time.time()-t0:.1f}s")
    for aid in NEW_IDS:
        a = df_ac[f"{aid}_applied"]
        s = df_ac[f"{aid}_score"]
        print(f"    {aid}: applied={a.mean():.3f}  "
              f"scored={s.notna().mean():.3f}  mean={s.mean():.3f}")

    # ---- CC eval cell (the v2 unified one with claude_label + language) ---- #
    print("\n[CC] loading comp_unified_claude_bank_scores_v2.parquet "
          "for pair_id list; pulling candidate text from cc_pool + shards")
    cc_v2 = pd.read_parquet(REPO /
        "outputs/v2_analysis/comp_unified_claude_bank_scores_v2.parquet")
    print(f"  CC v2 shape={cc_v2.shape}  language dist="
          f"{cc_v2['language'].value_counts().to_dict()}")

    # Get candidate_text. For "cc" source pairs, candidate_text lives in
    # comp_unified_claude5k_shards (see comp_unified_claude_bank_score_new_metrics.py
    # logic). For "luogu" source pairs, candidate_text is also there.
    import glob, json
    text_map = {}
    for p in sorted(glob.glob(str(REPO /
        "outputs/v2_analysis/comp_unified_claude5k_shards/shards/shard_*.jsonl"))):
        for ln in open(p):
            r = json.loads(ln)
            text_map[r["pair_id"]] = {
                "candidate_text": r.get("candidate_text") or "",
                "language": r.get("language") or "unknown",
            }
    print(f"  loaded {len(text_map)} pair_id -> text entries from 5K shards")
    # Also pull the 3900 batch (Qwen seed)
    df3900 = pd.read_parquet(REPO /
        "outputs/v2_analysis/comp_unified_claude3900_shards/_input_3900.parquet")
    for _, r in df3900.iterrows():
        if r["pair_id"] in text_map:
            continue
        text_map[r["pair_id"]] = {
            "candidate_text": r.get("candidate_code") or "",
            "language": r.get("language") or "unknown",
        }
    print(f"  text_map after 3900 merge: {len(text_map)}")

    rows = []
    miss = 0
    for _, r in cc_v2.iterrows():
        pid = r["pair_id"]
        info = text_map.get(pid)
        if not info:
            miss += 1
            rows.append({
                "pair_id": pid,
                "candidate_lang": r.get("language") or "unknown",
                "candidate_text": "",
            })
            continue
        rows.append({
            "pair_id": pid,
            "candidate_lang": info["language"],
            "candidate_text": info["candidate_text"],
        })
    print(f"  candidate_text missing: {miss}/{len(rows)}")

    t0 = time.time()
    out_cc = score_one_cell(rows, metrics)
    df_cc = pd.DataFrame(out_cc)
    df_cc.to_parquet(OUT_DIR / "cc_cpp_scores.parquet")
    print(f"  wrote {OUT_DIR / 'cc_cpp_scores.parquet'}  shape={df_cc.shape}  "
          f"runtime={time.time()-t0:.1f}s")
    for aid in NEW_IDS:
        a = df_cc[f"{aid}_applied"]
        s = df_cc[f"{aid}_score"]
        print(f"    {aid}: applied={a.mean():.3f}  "
              f"scored={s.notna().mean():.3f}  mean={s.mean():.3f}")


if __name__ == "__main__":
    main()
