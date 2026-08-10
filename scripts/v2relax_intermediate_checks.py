"""Intermediate checks on partial v2relax responses.

Check 1: per-aspect delta with relaxed applicability vs legacy.
  - For each aspect scored so far, compute new delta + N, compare to
    legacy delta (qwen_thinking_fp8). Look for the previously-high-delta-
    low-N rubrics (callback a225, premise hook a91, audience a37, etc.):
    did their effects hold at higher N?

Check 2: per-aspect score distribution.
  - For each aspect, % of cells with score=0/0.5/1 (among applicable).
  - High-0.5-rate aspects = judge is uncertain on most stories =
    feature carries less signal even if applicable.
"""
import json
import os
import re
import statistics as stats
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

REPO = Path("/lfs/skampere3/0/alexspan/norm-research")
RESPONSE_ROOT = REPO / "runs/cw_relax_appl/v2relax_responses"
LEGACY_PARQUET = REPO / "outputs/v2_db/cells_v1/task=creative_writing/judge=qwen_thinking_fp8/data.parquet"
LABELS_FILE = REPO / "runs/validity_full/v2/creative_writing/datapoints.json"
ASPECTS_FILE = REPO / "runs/validity_full/v2/creative_writing/aspects.json"


def _extract_json(raw: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    depth = 0; end = None
    for i in range(len(raw) - 1, -1, -1):
        if raw[i] == "}":
            if depth == 0: end = i
            depth += 1
        elif raw[i] == "{":
            depth -= 1
            if depth == 0:
                cand = raw[i:end + 1]
                if '"results"' in cand:
                    try: return json.loads(cand)
                    except: pass
                end = None
    m = re.search(r"\{[\s\S]*\"results\"[\s\S]*\}", raw)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None


def collect_new_cells():
    rows = []
    for shard in sorted(RESPONSE_ROOT.glob("shard_*")):
        for f in shard.iterdir():
            if not f.name.startswith("creative_writing__"):
                continue
            raw = f.read_text(errors="ignore")
            j = _extract_json(raw)
            if not j or "results" not in j: continue
            for tr in j["results"]:
                if not isinstance(tr, dict): continue
                dp = tr.get("text_id")
                if not dp: continue
                scores = tr.get("scores")
                if not isinstance(scores, list): continue
                for sc in scores:
                    if not isinstance(sc, dict): continue
                    aid = sc.get("aspect_id")
                    if not aid: continue
                    rows.append({
                        "datapoint_id": dp,
                        "aspect_id": aid,
                        "applicable": bool(sc.get("applicable")),
                        "score": sc.get("score"),
                    })
    return pd.DataFrame(rows)


def main():
    aspects_meta = {a["aspect_id"]: a
                    for a in json.loads(ASPECTS_FILE.read_text())}
    labels = pd.Series({
        d["datapoint_id"]: int(d["judgement"])
        for d in json.loads(LABELS_FILE.read_text())
        if d.get("judgement") is not None
    })
    new = collect_new_cells()
    print(f"new cells collected: {len(new)}")
    print(f"new datapoints covered: {new['datapoint_id'].nunique()}")
    print(f"new aspects covered: {new['aspect_id'].nunique()}")
    if new.empty: return

    # ---- Check 1: per-aspect delta, new vs legacy ----
    legacy = pd.read_parquet(LEGACY_PARQUET)
    legacy["score_num"] = legacy["score"].where(legacy["applicable"], np.nan).astype(float)

    def per_aspect_delta(df, label="new"):
        df = df.copy()
        if "score_num" not in df.columns:
            df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
            df.loc[~df["applicable"], "score_num"] = np.nan
        out = {}
        for aid, sub in df.groupby("aspect_id"):
            mask = sub["score_num"].notna()
            if mask.sum() < 10: continue
            sub2 = sub[mask].copy()
            sub2["y"] = sub2["datapoint_id"].map(labels)
            sub2 = sub2[sub2["y"].notna()]
            s0 = sub2[sub2["y"] == 0]["score_num"]
            s1 = sub2[sub2["y"] == 1]["score_num"]
            if len(s0) < 5 or len(s1) < 5: continue
            delta = float(s1.mean() - s0.mean())
            try: _, p = ttest_ind(s1, s0, equal_var=False)
            except: p = float("nan")
            appl_rate = float(mask.sum() / len(sub))
            out[aid] = {"delta": delta, "p": float(p),
                        "n0": int(len(s0)), "n1": int(len(s1)),
                        "appl_rate": appl_rate}
        return out

    new_lift = per_aspect_delta(new, "new")
    legacy_lift = per_aspect_delta(legacy.rename(columns={}), "legacy")

    # Compare for the headline audience-appeal aspects
    HEADLINE = [("a91", "Premise clarity / framing / hook"),
                ("a225", "Callback design and deployment (humor task aspect, may also exist here)"),
                ("a37", "Audience targeting / contextual fit (humor task)"),
                ("a129", "Opening hook and momentum"),
                ("a130", "Ending payoff / closure / resonance"),
                ("a133", "Narrative drive: tension / questions / stakes"),
                ("a184", "Affective impact and reader resonance"),
                ("a187", "Ending strategy and effect"),
                ("a233", "Emotional engagement and resonance"),
                ("a239", "Reader immersion and sustained interest"),
                ("a254", "Opening hook (page-one effectiveness)"),
                ("a258", "Reader momentum / page-turnability"),
                ("a278", "Opening: hook, orientation, promise alignment"),
                ("a303", "Reader engagement and sustained attention"),
                ("a307", "Opening Hook Effectiveness"),
                ("a49", "Foreshadowing / misdirection")]
    print()
    print("=" * 95)
    print("CHECK 1: HEADLINE AUDIENCE-APPEAL ASPECTS — legacy vs v2relax delta")
    print("=" * 95)
    print(f"{'aspect_id':<12} {'name':<48} {'legacy delta (n0/n1)':<26} {'new delta (n0/n1)':<22}")
    print("-" * 95)
    for aid, nm in HEADLINE:
        L = legacy_lift.get(aid); N = new_lift.get(aid)
        if L is None and N is None: continue
        L_s = f"{L['delta']:+.3f} ({L['n0']:>3d}/{L['n1']:>3d})" if L else "—"
        N_s = f"{N['delta']:+.3f} ({N['n0']:>3d}/{N['n1']:>3d})" if N else "(no data yet)"
        print(f"{aid:<12} {nm[:48]:<48} {L_s:<26} {N_s:<22}")

    # ---- Check 2: per-aspect score distribution ----
    print()
    print("=" * 95)
    print("CHECK 2: NEW PER-ASPECT SCORE DISTRIBUTION (% 0.0 / 0.5 / 1.0 / N/A)")
    print("=" * 95)
    summary = []
    for aid in sorted(new["aspect_id"].unique()):
        sub = new[new["aspect_id"] == aid]
        n_total = len(sub)
        appl = sub[sub["applicable"]]
        n_app = len(appl)
        n_na = n_total - n_app
        sc = Counter(appl["score"])
        n0 = sc.get(0.0, 0); n5 = sc.get(0.5, 0); n1 = sc.get(1.0, 0)
        summary.append({"aspect_id": aid, "n": n_total,
                        "pct_na": n_na/n_total*100,
                        "pct_0": n0/n_app*100 if n_app else 0,
                        "pct_5": n5/n_app*100 if n_app else 0,
                        "pct_1": n1/n_app*100 if n_app else 0,
                        "name": aspects_meta.get(aid, {}).get("name", "")[:60]})
    summ_df = pd.DataFrame(summary).sort_values("pct_5", ascending=False)
    print(f"{'aspect_id':<8} {'n':>5} {'%na':>5} {'%0':>5} {'%0.5':>6} {'%1':>5}  name")
    print("-" * 95)
    # show top-10 high-uncertainty (high 0.5 rate) and top-10 low-uncertainty
    print("--- TOP 10 high-uncertainty (judge mostly says 0.5) ---")
    for _, r in summ_df.head(10).iterrows():
        print(f"{r['aspect_id']:<8} {int(r['n']):>5} {r['pct_na']:>5.0f} "
              f"{r['pct_0']:>5.0f} {r['pct_5']:>6.0f} {r['pct_1']:>5.0f}  {r['name']}")
    print("--- TOP 10 low-uncertainty (judge often gives 0 or 1) ---")
    for _, r in summ_df.tail(10).iterrows():
        print(f"{r['aspect_id']:<8} {int(r['n']):>5} {r['pct_na']:>5.0f} "
              f"{r['pct_0']:>5.0f} {r['pct_5']:>6.0f} {r['pct_1']:>5.0f}  {r['name']}")

    # Aggregate
    print()
    print(f"Mean % 0.5 across aspects: {summ_df['pct_5'].mean():.1f}%")
    print(f"Mean % NA across aspects:  {summ_df['pct_na'].mean():.1f}%")
    print(f"Aspects with >70% 0.5: {(summ_df['pct_5'] > 70).sum()}")


if __name__ == "__main__":
    main()
