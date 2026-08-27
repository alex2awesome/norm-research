#!/usr/bin/env python3
"""SNL cut-for-time VERDICT cell — ASR lane (identical-by-construction pipeline).

Both classes' AUDIO transcribed by ONE ASR system (faster-whisper large-v3,
transcribe_asr.py, same flags, same machine): the text-production pipeline is
identical by construction, removing the fan-transcript cell's class-correlated
transcriber-authorship confound. Same canonical renderer (norm imported
verbatim from reddit_jokes build_removal_v2_normalized), same class-name
scrub, same sha256 splits, same 1:1 within-season length-matched draw design
with canonical draw + 25 redraw probe battery as build_verdict_cell.py.

Outputs snl_population_asr.csv.gz + snl_population_asr_meta.json.
NEVER overwrites the fan-transcript population (separate _asr suffix; refuses
to overwrite its own outputs too).
"""
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parent
REPO = D.parents[2]
sys.path.insert(0, str(REPO / "datasets/humor/reddit_jokes"))
from build_removal_v2_normalized import norm  # canonical matched renderer

MIN_LEN = 200
N_REDRAWS = 25
CLASS_NAME_RE = re.compile(
    r"cut[\s\-]*(for[\s\-]*time|after[\s\-]*dress|from[\s\-]*dress|sketch)",
    re.IGNORECASE)


def split_of(row_id: str) -> str:
    h = int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def probe_aucs(df: pd.DataFrame):
    """Char 3-5 gram vs word 1-2 gram grouped-OOF logistic AUC, grouped by
    season (identical spec to build_verdict_cell.py::probe_aucs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    y = df.judgement.values
    out = {}
    for name, vec in (
        ("char35", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                   max_features=30000, min_df=2)),
        ("word12", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                   max_features=30000, min_df=2)),
    ):
        X = vec.fit_transform(df.text)
        groups = df.group.values
        k = min(5, len(set(groups)))
        if k < 2:
            out[f"{name}_by_season"] = None
            continue
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(k).split(X, groups=groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        out[f"{name}_by_season"] = round(float(roc_auc_score(y, oof)), 4)
    return out


def main():
    man = {r["id"]: r for r in
           (json.loads(l) for l in open(D / "snl_asr_manifest.jsonl"))}
    rendered, n_missing, n_short = {}, 0, 0
    for rid, r in man.items():
        fp = D / "asr_json" / f"{rid}.asr.json"
        if not fp.exists():
            n_missing += 1
            continue
        j = json.loads(fp.read_text())
        txt = norm(CLASS_NAME_RE.sub(" ", j["text"]))
        if len(txt) < MIN_LEN:
            n_short += 1
            continue
        rendered[rid] = {
            "key": rid, "verdict": r["class"], "season": r["season"],
            "title": CLASS_NAME_RE.sub(" ", r.get("title") or "").strip(" :|-"),
            "text": txt, "url": r["url"],
            "audio_dur": j.get("audio_duration"),
        }

    cut = [v for v in rendered.values() if v["verdict"] == "cut_for_time"]
    aired = [v for v in rendered.values() if v["verdict"] == "aired"]
    print(f"rendered usable: aired={len(aired)} cut={len(cut)} "
          f"(missing asr={n_missing}, dropped short<{MIN_LEN}={n_short})",
          flush=True)

    def draw(idx: int) -> pd.DataFrame:
        rng = np.random.default_rng(
            int(hashlib.sha256(f"snl_cft_asr_draw|{idx}".encode())
                .hexdigest()[:8], 16))
        used, recs = set(), []
        for c in sorted(cut, key=lambda x: x["key"]):
            cands = [a for a in aired
                     if a["season"] == c["season"] and a["key"] not in used]
            if not cands:
                cands = [a for a in aired
                         if abs(a["season"] - c["season"]) <= 1
                         and a["key"] not in used]
            if not cands:
                continue
            cands.sort(key=lambda a: abs(np.log(len(a["text"]) / len(c["text"]))))
            pick = cands[int(rng.integers(0, min(8, len(cands))))]
            used.add(pick["key"])
            for item, lab in ((c, 0), (pick, 1)):
                recs.append({"row_id": f"snl_cft_asr:{item['key']}",
                             "judgement": lab, "text": item["text"],
                             "group": f"s{item['season']}",
                             "title": item["title"],
                             "n_chars": len(item["text"])})
        return pd.DataFrame(recs)

    df0 = draw(0)
    df0["split"] = df0.row_id.map(split_of)
    probes0 = probe_aucs(df0)
    print("canonical draw probes:", json.dumps(probes0), flush=True)

    redraw = {k: [] for k in probes0}
    for i in range(1, N_REDRAWS + 1):
        p = probe_aucs(draw(i))
        for k, v in p.items():
            if v is not None:
                redraw[k].append(v)
    redraw_summary = {k: {"mean": round(float(np.mean(v)), 4),
                          "sd": round(float(np.std(v)), 4), "n_draws": len(v)}
                      for k, v in redraw.items() if v}
    print("redraw probe summary:", json.dumps(redraw_summary), flush=True)

    out_csv = D / "snl_population_asr.csv.gz"
    out_meta = D / "snl_population_asr_meta.json"
    for p in (out_csv, out_meta):
        if p.exists():
            raise SystemExit(f"REFUSING to overwrite existing {p} — "
                             f"version-suffix manually")

    cols = ["row_id", "judgement", "text", "group", "split", "n_chars", "title"]
    df0[cols].sort_values("row_id").to_csv(out_csv, index=False,
                                           compression="gzip")

    lenstats = df0.groupby("judgement").n_chars.describe()[["count", "mean", "50%"]]
    from collections import Counter
    meta = {
        "cell": "snl_cut_for_time VERDICT — ASR lane (humor; aired=1 vs "
                "cut_for_time=0); identical-by-construction text pipeline",
        "built": "2026-08-22",
        "n": int(len(df0)),
        "n_cut_usable": len(cut),
        "n_aired_usable": len(aired),
        "n_cut_catalog_total": 87,
        "n_cut_with_youtube_url": 36,
        "design": "1:1 within-season length-matched aired controls drawn from a "
                  "3:1 season+duration-matched download pool; canonical draw=0 "
                  "of 26 deterministic draws",
        "cut_season_distribution": dict(sorted(Counter(
            f"S{c['season']}" for c in cut).items())),
        "length_stats_by_class": json.loads(lenstats.to_json()),
        "splits": "sha256(row_id) mod 100 -> 80/10/10 (stable hash)",
        "renderer_provenance": {
            "asr": "faster-whisper large-v3, transcribe_asr.py — ONE ASR system, "
                   "one machine (sk2), one flag set, BOTH classes: text pipeline "
                   "identical by construction (no fan transcribers anywhere)",
            "normalizer": "datasets/humor/reddit_jokes/"
                          "build_removal_v2_normalized.py::norm (imported verbatim)",
            "class_name_scrub": CLASS_NAME_RE.pattern,
        },
        "confounds_declared": [
            "class-correlated PRODUCTION quality may remain: cut-for-time uploads "
            "are dress-rehearsal recordings (audience/mix differ from live "
            "broadcast); ASR of laugh-track/audio-quality differences can leave a "
            "residual acoustic-channel fingerprint in text (e.g., transcription "
            "noise). This is a property of the material, not the pipeline.",
            "only 36/87 catalog cut-for-time sketches have YouTube URLs (S42-S47); "
            "cell measures the 2016-2022 official-upload slice.",
            "aired pool located by exact title match into the official SNL channel "
            "(994 unique matches); ambiguous titles dropped.",
        ],
        "artifact_probe": {
            "spec": "grouped-OOF logistic; char_wb 3-5gram vs word 1-2gram; "
                    "grouped by season; GroupKFold",
            "canonical_draw": probes0,
            "redraw_mean_sd_over_25": redraw_summary,
            "gate_rule": "char AUC >~.65 with word AUC comparable-or-lower flags "
                         "pipeline fingerprint",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=1))
    print(f"WROTE {out_csv} ({len(df0)} rows) + {out_meta}")
    print("SNL_ASR_CELL_DONE")


if __name__ == "__main__":
    main()
