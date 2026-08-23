#!/usr/bin/env python3
"""SNL cut-for-time VERDICT cell (humor verdict channel: aired=1 vs cut_for_time=0).

MATCHED PIPELINE: both classes are fan transcripts from snltranscripts.jt.org,
scraped by ONE scraper (scrape_transcripts.py), extracted by ONE extractor
(this file), normalized by ONE renderer (imported verbatim from
datasets/humor/reddit_jokes/build_removal_v2_normalized.py::norm — the
jokes-v2 leak-fix chain: html unescape x3, NFKC, invisibles, ASCII punct,
markdown strip, URL strip, whitespace collapse).

DECLARED CONFOUND: transcript AUTHORSHIP is class-correlated — aired sketches
were fan-transcribed over decades by many transcribers; cut-for-time pages were
added later (2018-2019 era) possibly by different/fewer transcribers. The
scrape+extract+render pipeline is shared, but transcriber style is not
controllable here. Within-class two-source contrast (raw/transcript_samples
HTML vs transcripts_fan json for the same pages) checks the EXTRACTION stage
only, not authorship.

Class-name scrub: any literal "cut for time"/"cut after dress"/etc. removed
from text bodies of BOTH classes.

PILOT-n: only 20 cut-for-time sketches have fan transcripts (S44 x2, S45 x18).
Population = 1:1 within-season, length-matched aired controls. Artifact probes
(char 3-5 gram and word 1-2 gram grouped-OOF logistic) reported for the
canonical draw plus mean±sd over 25 matched redraws.

Splits: sha256(row_id) -> 80/10/10 (stable hash, never seeded shuffle).
"""
import gzip
import hashlib
import html as _html
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

MIN_LEN = 200          # post-norm chars; a sketch transcript shorter = torn extraction
N_REDRAWS = 25

FOOTER_MARKERS = [
    "\nAuthor:\n", "\nAuthor \n", "View all posts by", "\nPosted on \n",
    "Leave a Reply", "Post navigation", "Contact us @",
]
CLASS_NAME_RE = re.compile(
    r"cut[\s\-]*(for[\s\-]*time|after[\s\-]*dress|from[\s\-]*dress|sketch)",
    re.IGNORECASE)
HEADER_LINE_RES = [
    re.compile(r"^saturday night live transcripts$", re.I),
    re.compile(r"^season\s+\d+:\s*episode\s+\d+$", re.I),
    re.compile(r"^\S{1,6}:\s*[^/]{1,40}\s/\s.{1,60}$"),   # "13q: Host / Musical guest"
    re.compile(r"^s\d+e\d+\b", re.I),
]
CREDIT_LINE_RE = re.compile(r"^(\.{2,}|…+\.*)")            # ".....Anna Kendrick" credits
# cast-credit lines: 1-3 dot-separated groups of capitalized name tokens, >=2
# tokens total, no colon ("Matt Damon", "Rudy Giuliani. Kate McKinnon")
NAME_GRP = r"[A-Z][\w'&\-]*(?:\s+[A-Z][\w'&\-]*){0,4}"
BARE_NAME_RE = re.compile(rf"^(?:{NAME_GRP}\.?\s*){{1,3}}$")
# "Character... Actor" credit lines (ellipsis/dots separator), both classes
CREDIT_ELL_RE = re.compile(rf"^{NAME_GRP}(?:'s [a-z]+)?\s*(?:…+|\.{{2,}})+\s*\.*{NAME_GRP}$")


def crude_strip(html_text: str) -> str:
    """Replicate scrape_transcripts.py's tag strip so raw HTML and stored
    raw_text enter the SAME downstream extractor."""
    body = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html_text, flags=re.S)
    body = re.sub(r"<[^>]+>", "\n", body)
    return re.sub(r"\n{3,}", "\n\n", body)


def extract_body(raw_text: str) -> str:
    """Shared extractor, identical for both classes: cut site header (chrome up
    through the in-page h1 = page title, self-derived from the <title> text at
    the top of the page), cut footer (author box onward), drop structural
    header lines, scrub class-naming strings."""
    t = raw_text
    # self-derive page title from the <title> line at the head of the page
    head = t[:600]
    tm = re.search(r"\s*-\s*SNL Transcripts Tonight", head)
    tit = ""
    if tm:
        pre = head[: tm.start()].strip()
        tit = pre.splitlines()[-1].strip() if pre else ""
    m = t.find("For Die Hard Saturday Night Live Fans")
    if m >= 0:
        t = t[m + len("For Die Hard Saturday Night Live Fans"):]
    # h1 == page title; plain substring find on truncated form (entity variants
    # differ between <title> and h1, so use the prefix before any entity)
    if tit:
        probe_s = re.split(r"&#\d+;", tit)[0].strip()[:60]
        if len(probe_s) >= 8:
            i = t.find(probe_s)
            if i >= 0:
                # skip to end of that h1 line
                j = t.find("\n", i)
                t = t[j if j >= 0 else i + len(probe_s):]
    # footer cut: earliest marker
    ends = [t.find(mk) for mk in FOOTER_MARKERS if t.find(mk) >= 0]
    if ends:
        t = t[: min(ends)]
    # line-level structural filters (identical both classes)
    lines, kept = t.split("\n"), []
    n_seen_content = 0
    for ln in lines:
        # unescape entities BEFORE line filters (credits arrive as "&#8230;")
        s = _html.unescape(_html.unescape(ln)).strip()
        if not s:
            continue
        if any(rx.match(s) for rx in HEADER_LINE_RES):
            continue
        if CREDIT_LINE_RE.match(s):
            continue
        if CREDIT_ELL_RE.match(s):
            continue
        # bare cast-credit lines (capitalized name groups, >=2 name tokens,
        # no colon): dropped only in the first 30 content lines where credits
        # live — same window for both classes
        if (n_seen_content < 30 and ":" not in s and BARE_NAME_RE.match(s)
                and len(re.findall(r"[A-Z][\w'&\-]*", s)) >= 2):
            continue
        n_seen_content += 1
        kept.append(s)
    body = "\n".join(kept)
    body = CLASS_NAME_RE.sub(" ", body)
    return norm(body)


def urlkey(u: str) -> str:
    return hashlib.sha256(u.encode()).hexdigest()[:16]


def split_of(row_id: str) -> str:
    h = int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def probe_aucs(df: pd.DataFrame):
    """Char 3-5 gram vs word 1-2 gram grouped-OOF logistic AUC, grouped by
    season/era (the only non-degenerate group at PILOT-n: per-episode codes are
    not recoverable for the aired fan URLs)."""
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
    rows = [json.loads(l) for l in open(D / "snl_catalog.jsonl")]
    fan = [r for r in rows if "snltranscripts" in (r.get("url") or "")]

    rendered, n_missing, n_short = {}, 0, 0
    for r in fan:
        fp = D / "transcripts_fan" / f"{urlkey(r['url'])}.json"
        if not fp.exists():
            n_missing += 1
            continue
        j = json.loads(fp.read_text())
        txt = extract_body(j["raw_text"])
        if len(txt) < MIN_LEN:
            n_short += 1
            continue
        rendered[urlkey(r["url"])] = {
            "key": urlkey(r["url"]), "verdict": r["verdict"], "season": r["season"],
            "title": CLASS_NAME_RE.sub(" ", r.get("title") or "").strip(" :|-"),
            "text": txt, "url": r["url"],
        }

    cut = [v for v in rendered.values() if v["verdict"] == "cut_for_time"]
    aired = [v for v in rendered.values() if v["verdict"] == "aired"]
    print(f"rendered usable: aired={len(aired)} cut={len(cut)} "
          f"(missing files={n_missing}, dropped short<{MIN_LEN}={n_short})", flush=True)

    # ---- within-class two-source pipeline-fingerprint check -----------------
    import difflib
    fp_check = {"aired": [], "cut_for_time": []}
    for r in rows:
        tp = r.get("transcript_path") or ""
        if "transcript_samples" not in tp or "snltranscripts" not in (r.get("url") or ""):
            continue
        k = urlkey(r["url"])
        if k not in rendered:
            continue
        hp = D / tp
        if not hp.exists():
            continue
        alt = extract_body(crude_strip(hp.read_text(errors="replace")))
        a, b = rendered[k]["text"], alt
        if not b:
            continue
        sim = difflib.SequenceMatcher(None, a[:20000], b[:20000]).ratio()
        fp_check[r["verdict"]].append(round(sim, 4))
    fp_summary = {c: {"n_pairs": len(v),
                      "mean_sim": round(float(np.mean(v)), 4) if v else None,
                      "min_sim": round(float(np.min(v)), 4) if v else None}
                  for c, v in fp_check.items()}
    print("two-source fingerprint check:", json.dumps(fp_summary), flush=True)

    # ---- 1:1 within-season length-matched draws ------------------------------
    def draw(idx: int) -> pd.DataFrame:
        rng = np.random.default_rng(
            int(hashlib.sha256(f"snl_cft_draw|{idx}".encode()).hexdigest()[:8], 16))
        used, recs = set(), []
        for c in sorted(cut, key=lambda x: x["key"]):
            cands = [a for a in aired if a["season"] == c["season"] and a["key"] not in used]
            if not cands:
                cands = [a for a in aired if abs(a["season"] - c["season"]) <= 1
                         and a["key"] not in used]
            if not cands:
                continue
            # length bin: nearest by |log len ratio|, tie-broken randomly among top 8
            cands.sort(key=lambda a: abs(np.log(len(a["text"]) / len(c["text"]))))
            pick = cands[int(rng.integers(0, min(8, len(cands))))]
            used.add(pick["key"])
            for item, lab in ((c, 0), (pick, 1)):
                recs.append({"row_id": f"snl_cft:{item['key']}", "judgement": lab,
                             "text": item["text"], "group": f"s{item['season']}",
                             "title": item["title"], "n_chars": len(item["text"])})
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

    # ---- outputs -------------------------------------------------------------
    out_csv = D / "snl_population.csv.gz"
    out_meta = D / "snl_population_meta.json"
    for p in (out_csv, out_meta):
        if p.exists():
            raise SystemExit(f"REFUSING to overwrite existing {p} — version-suffix manually")

    cols = ["row_id", "judgement", "text", "group", "split", "n_chars", "title"]
    df0[cols].sort_values("row_id").to_csv(out_csv, index=False, compression="gzip")

    lenstats = df0.groupby("judgement").n_chars.describe()[["count", "mean", "50%"]]
    meta = {
        "cell": "snl_cut_for_time VERDICT (humor; aired=1 vs cut_for_time=0) — PILOT-n",
        "built": "2026-08-22",
        "n": int(len(df0)),
        "n_aired_usable_total": len(aired),
        "n_cut_usable_total": len(cut),
        "n_cut_catalog_total": 87,
        "n_cut_fan_transcripts": 20,
        "n_cut_yt_auto_subs_EXCLUDED": 6,
        "design": "1:1 within-season length-matched aired controls; PILOT-n; "
                  "canonical draw=0 of 26 deterministic draws",
        "cut_season_distribution": {"S44": 2, "S45": 18},
        "length_stats_by_class": json.loads(lenstats.to_json()),
        "splits": "sha256(row_id) mod 100 -> 80/10/10 (stable hash)",
        "renderer_provenance": {
            "scraper": "datasets/humor/snl_cut_for_time/scrape_transcripts.py (one scraper, both classes)",
            "extractor": "datasets/humor/snl_cut_for_time/build_verdict_cell.py::extract_body (one extractor, both classes)",
            "normalizer": "datasets/humor/reddit_jokes/build_removal_v2_normalized.py::norm (imported verbatim)",
            "class_name_scrub": CLASS_NAME_RE.pattern,
        },
        "confounds_declared": [
            "transcriber authorship is class-correlated: aired = fan-transcribed over decades "
            "by many transcribers; cut-for-time pages added ~2018-2019, possibly different/fewer "
            "transcribers. Shared scrape/extract/render pipeline does NOT remove authorship style.",
            "era coverage: cut fan transcripts exist only for S44-S45; aired matched within-season, "
            "so the cell measures a 2018-2019 slice only.",
            "PILOT-n: 20 cut items; all probe/scoring readouts are wide.",
            "6 cut-for-time YouTube auto-sub transcripts EXCLUDED (class-pure different pipeline "
            "= guaranteed fingerprint if included).",
        ],
        "pipeline_fingerprint_check_two_source": {
            "what": "same-page raw HTML (raw/transcript_samples) vs scraper raw_text, both through "
                    "the shared extractor+renderer; difflib ratio per sketch",
            **fp_summary,
        },
        "artifact_probe": {
            "spec": "grouped-OOF logistic; char_wb 3-5gram vs word 1-2gram; grouped by season/era; "
                    "GroupKFold (episode-level grouping degenerate at PILOT-n: aired fan URLs carry "
                    "no recoverable episode code)",
            "canonical_draw": probes0,
            "redraw_mean_sd_over_25": redraw_summary,
            "gate_rule": "char AUC >~.65 with word AUC comparable-or-lower flags pipeline fingerprint",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=1))
    print(f"WROTE {out_csv} ({len(df0)} rows) + {out_meta}")
    print("SNL_VERDICT_CELL_DONE")


if __name__ == "__main__":
    main()
