#!/usr/bin/env python3
"""ATTRIBUTION-ADEQUACY port (journalism analog of peer adequacy-mode):
per news claim, is it SOURCED (attributed to a named source/document/quote) or
ASSERTED_ONLY (reporter's own voice)? LLM-graded version of the CODE sourcing census.
Design: 700 twitter-y docs (outputs/multi_y_news/metrics_audited.csv), claims from the
head, judged against the FULL article; MISMATCHED placebo every 3rd doc (expect
NOT_FOUND); 6 blinded ANCHORS in the batch. Readouts: placebo/anchors, convergent
validity vs CODE census, grouped-CV AUC vs y_twitter with length audit.
Run on sk3: python -m methods.claim_verification.run_attrib_adequacy"""
import glob, json, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "methods")
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from claim_verification.core import Cache, _post, _parse_json, _key, extract_claims, split_head_body
from claim_verification.evidence_api import clean_evidence_text

ROOT = "/lfs/skampere3/0/alexspan/norm-research"
NH = f"{ROOT}/datasets/news-homepages"
CFG = {"base_url": "http://127.0.0.1:8006/v1", "model": "gemma",
       "doc_kind": "news article", "max_claims": 6}

ATTR = """Below is a NEWS ARTICLE and a CLAIM.

ARTICLE:
{art}

CLAIM: "{claim}"

Judge how the article supports the claim:
- "SOURCED": the claim is attributed to a named source — a person, organization,
  document, dataset, study, or official statement — or supported by a direct quote.
- "ASSERTED_ONLY": the claim appears but is stated in the reporter's own voice,
  with no attribution to any source.
- "NOT_FOUND": the claim does not appear in this article.

Return ONLY JSON: {{"verdict": "SOURCED"|"ASSERTED_ONLY"|"NOT_FOUND", "who": "<the source, or ''>"}}"""

ANCHORS = [
    ("SOURCED", "The city will close two bridges for repairs next month.",
     "Officials announced road work Tuesday. \"We will close the Elm Street and Oak Avenue "
     "bridges for repairs starting next month,\" said transportation director Maria Lopez. "
     "The closures are expected to last twelve weeks, the department said in a statement."),
    ("SOURCED", "Unemployment in the county fell to 4.1 percent in May.",
     "New numbers paint a brighter picture for the region. County unemployment fell to 4.1 "
     "percent in May, according to data released Friday by the state labor department. "
     "Economists credited seasonal hiring at the port."),
    ("ASSERTED_ONLY", "The new stadium will transform the city's downtown.",
     "The new stadium will transform the city's downtown. Cranes already tower over the "
     "district, and the project is the largest construction effort in a generation. "
     "Local shop owners are preparing for bigger crowds."),
    ("ASSERTED_ONLY", "The mayor's plan has stalled amid council infighting.",
     "The mayor's plan has stalled amid council infighting. Weeks of meetings have produced "
     "no vote, and the proposal now sits in committee. The next session is scheduled for "
     "Thursday evening at city hall."),
    ("NOT_FOUND", "The governor vetoed the education funding bill on Friday.",
     "The bakery on Fifth Street celebrated its fiftieth anniversary this weekend with free "
     "pastries and live music. Hundreds of neighbors stopped by. \"This block wouldn't be "
     "the same without them,\" said longtime customer Joan Pierce."),
    ("NOT_FOUND", "Researchers discovered a new species of deep-sea coral.",
     "The high school football team won its third straight title on Saturday. Coach Daniels "
     "praised the defense after the 21-7 victory. The team finishes the season undefeated "
     "at home for the first time since 2009."),
]

def attr_check(claim, art, cache):
    k = _key("attr_adeq", CFG["model"], claim[:180], art[:200])
    hit = cache.get(k)
    if hit is not None: return hit
    raw = _post(CFG["base_url"], CFG["model"],
                ATTR.format(art=art[:3500], claim=claim[:350]), max_tokens=140)
    obj = _parse_json(raw) or {}
    v = str(obj.get("verdict", "")).upper()
    if v not in ("SOURCED", "ASSERTED_ONLY", "NOT_FOUND"): v = "PARSE_FAIL"
    out = {"verdict": v, "who": str(obj.get("who", ""))[:80]}
    cache.put(k, out)
    return out

def gauc(M, cols, y, g, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
    X = M[cols].values.astype(float)
    mk = ~np.all(np.isnan(X), axis=1) & pd.Series(y).notna().values
    if mk.sum() < 80 or len(set(np.asarray(y)[mk])) < 2:
        print(f"  {label:42} SKIP (n={int(mk.sum())})", flush=True); return
    folds = min(5, len(set(np.asarray(g)[mk])))
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    a = cross_val_score(pipe, X[mk], np.asarray(y)[mk].astype(int),
                        cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
                        groups=np.asarray(g)[mk], scoring="roc_auc")
    print(f"  {label:42} AUC={np.nanmean(a):.4f} (n={int(mk.sum())})", flush=True)

def main():
    M = pd.read_csv(f"{ROOT}/outputs/multi_y_news/metrics_audited.csv")
    urls = set(M.url)
    text = {}
    for p in glob.glob(f"{NH}/fulltext/fulltext_v2_shard*.jsonl"):
        for ln in open(p):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("url") in urls and len(r.get("text") or "") > 400:
                text[r["url"]] = clean_evidence_text(r["text"])
    docs = [(u, text[u]) for u in M.url if u in text]
    print(f"[attr] {len(docs)} docs with text", flush=True)
    cache = Cache(f"{ROOT}/outputs/attrib_adequacy/cache.jsonl")
    lock = Lock(); rows = []
    def work(i):
        u, t = docs[i]
        head, _ = split_head_body(t)
        try: claims = extract_claims(head, CFG, cache)
        except Exception: return
        out = []
        for c in claims:
            cl = c["claim"] if isinstance(c, dict) else str(c)
            try: r = attr_check(cl, t, cache)
            except Exception: continue
            out.append({"url": u, "arm": "matched", "claim": cl[:120], **r})
            if i % 3 == 0:  # placebo: same claim vs a far-away article
                try: rp = attr_check(cl, docs[(i + len(docs) // 2) % len(docs)][1], cache)
                except Exception: continue
                out.append({"url": u, "arm": "placebo", "claim": cl[:120], **rp})
        with lock:
            rows.extend(out)
            done = len({r_["url"] for r_ in rows})
            if done % 50 == 0: print(f"[attr] ~{done}/{len(docs)} docs", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(work, range(len(docs))))
    # anchors (blinded known-label)
    ok = 0
    for want, cl, art in ANCHORS:
        v = attr_check(cl, art, cache)["verdict"]
        ok += int(v == want)
    print(f"[attr] ANCHORS {ok}/{len(ANCHORS)}", flush=True)
    F = pd.DataFrame(rows)
    F.to_csv(f"{ROOT}/outputs/attrib_adequacy/results.csv", index=False)
    for arm, g_ in F.groupby("arm"):
        print(f"  {arm:8} n={len(g_):4d} "
              f"{g_.verdict.value_counts(normalize=True).round(3).to_dict()}", flush=True)
    # per-doc aggregates (matched arm)
    m = F[F.arm == "matched"]
    agg = m.groupby("url").verdict.agg(
        sourced_rate=lambda v: (v == "SOURCED").mean(),
        asserted_rate=lambda v: (v == "ASSERTED_ONLY").mean(),
        nf_rate=lambda v: (v == "NOT_FOUND").mean(), n_cl="size").reset_index()
    M2 = M.merge(agg, on="url", how="inner")
    print(f"\n[attr] {len(M2)} docs with aggregates; sourced_rate mean "
          f"{M2.sourced_rate.mean():.3f}", flush=True)
    # convergent validity vs CODE census
    dens = [c for c in M2.columns if c.endswith("_density")]
    for c in ["c_attrib_verbs", "c_attrib_density", "c_quote_density", "wordcount"]:
        if c in M2:
            print(f"  corr(sourced_rate, {c:18}) r={M2.sourced_rate.corr(M2[c]):+.3f}", flush=True)
    # race vs y_twitter
    y = M2.y_twitter.values
    g = (M2.outlet.astype(str) + "|" + M2.day.astype(str)).values
    print("\n=== attribution-adequacy vs twitter-y (CODE census benchmark .67-.69) ===", flush=True)
    gauc(M2, ["sourced_rate", "asserted_rate", "nf_rate", "n_cl"], y, g, "LLM attribution-adequacy")
    gauc(M2, ["sourced_rate"], y, g, "sourced_rate alone")
    counts = [c for c in M2.columns if c.startswith("c_") and not c.endswith("_density")]
    gauc(M2, counts + dens, y, g, "CODE census (same docs)")
    gauc(M2, ["sourced_rate", "asserted_rate"] + counts + dens, y, g, "LLM + CODE")
    if "wordcount" in M2:
        gauc(M2, ["wordcount"], y, g, "wordcount alone")
        M2["lq"] = pd.qcut(M2.wordcount, 5, labels=False, duplicates="drop")
        aucs = [roc_auc_score(s.y_twitter, s.sourced_rate) for _, s in M2.groupby("lq")
                if s.y_twitter.nunique() == 2 and len(s) > 40]
        print(f"  sourced_rate within length-quintile: {[round(a,3) for a in aucs]} "
              f"mean={np.mean(aucs):.4f}" if aucs else "  quintile audit: too few", flush=True)
    M2.to_csv(f"{ROOT}/outputs/attrib_adequacy/doc_metrics.csv", index=False)
    print("ATTRIB_ADEQ_DONE", flush=True)

if __name__ == "__main__":
    main()
